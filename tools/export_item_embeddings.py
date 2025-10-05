#!/usr/bin/env python3
"""
Export item embedding matrix from a trained GAT checkpoint.
Requires the same inputs as training (node_maps.json, features.npy, and the train_pos graph to build edge_index).
Writes item_embeddings.npy to local path or GCS.

Example:
python tools/export_item_embeddings.py \
  --model-family gat_pyg \
  --checkpoint gs://plotpointe-artifacts/models/gat/pyg/checkpoints/gat_pyg_d128_...pt \
  --staging-prefix gs://plotpointe-artifacts/staging/amazon_electronics \
  --embeddings-prefix gs://plotpointe-artifacts/embeddings \
  --graphs-prefix gs://plotpointe-artifacts/graphs \
  --item-features fused \
  --project-id plotpointe \
  --out-gcs gs://plotpointe-artifacts/models/gat/pyg/exports/item_embeddings.npy
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from google.cloud import storage

# Import trainers' components lazily
from scripts.train_gat_pyg import PyGGAT as PyGGATModel, build_splits as build_splits_pyg, to_indexed as to_indexed_pyg, build_edge_index as build_edge_index_pyg, Config as PyGConfig
from scripts.train_gat_custom import CustomGAT as CustomModel, build_splits as build_splits_custom, to_indexed as to_indexed_custom, build_edge_index as build_edge_index_custom, Config as CustomConfig


def gcs_download(project_id: str, gcs_uri: str, local_path: str) -> None:
    storage_client = storage.Client(project=project_id)
    bucket_name = gcs_uri.split("/")[2]
    blob_path = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)


def gcs_upload(project_id: str, local_path: str, gcs_uri: str) -> None:
    storage_client = storage.Client(project=project_id)
    bucket_name = gcs_uri.split("/")[2]
    blob_path = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def main():
    ap = argparse.ArgumentParser(description="Export item embeddings from GAT checkpoint")
    ap.add_argument("--model-family", choices=["gat_pyg", "gat_custom"], required=True)
    ap.add_argument("--checkpoint", required=True, help="Local or GCS path to checkpoint .pt")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--staging-prefix", required=True)
    ap.add_argument("--graphs-prefix", required=True)
    ap.add_argument("--embeddings-prefix", required=True)
    ap.add_argument("--item-features", choices=["fused", "txt"], default="fused")
    ap.add_argument("--out-local", default="tmp/item_embeddings.npy")
    ap.add_argument("--out-gcs", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download inputs
    inter_gcs = f"{args.staging_prefix}/interactions.parquet"
    maps_gcs = f"{args.graphs_prefix}/node_maps.json"
    feat_name = "fused_interacted.npy" if args.item_features == "fused" else "txt_interacted.npy"
    feat_gcs = f"{args.embeddings_prefix}/{feat_name}"
    print("[EXPORT] Downloading inputs...")
    gcs_download(args.project_id, inter_gcs, "tmp/interactions.parquet")
    gcs_download(args.project_id, maps_gcs, "tmp/node_maps.json")
    gcs_download(args.project_id, feat_gcs, "tmp/features.npy")

    interactions = None
    try:
        import pandas as pd
        interactions = pd.read_parquet("tmp/interactions.parquet")
    except Exception:
        raise RuntimeError("pandas/pyarrow required to read parquet for splits")

    with open("tmp/node_maps.json") as f:
        maps = json.load(f)

    user_to_idx = {k: int(v) for k, v in maps["user_to_idx"].items()}
    item_to_idx = {k: int(v) for k, v in maps["item_to_idx"].items()}

    # Build splits and edge index
    if args.model_family == "gat_pyg":
        train_pos_raw, _, _ = build_splits_pyg(interactions)
    else:
        train_pos_raw, _, _ = build_splits_custom(interactions)

    train_pos_idx: Dict[int, np.ndarray] = {}
    for u_raw, items in train_pos_raw.items():
        u = user_to_idx.get(str(u_raw), user_to_idx.get(u_raw, None))
        if u is None:
            continue
        idx_items = []
        for it in items:
            it_idx = item_to_idx.get(str(it), item_to_idx.get(it, None))
            if it_idx is not None:
                idx_items.append(it_idx)
        if idx_items:
            train_pos_idx[int(u)] = np.array(idx_items, dtype=np.int64)

    n_users = int(maps["n_users"]) ; n_items = int(maps["n_items"]) 

    if args.model_family == "gat_pyg":
        edge_index = build_edge_index_pyg(n_users, n_items, train_pos_idx).to(device)
    else:
        edge_index = build_edge_index_custom(n_users, n_items, train_pos_idx).to(device)

    item_feats = torch.tensor(np.load("tmp/features.npy"), dtype=torch.float32).to(device)
    assert item_feats.shape[0] == n_items

    # Load model
    print("[EXPORT] Loading checkpoint...")
    ckpt_local = args.checkpoint
    if args.checkpoint.startswith("gs://"):
        ckpt_local = "tmp/model.pt"
        gcs_download(args.project_id, args.checkpoint, ckpt_local)
    ckpt = torch.load(ckpt_local, map_location=device)

    if args.model_family == "gat_pyg":
        cfg = PyGConfig(project_id=args.project_id, region="us-central1", staging_prefix=args.staging_prefix,
                        graphs_prefix=args.graphs_prefix, embeddings_prefix=args.embeddings_prefix, models_prefix="")
        model = PyGGATModel(n_users, n_items, item_feat_dim=item_feats.size(1), hidden=ckpt["config"].get("hidden_dim", 128),
                            layers=ckpt["config"].get("layers", 2), heads=ckpt["config"].get("heads", 1),
                            attn_dropout=ckpt["config"].get("attn_dropout", 0.1)).to(device)
    else:
        cfg = CustomConfig(project_id=args.project_id, region="us-central1", staging_prefix=args.staging_prefix,
                           graphs_prefix=args.graphs_prefix, embeddings_prefix=args.embeddings_prefix, models_prefix="")
        model = CustomModel(n_users, n_items, item_feat_dim=item_feats.size(1), hidden=ckpt["config"].get("hidden_dim", 128),
                            layers=ckpt["config"].get("layers", 2)).to(device)

    model.load_state_dict(ckpt["state_dict"]) ; model.eval()
    with torch.no_grad():
        Z = model(item_feats, edge_index)
        I = Z[n_users:].detach().cpu().numpy().astype(np.float32)

    Path(args.out_local).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_local, I)
    print(f"[EXPORT] Wrote item embeddings: {args.out_local} shape={I.shape}")

    if args.out_gcs:
        gcs_upload(args.project_id, args.out_local, args.out_gcs)
        print(f"[EXPORT] Uploaded to: {args.out_gcs}")


if __name__ == "__main__":
    main()

