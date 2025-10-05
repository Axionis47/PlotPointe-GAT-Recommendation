#!/usr/bin/env python3
"""
LightGCN baseline training on interacted-only universe using BPR.
- Loads interactions.parquet and node_maps.json from GCS
- Chronological leave-last-1 test, second-last val per user (fallback to last-1 if <3)
- Trains LightGCN (embedding_dim L, K layers) with uniform negative sampling
- Logs sampled Recall@{10,20} and NDCG@{10,20} using 1000 negative samples/user
- Saves best checkpoint and metrics.json to GCS
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from google.cloud import storage


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    project_id: str
    region: str
    staging_prefix: str
    graphs_prefix: str
    models_prefix: str
    embed_dim: int = 64
    n_layers: int = 3
    lr: float = 1e-3
    l2: float = 1e-4
    epochs: int = 50
    batch_size: int = 8192
    neg_per_pos: int = 5
    seed: int = 42
    eval_neg_k: int = 1000


class LightGCN(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int, adj_indices: torch.Tensor, adj_values: torch.Tensor):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.E_user = torch.nn.Embedding(n_users, embed_dim)
        self.E_item = torch.nn.Embedding(n_items, embed_dim)
        torch.nn.init.normal_(self.E_user.weight, std=0.1)
        torch.nn.init.normal_(self.E_item.weight, std=0.1)
        self.register_buffer("adj_indices", adj_indices)  # 2 x nnz
        self.register_buffer("adj_values", adj_values)    # nnz
        self.embed_dim = embed_dim

    def propagate(self, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.n_users + self.n_items
        x0 = torch.cat([self.E_user.weight, self.E_item.weight], dim=0)
        out = x0
        acc = x0
        for _ in range(K):
            A = torch.sparse_coo_tensor(self.adj_indices, self.adj_values, size=(n, n))
            out = torch.sparse.mm(A, out)
            acc = acc + out
        acc = acc / (K + 1)
        users = acc[: self.n_users]
        items = acc[self.n_users :]
        return users, items

    def score(self, U: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        return (U * I).sum(dim=-1)


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


def build_splits(interactions: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    by_user = interactions.sort_values("ts").groupby("user_id")
    train_pos: Dict[int, np.ndarray] = {}
    val_pos: Dict[int, int] = {}
    test_pos: Dict[int, int] = {}
    for u, g in by_user:
        items = g["asin"].values
        if len(items) >= 3:
            train_pos[u] = items[:-2]
            val_pos[u] = items[-2]
            test_pos[u] = items[-1]
        elif len(items) >= 2:
            train_pos[u] = items[:-1]
            test_pos[u] = items[-1]
        else:
            # skip ultra-cold users from training/eval
            continue
    return train_pos, val_pos, test_pos


def to_indexed(interactions: pd.DataFrame, user_to_idx: Dict[str, int], item_to_idx: Dict[str, int]) -> pd.DataFrame:
    df = interactions[["user_id", "asin", "ts"]].copy()
    df["u"] = df["user_id"].map(user_to_idx)
    df["i"] = df["asin"].map(item_to_idx)
    # Drop rows with unknown mapping (should be none in interacted-only universe)
    df = df.dropna(subset=["u", "i"]).astype({"u": int, "i": int})
    return df


def build_adj(n_users: int, n_items: int, train_pos_idx: Dict[int, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Build bidirectional edges and normalized adjacency (LightGCN A_hat)
    rows = []
    cols = []
    for u_idx, items in train_pos_idx.items():
        for i_idx in items:
            rows.append(u_idx)
            cols.append(n_users + i_idx)
            rows.append(n_users + i_idx)
            cols.append(u_idx)
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    data = np.ones(len(rows), dtype=np.float32)
    # Symmetric normalization D^{-1/2} A D^{-1/2}
    n = n_users + n_items
    deg = np.zeros(n, dtype=np.float32)
    for r, c in zip(rows[::2], cols[::2]):
        deg[r] += 1
        deg[c] += 1
    # each undirected edge contributes 1 to both endpoints; we doubled edges
    deg = np.clip(deg, 1.0, None)
    norm_vals = []
    for r, c in zip(rows, cols):
        v = 1.0 / math.sqrt(float(deg[r]) * float(deg[c]))
        norm_vals.append(v)
    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    values = torch.tensor(np.array(norm_vals, dtype=np.float32))
    return indices, values


def sample_bpr_batches(train_pos_idx: Dict[int, np.ndarray], n_items: int, neg_per_pos: int, batch_size: int):
    users = list(train_pos_idx.keys())
    user_pos_sets = {u: set(pos) for u, pos in train_pos_idx.items()}
    u_list, i_list, j_list = [], [], []
    for u in users:
        pos_items = train_pos_idx[u]
        for i in pos_items:
            for _ in range(neg_per_pos):
                while True:
                    j = random.randrange(n_items)
                    if j not in user_pos_sets[u]:
                        break
                u_list.append(u)
                i_list.append(int(i))
                j_list.append(j)
                if len(u_list) >= batch_size:
                    yield np.array(u_list), np.array(i_list), np.array(j_list)
                    u_list, i_list, j_list = [], [], []
    if u_list:
        yield np.array(u_list), np.array(i_list), np.array(j_list)


def eval_sampled(model: LightGCN, cfg: Config, train_pos_idx: Dict[int, np.ndarray], eval_pos: Dict[int, int], Ks=(10, 20)):
    device = next(model.parameters()).device
    with torch.no_grad():
        U, I = model.propagate(cfg.n_layers)
        U = U.to(device)
        I = I.to(device)
        user_pos_sets = {u: set(pos) for u, pos in train_pos_idx.items()}
        metrics = {f"recall@{k}": [] for k in Ks}
        metrics.update({f"ndcg@{k}": [] for k in Ks})
        all_items = np.arange(model.n_items)
        for u, pos_i in eval_pos.items():
            # Sample negatives not seen in train
            avoid = user_pos_sets.get(u, set()) | {pos_i}
            negs = []
            while len(negs) < cfg.eval_neg_k:
                cand = np.random.randint(0, model.n_items)
                if cand not in avoid:
                    negs.append(cand)
            cand_items = np.array([pos_i] + negs, dtype=np.int64)
            u_emb = U[u]
            i_emb = I[cand_items]
            scores = (i_emb @ u_emb).cpu().numpy()
            # higher is better; rank position of pos (index 0)
            rank = (scores > scores[0]).sum() + 1  # 1-based
            for k in Ks:
                hit = 1.0 if rank <= k else 0.0
                metrics[f"recall@{k}"].append(hit)
                if hit:
                    metrics[f"ndcg@{k}"].append(1.0 / math.log2(rank + 1))
                else:
                    metrics[f"ndcg@{k}"].append(0.0)
        # aggregate
        out = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description="Train LightGCN baseline (BPR)")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--staging-prefix", default="gs://plotpointe-artifacts/staging/amazon_electronics")
    ap.add_argument("--graphs-prefix", default="gs://plotpointe-artifacts/graphs")
    ap.add_argument("--models-prefix", default="gs://plotpointe-artifacts/models/lightgcn")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--neg-per-pos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-neg-k", type=int, default=1000)
    args = ap.parse_args()

    cfg = Config(
        project_id=args.project_id,
        region=args.region,
        staging_prefix=args.staging_prefix,
        graphs_prefix=args.graphs_prefix,
        models_prefix=args.models_prefix,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        lr=args.lr,
        l2=args.l2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
        eval_neg_k=args.eval_neg_k,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LGCN] Device: {device}")

    # Download artifacts
    items_gcs = f"{cfg.staging_prefix}/items.parquet"
    inter_gcs = f"{cfg.staging_prefix}/interactions.parquet"
    maps_gcs = f"{cfg.graphs_prefix}/node_maps.json"
    print("[LGCN] Downloading inputs...")
    gcs_download(cfg.project_id, inter_gcs, "tmp/interactions.parquet")
    gcs_download(cfg.project_id, maps_gcs, "tmp/node_maps.json")

    interactions = pd.read_parquet("tmp/interactions.parquet")
    with open("tmp/node_maps.json") as f:
        maps = json.load(f)
    user_to_idx = {k: int(v) for k, v in maps["user_to_idx"].items()}
    item_to_idx = {k: int(v) for k, v in maps["item_to_idx"].items()}

    # Reindex interactions to idx space
    df_idx = to_indexed(interactions, user_to_idx, item_to_idx)

    # Build splits using raw ids then map to indices
    train_pos_raw, val_pos_raw, test_pos_raw = build_splits(interactions)
    # Map to indices (filter users/items that may have been dropped)
    train_pos_idx: Dict[int, np.ndarray] = {}
    val_pos_idx: Dict[int, int] = {}
    test_pos_idx: Dict[int, int] = {}
    for u_raw, items in train_pos_raw.items():
        u = user_to_idx.get(str(u_raw), None) if not isinstance(u_raw, str) else user_to_idx.get(u_raw, None)
        if u is None:
            continue
        idx_items = []
        for it in items:
            it_idx = item_to_idx.get(str(it), None) if not isinstance(it, str) else item_to_idx.get(it, None)
            if it_idx is not None:
                idx_items.append(it_idx)
        if idx_items:
            train_pos_idx[int(u)] = np.array(idx_items, dtype=np.int64)
    for d_raw, d_idx in [(val_pos_raw, val_pos_idx), (test_pos_raw, test_pos_idx)]:
        for u_raw, it in d_raw.items():
            u = user_to_idx.get(str(u_raw), None) if not isinstance(u_raw, str) else user_to_idx.get(u_raw, None)
            it_idx = item_to_idx.get(str(it), None) if not isinstance(it, str) else item_to_idx.get(it, None)
            if u is not None and it_idx is not None:
                d_idx[int(u)] = int(it_idx)

    n_users = int(maps["n_users"])  # 192,403
    n_items = int(maps["n_items"])  # 63,001
    print(f"[LGCN] n_users={n_users}, n_items={n_items}")

    # Build normalized adjacency for LightGCN
    print("[LGCN] Building normalized adjacency...")
    adj_idx, adj_vals = build_adj(n_users, n_items, train_pos_idx)

    model = LightGCN(n_users, n_items, cfg.embed_dim, adj_idx, adj_vals).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

    best_ndcg20 = -1.0
    best_path = "tmp/lightgcn_best.pt"
    metrics_path = "tmp/metrics.json"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for u_arr, i_arr, j_arr in sample_bpr_batches(train_pos_idx, n_items, cfg.neg_per_pos, cfg.batch_size):
            # Recompute propagation per batch to avoid autograd graph reuse
            U_emb, I_emb = model.propagate(cfg.n_layers)
            U_emb = U_emb.to(device)
            I_emb = I_emb.to(device)

            u = torch.from_numpy(u_arr).long().to(device)
            i = torch.from_numpy(i_arr).long().to(device)
            j = torch.from_numpy(j_arr).long().to(device)
            u_vec = U_emb[u]
            i_vec = I_emb[i]
            j_vec = I_emb[j]
            pos = (u_vec * i_vec).sum(dim=-1)
            neg = (u_vec * j_vec).sum(dim=-1)
            loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        print(f"[LGCN][Epoch {epoch}] train_bpr_loss={avg_loss:.4f}")

        # Eval on validation (sampled)
        model.eval()
        val_metrics = eval_sampled(model, cfg, train_pos_idx, val_pos_idx)
        print(f"[LGCN][Epoch {epoch}] val: {val_metrics}")
        if val_metrics.get("ndcg@20", 0.0) > best_ndcg20:
            best_ndcg20 = val_metrics.get("ndcg@20", 0.0)
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
            }, best_path)
            print("[LGCN] Saved new best checkpoint")

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_metrics = eval_sampled(model, cfg, train_pos_idx, test_pos_idx)
    print(f"[LGCN] test: {test_metrics}")

    out = {
        "best_val_ndcg@20": float(best_ndcg20),
        "val": val_metrics,
        "test": test_metrics,
        "config": cfg.__dict__,
        "notes": "Evaluation uses sampled 1000 negatives/user."
    }
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)

    # Upload artifacts
    run_id = f"lgcn_d{cfg.embed_dim}_{int(time.time())}"
    chk_gcs = f"{cfg.models_prefix}/checkpoints/{run_id}.pt"
    met_gcs = f"{cfg.models_prefix}/metrics_{run_id}.json"
    gcs_upload(cfg.project_id, best_path, chk_gcs)
    gcs_upload(cfg.project_id, metrics_path, met_gcs)
    print(f"[LGCN] ✅ Complete. Uploaded {chk_gcs} and {met_gcs}")


if __name__ == "__main__":
    main()

