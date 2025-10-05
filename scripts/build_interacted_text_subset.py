#!/usr/bin/env python3
"""
Build a subset of item embeddings containing only items that appear in interactions ("interacted-only").
- Reads items.parquet and interactions.parquet from the staging prefix
- Loads full embeddings .npy (text by default)
- Filters rows to those items with at least one interaction (preserving items.parquet order)
- Writes txt_interacted.npy and interacted_items.json to the embeddings output prefix
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import storage


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
    ap = argparse.ArgumentParser(description="Build interacted-only embeddings subset")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--staging-prefix", default="gs://plotpointe-artifacts/staging/amazon_electronics")
    ap.add_argument("--embeddings-uri", default="gs://plotpointe-artifacts/embeddings/txt.npy",
                    help="GCS URI to the full embeddings .npy to filter (e.g., txt.npy)")
    ap.add_argument("--output-prefix", default="gs://plotpointe-artifacts/embeddings",
                    help="GCS prefix where subset files will be written")
    ap.add_argument("--output-name", default="txt_interacted.npy",
                    help="Output .npy filename under output-prefix (e.g., txt_interacted.npy or fused_interacted.npy)")
    args = ap.parse_args()

    items_gcs = f"{args.staging_prefix}/items.parquet"
    inter_gcs = f"{args.staging_prefix}/interactions.parquet"

    print(f"[SUBSET] Downloading items: {items_gcs}")
    gcs_download(args.project_id, items_gcs, "tmp/items.parquet")
    print(f"[SUBSET] Downloading interactions: {inter_gcs}")
    gcs_download(args.project_id, inter_gcs, "tmp/interactions.parquet")

    items = pd.read_parquet("tmp/items.parquet")
    interactions = pd.read_parquet("tmp/interactions.parquet")
    print(f"[SUBSET] Loaded items={len(items):,}, interactions={len(interactions):,}")

    interacted_asins = pd.Index(interactions["asin"].unique())
    print(f"[SUBSET] Unique interacted items: {len(interacted_asins):,}")

    # Build boolean mask over items in original order
    mask = items["asin"].isin(interacted_asins).values
    kept_count = int(mask.sum())
    print(f"[SUBSET] Keeping {kept_count:,} items ({kept_count/len(items)*100:.2f}% of catalog)")

    # Download embeddings and filter rows accordingly
    print(f"[SUBSET] Downloading embeddings: {args.embeddings_uri}")
    gcs_download(args.project_id, args.embeddings_uri, "tmp/embeddings.npy")

    emb = np.load("tmp/embeddings.npy", mmap_mode="r")
    if emb.shape[0] != len(items):
        raise SystemExit(f"Embeddings row count {emb.shape[0]} does not match items {len(items)}")

    subset = np.array(emb[mask])  # materialize to contiguous array
    print(f"[SUBSET] Subset shape: {subset.shape}")

    # Save subset and the ordered asin list
    Path("tmp").mkdir(exist_ok=True)
    subset_path = f"tmp/{args.output_name}"
    json_path = "tmp/interacted_items.json"

    np.save(subset_path, subset)
    interacted_items_ordered = items.loc[mask, "asin"].tolist()
    with open(json_path, "w") as f:
        json.dump({"count": kept_count, "asins": interacted_items_ordered}, f)

    out_subset_gcs = f"{args.output_prefix}/{args.output_name}"
    out_items_gcs = f"{args.output_prefix}/interacted_items.json"

    print(f"[SUBSET] Uploading subset to {out_subset_gcs}")
    gcs_upload(args.project_id, subset_path, out_subset_gcs)
    print(f"[SUBSET] Uploading item list to {out_items_gcs}")
    gcs_upload(args.project_id, json_path, out_items_gcs)

    print("[SUBSET] ✅ Complete")


if __name__ == "__main__":
    main()

