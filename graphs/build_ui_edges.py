#!/usr/bin/env python3
"""
Build User-Item (U-I) bipartite graph edges from interactions.
Creates edge list and node mappings.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import storage
from scipy.sparse import coo_matrix, save_npz


def main():
    parser = argparse.ArgumentParser(description="Build U-I edges")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--staging-prefix", required=True, help="gs://bucket/path")
    parser.add_argument("--output-prefix", required=True, help="gs://bucket/path")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("[BUILD_UI] Starting U-I edge construction...")
    print(f"[BUILD_UI] Staging: {args.staging_prefix}")
    print(f"[BUILD_UI] Output: {args.output_prefix}")
    
    # Download interactions
    print("[BUILD_UI] Downloading interactions...")
    storage_client = storage.Client(project=args.project_id)
    bucket_name = args.staging_prefix.split("/")[2]
    blob_path = "/".join(args.staging_prefix.split("/")[3:]) + "/interactions.parquet"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    Path("tmp").mkdir(exist_ok=True)
    interactions_path = "tmp/interactions.parquet"
    blob.download_to_filename(interactions_path)
    
    interactions = pd.read_parquet(interactions_path)
    print(f"[BUILD_UI] Loaded {len(interactions)} interactions")
    print(f"[BUILD_UI] Columns: {list(interactions.columns)}")
    
    # Create user and item mappings
    print("[BUILD_UI] Creating node mappings...")
    unique_users = interactions["user_id"].unique()
    unique_items = interactions["asin"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    print(f"[BUILD_UI] Users: {n_users:,}")
    print(f"[BUILD_UI] Items: {n_items:,}")
    
    # Build edge list
    print("[BUILD_UI] Building edge list...")
    user_indices = interactions["user_id"].map(user_to_idx).values
    item_indices = interactions["asin"].map(item_to_idx).values
    
    # Create sparse adjacency matrix (users x items)
    # Values are 1 for interaction (could use ratings if available)
    values = np.ones(len(interactions), dtype=np.float32)
    
    ui_matrix = coo_matrix(
        (values, (user_indices, item_indices)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    print(f"[BUILD_UI] Edge count: {ui_matrix.nnz:,}")
    print(f"[BUILD_UI] Sparsity: {100 * (1 - ui_matrix.nnz / (n_users * n_items)):.4f}%")
    
    # Save locally
    ui_edges_path = "tmp/ui_edges.npz"
    save_npz(ui_edges_path, ui_matrix)
    print(f"[BUILD_UI] Saved to {ui_edges_path}")
    
    # Save node mappings
    node_maps = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": {idx: uid for uid, idx in user_to_idx.items()},
        "idx_to_item": {idx: iid for iid, idx in item_to_idx.items()},
        "n_users": n_users,
        "n_items": n_items,
    }
    
    node_maps_path = "tmp/node_maps.json"
    with open(node_maps_path, "w") as f:
        json.dump(node_maps, f, indent=2)
    print(f"[BUILD_UI] Saved node maps to {node_maps_path}")
    
    # Compute statistics
    user_degrees = np.array(ui_matrix.sum(axis=1)).flatten()
    item_degrees = np.array(ui_matrix.sum(axis=0)).flatten()
    
    stats = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_edges": int(ui_matrix.nnz),
        "sparsity": float(1 - ui_matrix.nnz / (n_users * n_items)),
        "user_degree_mean": float(user_degrees.mean()),
        "user_degree_std": float(user_degrees.std()),
        "user_degree_min": int(user_degrees.min()),
        "user_degree_max": int(user_degrees.max()),
        "item_degree_mean": float(item_degrees.mean()),
        "item_degree_std": float(item_degrees.std()),
        "item_degree_min": int(item_degrees.min()),
        "item_degree_max": int(item_degrees.max()),
        "build_time_sec": time.time() - start_time,
    }
    
    stats_path = "tmp/ui_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[BUILD_UI] Saved stats to {stats_path}")
    
    print("\n[BUILD_UI] Statistics:")
    print(f"  Edges: {stats['n_edges']:,}")
    print(f"  Sparsity: {stats['sparsity']*100:.4f}%")
    print(f"  Avg user degree: {stats['user_degree_mean']:.2f}")
    print(f"  Avg item degree: {stats['item_degree_mean']:.2f}")
    
    # Upload to GCS
    print("\n[BUILD_UI] Uploading to GCS...")
    output_bucket_name = args.output_prefix.split("/")[2]
    output_blob_prefix = "/".join(args.output_prefix.split("/")[3:])
    output_bucket = storage_client.bucket(output_bucket_name)
    
    # Upload edges
    blob_edges = output_bucket.blob(f"{output_blob_prefix}/ui_edges.npz")
    blob_edges.upload_from_filename(ui_edges_path)
    print(f"[BUILD_UI] Uploaded {args.output_prefix}/ui_edges.npz")
    
    # Upload node maps
    blob_maps = output_bucket.blob(f"{output_blob_prefix}/node_maps.json")
    blob_maps.upload_from_filename(node_maps_path)
    print(f"[BUILD_UI] Uploaded {args.output_prefix}/node_maps.json")
    
    # Upload stats
    blob_stats = output_bucket.blob(f"{output_blob_prefix}/ui_stats.json")
    blob_stats.upload_from_filename(stats_path)
    print(f"[BUILD_UI] Uploaded {args.output_prefix}/ui_stats.json")
    
    total_time = time.time() - start_time
    print(f"\n[BUILD_UI] ✅ Complete in {total_time:.2f}s")


if __name__ == "__main__":
    main()

