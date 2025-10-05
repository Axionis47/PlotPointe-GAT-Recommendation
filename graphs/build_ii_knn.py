#!/usr/bin/env python3
"""
Build Item-Item (I-I) kNN graph from embeddings.
Uses cosine similarity to find k nearest neighbors for each item.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from google.cloud import storage
from scipy.sparse import coo_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser(description="Build I-I kNN graph")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--embeddings-path", required=True, help="gs://bucket/path/embeddings.npy")
    parser.add_argument("--output-prefix", required=True, help="gs://bucket/path")
    parser.add_argument("--output-name", required=True, help="Output filename (e.g., ii_edges_txt)")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum similarity threshold")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for similarity computation")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"[BUILD_II_KNN] Starting I-I kNN construction...")
    print(f"[BUILD_II_KNN] Embeddings: {args.embeddings_path}")
    print(f"[BUILD_II_KNN] Output: {args.output_prefix}/{args.output_name}.npz")
    print(f"[BUILD_II_KNN] k: {args.k}")
    print(f"[BUILD_II_KNN] Min similarity: {args.min_similarity}")
    
    # Download embeddings
    print("[BUILD_II_KNN] Downloading embeddings...")
    storage_client = storage.Client(project=args.project_id)
    bucket_name = args.embeddings_path.split("/")[2]
    blob_path = "/".join(args.embeddings_path.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    Path("tmp").mkdir(exist_ok=True)
    embeddings_local_path = "tmp/embeddings.npy"
    blob.download_to_filename(embeddings_local_path)
    
    embeddings = np.load(embeddings_local_path)
    n_items, embed_dim = embeddings.shape
    print(f"[BUILD_II_KNN] Loaded embeddings: {n_items:,} items × {embed_dim} dims")
    
    # Normalize embeddings for cosine similarity
    print("[BUILD_II_KNN] Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)
    
    # Compute kNN in batches
    print(f"[BUILD_II_KNN] Computing kNN (batch_size={args.batch_size})...")
    all_rows = []
    all_cols = []
    all_sims = []
    
    n_batches = (n_items + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, n_items)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx + 1}/{n_batches} (items {start_idx:,}-{end_idx:,})")
        
        # Compute similarities for this batch
        batch_embeddings = embeddings_norm[start_idx:end_idx]
        similarities = cosine_similarity(batch_embeddings, embeddings_norm)
        
        # For each item in batch, find top-k neighbors (excluding self)
        for i, item_idx in enumerate(range(start_idx, end_idx)):
            sims = similarities[i]

            # Set self-similarity to -inf to exclude it
            sims[item_idx] = -np.inf

            # Get top-k indices
            top_k_indices = np.argpartition(sims, -args.k)[-args.k:]
            top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]
            top_k_sims = sims[top_k_indices]

            # Filter by similarity threshold (remove weak connections)
            valid_mask = top_k_sims >= args.min_similarity
            top_k_indices = top_k_indices[valid_mask]
            top_k_sims = top_k_sims[valid_mask]

            # Add edges (variable number per item after filtering)
            if len(top_k_indices) > 0:
                all_rows.extend([item_idx] * len(top_k_indices))
                all_cols.extend(top_k_indices)
                all_sims.extend(top_k_sims)
    
    # Create sparse matrix
    print("[BUILD_II_KNN] Creating sparse matrix...")
    rows = np.array(all_rows, dtype=np.int32)
    cols = np.array(all_cols, dtype=np.int32)
    sims = np.array(all_sims, dtype=np.float32)
    
    ii_matrix = coo_matrix(
        (sims, (rows, cols)),
        shape=(n_items, n_items),
        dtype=np.float32
    )
    
    print(f"[BUILD_II_KNN] Edge count: {ii_matrix.nnz:,}")
    print(f"[BUILD_II_KNN] Avg degree: {ii_matrix.nnz / n_items:.2f}")
    print(f"[BUILD_II_KNN] Avg similarity: {sims.mean():.4f}")
    print(f"[BUILD_II_KNN] Min similarity: {sims.min():.4f}")
    print(f"[BUILD_II_KNN] Max similarity: {sims.max():.4f}")
    
    # Save locally
    output_path = f"tmp/{args.output_name}.npz"
    save_npz(output_path, ii_matrix)
    print(f"[BUILD_II_KNN] Saved to {output_path}")
    
    # Save stats
    stats = {
        "n_items": int(n_items),
        "embed_dim": int(embed_dim),
        "k": args.k,
        "n_edges": int(ii_matrix.nnz),
        "avg_degree": float(ii_matrix.nnz / n_items),
        "avg_similarity": float(sims.mean()),
        "min_similarity": float(sims.min()),
        "max_similarity": float(sims.max()),
        "build_time_sec": time.time() - start_time,
    }
    
    stats_path = f"tmp/{args.output_name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[BUILD_II_KNN] Saved stats to {stats_path}")
    
    # Upload to GCS
    print("[BUILD_II_KNN] Uploading to GCS...")
    output_bucket_name = args.output_prefix.split("/")[2]
    output_blob_prefix = "/".join(args.output_prefix.split("/")[3:])
    output_bucket = storage_client.bucket(output_bucket_name)
    
    # Upload edges
    blob_edges = output_bucket.blob(f"{output_blob_prefix}/{args.output_name}.npz")
    blob_edges.upload_from_filename(output_path)
    print(f"[BUILD_II_KNN] Uploaded {args.output_prefix}/{args.output_name}.npz")
    
    # Upload stats
    blob_stats = output_bucket.blob(f"{output_blob_prefix}/{args.output_name}_stats.json")
    blob_stats.upload_from_filename(stats_path)
    print(f"[BUILD_II_KNN] Uploaded {args.output_prefix}/{args.output_name}_stats.json")
    
    total_time = time.time() - start_time
    print(f"\n[BUILD_II_KNN] ✅ Complete in {total_time:.2f}s")


if __name__ == "__main__":
    main()

