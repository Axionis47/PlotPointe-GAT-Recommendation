#!/usr/bin/env python3
"""
Text embedding pipeline for items.
Reads items.parquet, encodes title+description with sentence-transformers, saves txt.npy + meta.json.
Logs to Vertex AI Experiments.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import aiplatform, storage
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="plotpointe")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--staging-prefix", default="gs://plotpointe-artifacts/staging/amazon_electronics")
    parser.add_argument("--output-prefix", default="gs://plotpointe-artifacts/embeddings")
    parser.add_argument("--experiment-name", default="recsys-dev")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(project=args.project_id, location=args.region, experiment=args.experiment_name)

    run_id = f"embed-text-{int(time.time())}"
    print(f"[EMBED_TEXT] Run ID: {run_id}")

    # Start Vertex Experiment run (optional - skip if experiment doesn't exist)
    try:
        aiplatform.start_run(run_id)
        aiplatform.log_params({
            "task": "embed_text",
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "staging_prefix": args.staging_prefix,
        })
        print(f"[EMBED_TEXT] Logging to experiment: {args.experiment_name}")
    except Exception as e:
        print(f"[EMBED_TEXT] Warning: Could not log to experiment: {e}")
        print(f"[EMBED_TEXT] Continuing without experiment tracking...")
    
    start_time = time.time()
    
    # Load items
    print(f"[EMBED_TEXT] Loading items from {args.staging_prefix}/items.parquet")
    items_path = args.staging_prefix.replace("gs://", "/gcs/") + "/items.parquet"
    if not Path(items_path).exists():
        # Download from GCS
        storage_client = storage.Client(project=args.project_id)
        bucket_name = args.staging_prefix.split("/")[2]
        blob_path = "/".join(args.staging_prefix.split("/")[3:]) + "/items.parquet"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        Path("tmp").mkdir(exist_ok=True)
        items_path = "tmp/items.parquet"
        blob.download_to_filename(items_path)
    
    items = pd.read_parquet(items_path)
    print(f"[EMBED_TEXT] Loaded {len(items)} items")

    # Prepare text: title + brand + categories (enhanced features)
    print(f"[EMBED_TEXT] Preparing text features...")
    text_parts = []

    # Always include title
    text_parts.append(items["title"].fillna(""))

    # Add brand if available
    if "brand" in items.columns:
        brand_text = items["brand"].fillna("").apply(lambda x: f"Brand: {x}" if x else "")
        text_parts.append(brand_text)
        print(f"[EMBED_TEXT] Including brand information")

    # Add categories if available
    if "categories" in items.columns:
        cat_text = items["categories"].apply(
            lambda x: " ".join(x) if isinstance(x, list) and x else ""
        )
        text_parts.append(cat_text)
        print(f"[EMBED_TEXT] Including category information")

    # Add description if available (though it doesn't exist in Amazon Electronics)
    if "description" in items.columns:
        text_parts.append(items["description"].fillna(""))
        print(f"[EMBED_TEXT] Including description")

    # Combine all parts
    items["text"] = pd.Series([" ".join(parts).strip() for parts in zip(*text_parts)])

    # Log sample
    print(f"[EMBED_TEXT] Sample text (first item): {items['text'].iloc[0][:200]}...")
    
    # Load encoder
    print(f"[EMBED_TEXT] Loading encoder: {args.model_name}")
    encoder = SentenceTransformer(args.model_name)
    embed_dim = encoder.get_sentence_embedding_dimension()
    print(f"[EMBED_TEXT] Embedding dimension: {embed_dim}")
    
    # Encode
    print(f"[EMBED_TEXT] Encoding {len(items)} items (batch_size={args.batch_size})...")
    encode_start = time.time()
    embeddings = encoder.encode(
        items["text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encode_time = time.time() - encode_start
    throughput = len(items) / encode_time
    print(f"[EMBED_TEXT] Encoded in {encode_time:.2f}s ({throughput:.1f} items/sec)")
    
    # Save locally
    Path("tmp").mkdir(exist_ok=True)
    txt_npy_path = "tmp/txt.npy"
    np.save(txt_npy_path, embeddings)
    print(f"[EMBED_TEXT] Saved embeddings to {txt_npy_path} (shape={embeddings.shape})")
    
    # Metadata
    meta = {
        "encoder_name": args.model_name,
        "embed_dim": int(embed_dim),
        "count": len(items),
        "throughput_items_per_sec": float(throughput),
        "encode_time_sec": float(encode_time),
        "run_id": run_id,
    }
    meta_path = "tmp/txt_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[EMBED_TEXT] Saved metadata to {meta_path}")
    
    # Upload to GCS
    storage_client = storage.Client(project=args.project_id)
    bucket_name = args.output_prefix.split("/")[2]
    bucket = storage_client.bucket(bucket_name)
    
    blob_npy = bucket.blob("embeddings/txt.npy")
    blob_npy.upload_from_filename(txt_npy_path)
    npy_uri = f"{args.output_prefix}/txt.npy"
    print(f"[EMBED_TEXT] Uploaded to {npy_uri}")
    
    blob_meta = bucket.blob("embeddings/txt_meta.json")
    blob_meta.upload_from_filename(meta_path)
    meta_uri = f"{args.output_prefix}/txt_meta.json"
    print(f"[EMBED_TEXT] Uploaded to {meta_uri}")
    
    total_time = time.time() - start_time

    # Log metrics to Vertex Experiments (if available)
    try:
        aiplatform.log_metrics({
            "count": len(items),
            "embed_dim": embed_dim,
            "throughput_items_per_sec": throughput,
            "encode_time_sec": encode_time,
            "total_time_sec": total_time,
        })
        aiplatform.log_params({"txt_npy_uri": npy_uri, "txt_meta_uri": meta_uri})
        aiplatform.end_run()
        print(f"[EMBED_TEXT] Logged metrics to experiment")
    except Exception as e:
        print(f"[EMBED_TEXT] Warning: Could not log metrics: {e}")
    
    print(f"[EMBED_TEXT] ✅ Complete")
    print(f"  Run ID: {run_id}")
    print(f"  Count: {len(items)}")
    print(f"  Dimension: {embed_dim}")
    print(f"  Throughput: {throughput:.1f} items/sec")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Artifacts: {npy_uri}, {meta_uri}")


if __name__ == "__main__":
    main()

