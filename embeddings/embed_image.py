#!/usr/bin/env python3
"""
Image embedding pipeline for items with valid image_url.
Caps at ~120-150k items. Uses CLIP or similar vision encoder.
Designed for T4 GPU (preemptible).
"""
import argparse
import json
import time
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from google.cloud import aiplatform, storage
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from tqdm import tqdm


def download_image(url, timeout=5):
    """Download image from URL with timeout."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="plotpointe")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--staging-prefix", default="gs://plotpointe-artifacts/staging/amazon_electronics")
    parser.add_argument("--output-prefix", default="gs://plotpointe-artifacts/embeddings")
    parser.add_argument("--experiment-name", default="recsys-dev")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=150000)
    parser.add_argument("--download-timeout", type=int, default=5)
    parser.add_argument("--chunk-start", type=int, default=0, help="Start index for chunk processing")
    parser.add_argument("--chunk-end", type=int, default=-1, help="End index for chunk processing (-1 = all)")
    parser.add_argument("--chunk-id", type=str, default="", help="Chunk identifier for output files")
    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(project=args.project_id, location=args.region, experiment=args.experiment_name)

    run_id = f"embed-image-{int(time.time())}"
    print(f"[EMBED_IMAGE] Run ID: {run_id}")

    # Start Vertex Experiment run
    aiplatform.start_run(run_id)
    aiplatform.log_params({
        "task": "embed_image",
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_items": args.max_items,
        "staging_prefix": args.staging_prefix,
    })
    
    start_time = time.time()
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EMBED_IMAGE] Device: {device}")
    if device == "cuda":
        print(f"[EMBED_IMAGE] GPU: {torch.cuda.get_device_name(0)}")
    
    # Load items
    print(f"[EMBED_IMAGE] Loading items from {args.staging_prefix}/items.parquet")
    storage_client = storage.Client(project=args.project_id)
    bucket_name = args.staging_prefix.split("/")[2]
    blob_path = "/".join(args.staging_prefix.split("/")[3:]) + "/items.parquet"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    Path("tmp").mkdir(exist_ok=True)
    items_path = "tmp/items.parquet"
    blob.download_to_filename(items_path)
    
    items = pd.read_parquet(items_path)
    print(f"[EMBED_IMAGE] Loaded {len(items)} items")
    
    # Filter items with valid image_url
    items = items[items["image_url"].notna() & (items["image_url"] != "")]
    print(f"[EMBED_IMAGE] Items with image_url: {len(items)}")

    # Apply chunk filtering if specified
    if args.chunk_start > 0 or args.chunk_end > 0:
        chunk_end = args.chunk_end if args.chunk_end > 0 else len(items)
        items = items.iloc[args.chunk_start:chunk_end].reset_index(drop=True)
        print(f"[EMBED_IMAGE] Processing chunk: items {args.chunk_start} to {chunk_end} ({len(items)} items)")

    # Cap at max_items (after chunking)
    elif len(items) > args.max_items:
        items = items.sample(n=args.max_items, random_state=42)
        print(f"[EMBED_IMAGE] Capped to {args.max_items} items")
    
    # Load CLIP model
    print(f"[EMBED_IMAGE] Loading model: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    embed_dim = model.config.projection_dim
    print(f"[EMBED_IMAGE] Embedding dimension: {embed_dim}")
    
    # Download and encode images
    print(f"[EMBED_IMAGE] Downloading and encoding {len(items)} images...")
    embeddings_list = []
    valid_asins = []
    failed_count = 0

    encode_start = time.time()

    with torch.no_grad():
        for idx, row in tqdm(items.iterrows(), total=len(items)):
            img = download_image(row["image_url"], timeout=args.download_timeout)
            if img is None:
                failed_count += 1
                continue

            try:
                inputs = processor(images=img, return_tensors="pt").to(device)
                image_features = model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings_list.append(image_features.cpu().numpy()[0])
                valid_asins.append(row["asin"])
            except Exception as e:
                failed_count += 1
                continue
    
    encode_time = time.time() - encode_start
    
    embeddings = np.array(embeddings_list)
    valid_count = len(embeddings)
    throughput = valid_count / encode_time if encode_time > 0 else 0

    print(f"[EMBED_IMAGE] Encoded {valid_count} images in {encode_time:.2f}s ({throughput:.1f} items/sec)")
    print(f"[EMBED_IMAGE] Failed: {failed_count} ({failed_count/(valid_count+failed_count)*100:.1f}%)")

    # Normalize embeddings (ensure unit norm for cosine similarity)
    print(f"[EMBED_IMAGE] Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    mean_norm = np.linalg.norm(embeddings, axis=1).mean()
    print(f"[EMBED_IMAGE] Mean norm after normalization: {mean_norm:.4f}")

    # Determine output file names (with chunk ID if specified)
    chunk_suffix = f"_{args.chunk_id}" if args.chunk_id else ""

    # Save locally
    img_npy_path = f"tmp/img{chunk_suffix}.npy"
    np.save(img_npy_path, embeddings)
    print(f"[EMBED_IMAGE] Saved embeddings to {img_npy_path} (shape={embeddings.shape})")

    # Save valid item ASINs
    valid_items_df = pd.DataFrame({"asin": valid_asins})
    items_parquet_path = f"tmp/img_items{chunk_suffix}.parquet"
    valid_items_df.to_parquet(items_parquet_path, index=False)

    # Metadata
    meta = {
        "encoder_name": args.model_name,
        "embed_dim": int(embed_dim),
        "count": int(valid_count),
        "failed_count": int(failed_count),
        "throughput_items_per_sec": float(throughput),
        "encode_time_sec": float(encode_time),
        "run_id": run_id,
        "device": device,
        "chunk_id": args.chunk_id if args.chunk_id else "full",
        "chunk_start": args.chunk_start,
        "chunk_end": args.chunk_end,
    }
    meta_path = f"tmp/img_meta{chunk_suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[EMBED_IMAGE] Saved metadata to {meta_path}")

    # Upload to GCS
    blob_npy = bucket.blob(f"embeddings/img{chunk_suffix}.npy")
    blob_npy.upload_from_filename(img_npy_path)
    npy_uri = f"{args.output_prefix}/img{chunk_suffix}.npy"
    print(f"[EMBED_IMAGE] Uploaded to {npy_uri}")

    blob_meta = bucket.blob(f"embeddings/img_meta{chunk_suffix}.json")
    blob_meta.upload_from_filename(meta_path)
    meta_uri = f"{args.output_prefix}/img_meta{chunk_suffix}.json"
    print(f"[EMBED_IMAGE] Uploaded to {meta_uri}")
    
    blob_items = bucket.blob(f"embeddings/img_items{chunk_suffix}.parquet")
    blob_items.upload_from_filename(items_parquet_path)
    items_uri = f"{args.output_prefix}/img_items{chunk_suffix}.parquet"
    print(f"[EMBED_IMAGE] Uploaded to {items_uri}")
    
    total_time = time.time() - start_time
    
    # Log metrics to Vertex Experiments
    aiplatform.log_metrics({
        "count": valid_count,
        "failed_count": failed_count,
        "embed_dim": embed_dim,
        "throughput_items_per_sec": throughput,
        "encode_time_sec": encode_time,
        "total_time_sec": total_time,
    })
    
    # Log artifacts
    aiplatform.log_params({
        "img_npy_uri": npy_uri,
        "img_meta_uri": meta_uri,
        "img_items_uri": items_uri,
    })
    
    aiplatform.end_run()
    
    print(f"[EMBED_IMAGE] ✅ Complete")
    print(f"  Run ID: {run_id}")
    print(f"  Count: {valid_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Dimension: {embed_dim}")
    print(f"  Throughput: {throughput:.1f} items/sec")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Artifacts: {npy_uri}, {meta_uri}, {items_uri}")


if __name__ == "__main__":
    main()

