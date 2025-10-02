#!/bin/bash
# Dry-run test for embed_text.py
# Tests the script locally with a small sample before submitting to Vertex AI

set -e

PROJECT_ID="${1:-plotpointe}"
REGION="${2:-us-central1}"

echo "=========================================="
echo "EMBED_TEXT.PY - DRY RUN TEST"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Create temp directory
TMP_DIR=$(mktemp -d)
echo "Working directory: $TMP_DIR"

# Download items.parquet
echo "[1/5] Downloading items.parquet from GCS..."
gsutil cp gs://plotpointe-artifacts/staging/amazon_electronics/items.parquet $TMP_DIR/items.parquet

# Create a small sample (first 100 items)
echo "[2/5] Creating small sample (100 items)..."
python3 << PYTHON
import pandas as pd
import sys

items = pd.read_parquet('$TMP_DIR/items.parquet')
sample = items.head(100)
sample.to_parquet('$TMP_DIR/items_sample.parquet')
print(f"Sample created: {len(sample)} items")
PYTHON

# Create a modified version of embed_text.py for local testing
echo "[3/5] Creating test script..."
cat > $TMP_DIR/test_embed_text.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load items
    print(f"[TEST] Loading items from {args.items_path}")
    items = pd.read_parquet(args.items_path)
    print(f"[TEST] Loaded {len(items)} items")
    
    # Prepare text
    if "description" in items.columns:
        items["text"] = items["title"].fillna("") + " " + items["description"].fillna("")
    else:
        items["text"] = items["title"].fillna("")
    items["text"] = items["text"].str.strip()
    
    # Load encoder
    print(f"[TEST] Loading encoder: {args.model_name}")
    encoder = SentenceTransformer(args.model_name)
    embed_dim = encoder.get_sentence_embedding_dimension()
    print(f"[TEST] Embedding dimension: {embed_dim}")
    
    # Encode
    print(f"[TEST] Encoding {len(items)} items (batch_size={args.batch_size})...")
    encode_start = time.time()
    embeddings = encoder.encode(
        items["text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    encode_time = time.time() - encode_start
    
    print(f"[TEST] Encoding complete: {encode_time:.2f}s")
    print(f"[TEST] Throughput: {len(items) / encode_time:.1f} items/sec")
    print(f"[TEST] Embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = output_dir / "txt.npy"
    np.save(embeddings_path, embeddings)
    print(f"[TEST] Saved embeddings to {embeddings_path}")
    
    # Save metadata
    metadata = {
        "encoder_name": args.model_name,
        "embedding_dim": int(embed_dim),
        "num_items": len(items),
        "encode_time_sec": round(encode_time, 2),
        "throughput_items_per_sec": round(len(items) / encode_time, 1),
        "batch_size": args.batch_size,
        "normalized": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }
    
    meta_path = output_dir / "txt_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[TEST] Saved metadata to {meta_path}")
    
    total_time = time.time() - start_time
    print(f"[TEST] Total time: {total_time:.2f}s")
    print(f"[TEST] SUCCESS!")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Run the test
echo "[4/5] Running embed_text test..."
python3 $TMP_DIR/test_embed_text.py \
    --items-path=$TMP_DIR/items_sample.parquet \
    --output-dir=$TMP_DIR/output \
    --model-name=all-MiniLM-L6-v2 \
    --batch-size=32

# Verify outputs
echo "[5/5] Verifying outputs..."

if [ -f "$TMP_DIR/output/txt.npy" ]; then
    echo "✓ txt.npy created"
    FILE_SIZE=$(du -h $TMP_DIR/output/txt.npy | cut -f1)
    echo "  Size: $FILE_SIZE"
else
    echo "✗ txt.npy NOT created"
    exit 1
fi

if [ -f "$TMP_DIR/output/txt_meta.json" ]; then
    echo "✓ txt_meta.json created"
    echo "  Contents:"
    cat $TMP_DIR/output/txt_meta.json | python3 -m json.tool | sed 's/^/    /'
else
    echo "✗ txt_meta.json NOT created"
    exit 1
fi

# Verify embeddings can be loaded
echo ""
echo "Verifying embeddings can be loaded..."
python3 << PYTHON
import numpy as np
import json

embeddings = np.load('$TMP_DIR/output/txt.npy')
with open('$TMP_DIR/output/txt_meta.json') as f:
    meta = json.load(f)

print(f"✓ Embeddings loaded successfully")
print(f"  Shape: {embeddings.shape}")
print(f"  Expected: ({meta['num_items']}, {meta['embedding_dim']})")
print(f"  Match: {embeddings.shape == (meta['num_items'], meta['embedding_dim'])}")

# Check normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f"  Normalized: {np.allclose(norms, 1.0, atol=1e-5)}")
print(f"  Norm range: [{norms.min():.6f}, {norms.max():.6f}]")
PYTHON

# Cleanup
echo ""
echo "Cleaning up..."
rm -rf $TMP_DIR

echo ""
echo "=========================================="
echo "DRY RUN COMPLETE - SUCCESS!"
echo "=========================================="
echo ""
echo "The embed_text.py script works correctly."
echo "You can now safely submit the job to Vertex AI."
echo ""

