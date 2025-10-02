# Step 10 — Text & Image Embeddings Pipeline

## Overview
Generate three types of embeddings for items:
1. **Text embeddings** (txt.npy): Encode title+description using sentence-transformers (CPU)
2. **Image embeddings** (img.npy): Encode product images using CLIP (T4 GPU, preemptible, capped at ~120-150k items)
3. **Fused embeddings** (fused.npy): Concat text+image → MLP → 128d (CPU)

All runs logged to Vertex AI Experiments (`recsys-dev`).

---

## Pipeline Components

### 1. Text Embeddings (`embed_text.py`)
**Purpose:** Encode item text (title + description) into dense vectors.

**Configuration:**
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Input: `gs://plotpointe-artifacts/staging/amazon_electronics/items.parquet`
- Output: `gs://plotpointe-artifacts/embeddings/txt.npy` + `txt_meta.json`
- Machine: n1-standard-4 (CPU only)
- Batch size: 128

**Expected Output:**
- Count: ~498,196 items
- Dimension: 384
- Throughput: ~500-1000 items/sec (CPU)
- Total time: ~10-15 minutes

**Metadata (`txt_meta.json`):**
```json
{
  "encoder_name": "all-MiniLM-L6-v2",
  "embed_dim": 384,
  "count": 498196,
  "throughput_items_per_sec": 750.5,
  "encode_time_sec": 663.2,
  "run_id": "embed-text-1759344500"
}
```

---

### 2. Image Embeddings (`embed_image.py`)
**Purpose:** Encode product images using CLIP vision encoder.

**Configuration:**
- Model: `openai/clip-vit-base-patch32` (CLIP)
- Input: items with valid `image_url` (capped at 150k)
- Output: `gs://plotpointe-artifacts/embeddings/img.npy` + `img_meta.json` + `img_items.parquet`
- Machine: n1-standard-4 + NVIDIA_TESLA_T4 (preemptible)
- Batch size: 32
- Download timeout: 5s per image

**Expected Output:**
- Count: ~120-150k items (after filtering failed downloads)
- Failed: ~10-20% (network timeouts, invalid URLs)
- Dimension: 512
- Throughput: ~20-50 items/sec (GPU + download overhead)
- Total time: ~45-90 minutes

**Metadata (`img_meta.json`):**
```json
{
  "encoder_name": "openai/clip-vit-base-patch32",
  "embed_dim": 512,
  "count": 135420,
  "failed_count": 14580,
  "throughput_items_per_sec": 35.2,
  "encode_time_sec": 3847.1,
  "run_id": "embed-image-1759348100",
  "device": "cuda"
}
```

**Notes:**
- Uses SPOT (preemptible) T4 GPU for cost savings
- Saves `img_items.parquet` with ASINs of successfully encoded items
- Items without images will use zero vectors in fusion

---

### 3. Multimodal Fusion (`fuse_modal.py`)
**Purpose:** Fuse text and image embeddings into a unified 128d representation.

**Configuration:**
- Architecture: Concat(text, image) → MLP(hidden=256) → 128d
- Input: `txt.npy`, `img.npy`, `img_items.parquet`
- Output: `gs://plotpointe-artifacts/embeddings/fused.npy` + `fusion_config.json`
- Machine: n1-standard-4 (CPU)
- Training: 5 epochs, lr=1e-3, batch_size=512
- Loss: L2 norm regularization (normalized embeddings)

**Expected Output:**
- Count: ~498,196 items (all items, zero-padded for missing images)
- Items with images: ~135k
- Output dimension: 128
- Total time: ~10-20 minutes

**Fusion Config (`fusion_config.json`):**
```json
{
  "text_dim": 384,
  "img_dim": 512,
  "output_dim": 128,
  "hidden_dim": 256,
  "count": 498196,
  "items_with_images": 135420,
  "run_id": "fuse-modal-1759352000"
}
```

---

## Execution

### Run Full Pipeline
```bash
chmod +x scripts/run_embeddings_pipeline.sh
bash scripts/run_embeddings_pipeline.sh plotpointe us-central1
```

The script will:
1. Upload code to `gs://plotpointe-artifacts/code/embeddings/`
2. Submit text embeddings job (CPU)
3. Wait for completion, then submit image embeddings job (T4 preemptible)
4. Wait for completion, then submit fusion job (CPU)
5. Verify artifacts and display metadata

### Run Individual Steps
```bash
# Text only
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=embed-text-$(date +%s) \
  --config=vertex/configs/embed_text.yaml

# Image only (requires txt.npy to exist)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=embed-image-$(date +%s) \
  --config=vertex/configs/embed_image.yaml

# Fusion only (requires txt.npy and img.npy to exist)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=fuse-modal-$(date +%s) \
  --config=vertex/configs/fuse_modal.yaml
```

---

## Success Criteria

✅ **Three completed runs in Vertex Experiments:**
- `embed-text-*`: logged params (model_name, batch_size), metrics (count, throughput, encode_time_sec)
- `embed-image-*`: logged params (model_name, max_items), metrics (count, failed_count, throughput)
- `fuse-modal-*`: logged params (output_dim, hidden_dim), metrics (count, items_with_images)

✅ **Artifacts in GCS:**
- `gs://plotpointe-artifacts/embeddings/txt.npy` (shape: [498196, 384])
- `gs://plotpointe-artifacts/embeddings/txt_meta.json`
- `gs://plotpointe-artifacts/embeddings/img.npy` (shape: [~135k, 512])
- `gs://plotpointe-artifacts/embeddings/img_meta.json`
- `gs://plotpointe-artifacts/embeddings/img_items.parquet`
- `gs://plotpointe-artifacts/embeddings/fused.npy` (shape: [498196, 128])
- `gs://plotpointe-artifacts/embeddings/fusion_config.json`

✅ **Counts match:**
- Text: all items (~498k)
- Image: subset with valid images (~120-150k)
- Fused: all items (~498k, zero-padded for missing images)

✅ **Throughput & time documented:**
- Text: ~500-1000 items/sec, ~10-15 min
- Image: ~20-50 items/sec, ~45-90 min (GPU + download)
- Fusion: ~10-20 min

---

## Cost Estimates

- **Text embeddings (CPU):** ~$0.20-0.40 (n1-standard-4, 15 min)
- **Image embeddings (T4 preemptible):** ~$1.50-3.00 (n1-standard-4 + T4, 90 min, preemptible discount)
- **Fusion (CPU):** ~$0.20-0.40 (n1-standard-4, 20 min)
- **Total:** ~$2-4 per full pipeline run

---

## Verification Queries

```bash
# List artifacts
gsutil ls -lh gs://plotpointe-artifacts/embeddings/

# Check shapes
python3 << 'PY'
import numpy as np
from google.cloud import storage
client = storage.Client(project="plotpointe")
bucket = client.bucket("plotpointe-artifacts")

for name in ["txt.npy", "img.npy", "fused.npy"]:
    blob = bucket.blob(f"embeddings/{name}")
    blob.download_to_filename(f"tmp/{name}")
    arr = np.load(f"tmp/{name}")
    print(f"{name}: {arr.shape}, dtype={arr.dtype}")
PY

# View Vertex Experiments
gcloud ai experiments list --region=us-central1 --format="value(name)" | grep recsys-dev
```

---

## Next Steps (Step 11)
Once embeddings are complete, proceed to graph construction:
- U–I edges from interactions
- I–I kNN for text (k∈{15,20})
- I–I kNN for fused (k∈{15,20})
- Node maps and QA report

