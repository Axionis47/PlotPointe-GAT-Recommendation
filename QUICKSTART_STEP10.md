# Quick Start — Step 10: Embeddings Pipeline

## Prerequisites ✅
- Infrastructure complete (Steps 1-9)
- Amazon Electronics data staged in GCS
- GPU quotas: L4=2, T4 available (preemptible)

## Run the Full Pipeline

```bash
# Make script executable
chmod +x scripts/run_embeddings_pipeline.sh

# Execute (takes ~2 hours, costs ~$2-4)
bash scripts/run_embeddings_pipeline.sh plotpointe us-central1
```

The script will:
1. Upload code to GCS
2. Run text embeddings (CPU, ~15 min)
3. Run image embeddings (T4 preemptible, ~90 min)
4. Run multimodal fusion (CPU, ~20 min)
5. Verify artifacts and display metadata

## Expected Output

```
[EMBEDDINGS] Starting embeddings pipeline
Project: plotpointe, Region: us-central1
[EMBEDDINGS] Uploading code to GCS...
[EMBEDDINGS] Step 1/3: Text embeddings (CPU)...
  Job: projects/359145045403/locations/us-central1/customJobs/...
  Console: https://console.cloud.google.com/vertex-ai/...
  Waiting for text embeddings job...
  ✅ Text embeddings complete
[EMBEDDINGS] Step 2/3: Image embeddings (T4 preemptible)...
  Job: projects/359145045403/locations/us-central1/customJobs/...
  Console: https://console.cloud.google.com/vertex-ai/...
  Waiting for image embeddings job...
  ✅ Image embeddings complete
[EMBEDDINGS] Step 3/3: Multimodal fusion (CPU)...
  Job: projects/359145045403/locations/us-central1/customJobs/...
  Console: https://console.cloud.google.com/vertex-ai/...
  Waiting for fusion job...
  ✅ Fusion complete
[EMBEDDINGS] Verifying artifacts in GCS...
[EMBEDDINGS] ✅ Pipeline complete

=== Text Embeddings ===
{
  "encoder_name": "all-MiniLM-L6-v2",
  "embed_dim": 384,
  "count": 498196,
  "throughput_items_per_sec": 750.5,
  "encode_time_sec": 663.2,
  "run_id": "embed-text-..."
}

=== Image Embeddings ===
{
  "encoder_name": "openai/clip-vit-base-patch32",
  "embed_dim": 512,
  "count": 135420,
  "failed_count": 14580,
  "throughput_items_per_sec": 35.2,
  "encode_time_sec": 3847.1,
  "run_id": "embed-image-...",
  "device": "cuda"
}

=== Fusion Config ===
{
  "text_dim": 384,
  "img_dim": 512,
  "output_dim": 128,
  "hidden_dim": 256,
  "count": 498196,
  "items_with_images": 135420,
  "run_id": "fuse-modal-..."
}

Artifacts:
  gs://plotpointe-artifacts/embeddings/txt.npy
  gs://plotpointe-artifacts/embeddings/txt_meta.json
  gs://plotpointe-artifacts/embeddings/img.npy
  gs://plotpointe-artifacts/embeddings/img_meta.json
  gs://plotpointe-artifacts/embeddings/img_items.parquet
  gs://plotpointe-artifacts/embeddings/fused.npy
  gs://plotpointe-artifacts/embeddings/fusion_config.json
```

## Verify Success

### Check Vertex AI Experiments
```bash
gcloud ai experiments list --region=us-central1 --format="value(name)" | grep recsys-dev
```

### Check GCS Artifacts
```bash
gsutil ls -lh gs://plotpointe-artifacts/embeddings/
```

Expected files:
- `txt.npy` (~750 MB, shape: [498196, 384])
- `txt_meta.json`
- `img.npy` (~260 MB, shape: [~135k, 512])
- `img_meta.json`
- `img_items.parquet` (ASINs of items with images)
- `fused.npy` (~250 MB, shape: [498196, 128])
- `fusion_config.json`

### Verify Shapes
```bash
python3 << 'PY'
import numpy as np
from google.cloud import storage

client = storage.Client(project="plotpointe")
bucket = client.bucket("plotpointe-artifacts")

for name in ["txt.npy", "img.npy", "fused.npy"]:
    blob = bucket.blob(f"embeddings/{name}")
    blob.download_to_filename(f"tmp/{name}")
    arr = np.load(f"tmp/{name}")
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, norm={np.linalg.norm(arr[0]):.3f}")
PY
```

Expected output:
```
txt.npy: shape=(498196, 384), dtype=float32, norm=1.000
img.npy: shape=(135420, 512), dtype=float32, norm=1.000
fused.npy: shape=(498196, 128), dtype=float32, norm=1.000
```

## Troubleshooting

### Job Failed
```bash
# Check job status
gcloud ai custom-jobs list --region=us-central1 --filter='displayName~"embed"' --format='table(name,state,createTime)'

# View logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1
```

### Image Job Preempted (T4 SPOT)
The image embeddings job uses preemptible T4 GPUs and may be preempted. The job config includes `restartJobOnWorkerRestart: true`, so it will automatically restart. If it fails repeatedly, you can:
1. Wait and retry (preemption is random)
2. Reduce `--max-items` to speed up the job
3. Switch to on-demand T4 (edit `vertex/configs/embed_image.yaml`, remove `scheduling` section)

### Out of Memory
If fusion job runs out of memory:
```bash
# Edit vertex/configs/fuse_modal.yaml
# Change machineType: n1-standard-4 → n1-standard-8
```

## Run Individual Steps

### Text Only
```bash
gsutil -m cp embeddings/embed_text.py gs://plotpointe-artifacts/code/embeddings/
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=embed-text-$(date +%s) \
  --config=vertex/configs/embed_text.yaml
```

### Image Only (requires txt.npy)
```bash
gsutil -m cp embeddings/embed_image.py gs://plotpointe-artifacts/code/embeddings/
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=embed-image-$(date +%s) \
  --config=vertex/configs/embed_image.yaml
```

### Fusion Only (requires txt.npy and img.npy)
```bash
gsutil -m cp embeddings/fuse_modal.py gs://plotpointe-artifacts/code/embeddings/
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=fuse-modal-$(date +%s) \
  --config=vertex/configs/fuse_modal.yaml
```

## Cost Breakdown

- **Text embeddings:** n1-standard-4 × 15 min ≈ $0.30
- **Image embeddings:** (n1-standard-4 + T4 preemptible) × 90 min ≈ $2.50
- **Fusion:** n1-standard-4 × 20 min ≈ $0.40
- **Total:** ~$3.20 per full run

## Next Steps

After Step 10 completes successfully:
1. Review `docs/step10_embeddings_plan.md` for detailed results
2. Proceed to **Step 11: Graph Construction**
   - Build U–I edges from interactions
   - Build I–I kNN graphs from text and fused embeddings
   - Generate QA report

## Links

- **Vertex AI Console:** https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=plotpointe
- **Vertex Experiments:** https://console.cloud.google.com/vertex-ai/experiments?project=plotpointe
- **GCS Bucket:** https://console.cloud.google.com/storage/browser/plotpointe-artifacts/embeddings?project=plotpointe
- **Full Plan:** `docs/step10_embeddings_plan.md`
- **Progress Tracker:** `docs/PROGRESS.md`

