#!/usr/bin/env bash
# Parallel image embeddings across multiple GPUs
# Splits dataset into chunks and processes in parallel

set -e

PROJECT=${1:-plotpointe}
REGION=${2:-us-central1}
TOTAL_ITEMS=${3:-150000}

echo "============================================================"
echo "🚀 PARALLEL IMAGE EMBEDDINGS"
echo "============================================================"
echo "Project: $PROJECT"
echo "Region: $REGION"
echo "Total Items: $TOTAL_ITEMS"
echo "Strategy: Split into 4 chunks across 4 GPUs"
echo "============================================================"
echo ""

# Calculate chunk sizes
CHUNK_SIZE=$((TOTAL_ITEMS / 4))
echo "Chunk size: $CHUNK_SIZE items per GPU"
echo ""

# Upload code
echo "[SETUP] Uploading embedding scripts to GCS..."
gsutil cp embeddings/embed_image.py gs://plotpointe-artifacts/code/embeddings/embed_image.py

# Launch 4 parallel jobs
echo ""
echo "[LAUNCH] Starting 4 parallel jobs..."
echo ""

# Job 1: T4 on-demand (0-37500)
echo "Job 1: T4 (on-demand) - Items 0-37,500"
JOB1=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-image-chunk1-$(date +%s) \
  --config=vertex/configs/embed_image_chunk1.yaml \
  --format='value(name)')
echo "  Job ID: ${JOB1##*/}"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB1##*/}?project=$PROJECT"
echo ""

# Job 2: T4 preemptible (37500-75000)
echo "Job 2: T4 (preemptible) - Items 37,500-75,000"
JOB2=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-image-chunk2-$(date +%s) \
  --config=vertex/configs/embed_image_chunk2.yaml \
  --format='value(name)')
echo "  Job ID: ${JOB2##*/}"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB2##*/}?project=$PROJECT"
echo ""

# Job 3: L4 on-demand (75000-112500)
echo "Job 3: L4 (on-demand) - Items 75,000-112,500"
JOB3=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-image-chunk3-$(date +%s) \
  --config=vertex/configs/embed_image_chunk3.yaml \
  --format='value(name)')
echo "  Job ID: ${JOB3##*/}"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB3##*/}?project=$PROJECT"
echo ""

# Job 4: L4 preemptible (112500-150000)
echo "Job 4: L4 (preemptible) - Items 112,500-150,000"
JOB4=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-image-chunk4-$(date +%s) \
  --config=vertex/configs/embed_image_chunk4.yaml \
  --format='value(name)')
echo "  Job ID: ${JOB4##*/}"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB4##*/}?project=$PROJECT"
echo ""

echo "============================================================"
echo "✅ All 4 jobs launched!"
echo "============================================================"
echo ""
echo "Job IDs:"
echo "  Chunk 1 (T4):  ${JOB1##*/}"
echo "  Chunk 2 (T4p): ${JOB2##*/}"
echo "  Chunk 3 (L4):  ${JOB3##*/}"
echo "  Chunk 4 (L4p): ${JOB4##*/}"
echo ""
echo "Expected completion: ~2-2.5 hours"
echo ""
echo "Monitor progress:"
echo "  gcloud ai custom-jobs list --region=$REGION --filter='state:JOB_STATE_RUNNING'"
echo ""
echo "After all jobs complete, run merge script:"
echo "  bash scripts/merge_image_embeddings.sh"
echo ""

