#!/usr/bin/env bash
# Run the full embeddings pipeline: text → image → fuse
set -e

PROJECT=${1:-plotpointe}
REGION=${2:-us-central1}

echo "[EMBEDDINGS] Starting embeddings pipeline"
echo "Project: $PROJECT, Region: $REGION"

# Upload code to GCS
echo "[EMBEDDINGS] Uploading code to GCS..."
gsutil -m cp embeddings/*.py gs://plotpointe-artifacts/code/embeddings/

# Step 1: Text embeddings (CPU)
echo "[EMBEDDINGS] Step 1/3: Text embeddings (CPU)..."
JOB_TEXT=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-text-$(date +%s) \
  --config=vertex/configs/embed_text.yaml \
  --format='value(name)')

echo "  Job: $JOB_TEXT"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB_TEXT##*/}?project=$PROJECT"

# Wait for text embeddings to complete
echo "  Waiting for text embeddings job..."
echo "  (Polling every 30 seconds...)"
while true; do
  TEXT_STATE=$(gcloud ai custom-jobs describe $JOB_TEXT --region=$REGION --format='value(state)' 2>/dev/null || echo "UNKNOWN")
  echo "  [$(date +%H:%M:%S)] Job state: $TEXT_STATE"

  if [[ "$TEXT_STATE" == "JOB_STATE_SUCCEEDED" ]]; then
    echo "  ✅ Text embeddings complete"
    break
  elif [[ "$TEXT_STATE" == "JOB_STATE_FAILED" ]]; then
    echo "  ❌ Text embeddings job failed"
    exit 1
  elif [[ "$TEXT_STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo "  ❌ Text embeddings job cancelled"
    exit 1
  elif [[ "$TEXT_STATE" == "UNKNOWN" ]]; then
    echo "  ⚠️  Could not get job state, retrying..."
  fi

  sleep 30
done

# Step 2: Image embeddings (L4)
echo "[EMBEDDINGS] Step 2/3: Image embeddings (L4)..."
JOB_IMAGE=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-image-$(date +%s) \
  --config=vertex/configs/embed_image_l4.yaml \
  --format='value(name)')

echo "  Job: $JOB_IMAGE"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB_IMAGE##*/}?project=$PROJECT"

# Wait for image embeddings to complete
echo "  Waiting for image embeddings job..."
echo "  (Polling every 30 seconds...)"
while true; do
  IMAGE_STATE=$(gcloud ai custom-jobs describe $JOB_IMAGE --region=$REGION --format='value(state)' 2>/dev/null || echo "UNKNOWN")
  echo "  [$(date +%H:%M:%S)] Job state: $IMAGE_STATE"

  if [[ "$IMAGE_STATE" == "JOB_STATE_SUCCEEDED" ]]; then
    echo "  ✅ Image embeddings complete"
    break
  elif [[ "$IMAGE_STATE" == "JOB_STATE_FAILED" ]]; then
    echo "  ❌ Image embeddings job failed"
    exit 1
  elif [[ "$IMAGE_STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo "  ❌ Image embeddings job cancelled"
    exit 1
  elif [[ "$IMAGE_STATE" == "UNKNOWN" ]]; then
    echo "  ⚠️  Could not get job state, retrying..."
  fi

  sleep 30
done

# Step 3: Fusion (L4)
echo "[EMBEDDINGS] Step 3/3: Multimodal fusion (L4)..."
JOB_FUSE=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=fuse-modal-$(date +%s) \
  --config=vertex/configs/fuse_modal_l4.yaml \
  --format='value(name)')

echo "  Job: $JOB_FUSE"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/${JOB_FUSE##*/}?project=$PROJECT"

# Wait for fusion to complete
echo "  Waiting for fusion job..."
echo "  (Polling every 30 seconds...)"
while true; do
  FUSE_STATE=$(gcloud ai custom-jobs describe $JOB_FUSE --region=$REGION --format='value(state)' 2>/dev/null || echo "UNKNOWN")
  echo "  [$(date +%H:%M:%S)] Job state: $FUSE_STATE"

  if [[ "$FUSE_STATE" == "JOB_STATE_SUCCEEDED" ]]; then
    echo "  ✅ Fusion complete"
    break
  elif [[ "$FUSE_STATE" == "JOB_STATE_FAILED" ]]; then
    echo "  ❌ Fusion job failed"
    exit 1
  elif [[ "$FUSE_STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo "  ❌ Fusion job cancelled"
    exit 1
  elif [[ "$FUSE_STATE" == "UNKNOWN" ]]; then
    echo "  ⚠️  Could not get job state, retrying..."
  fi

  sleep 30
done

# Verify artifacts
echo "[EMBEDDINGS] Verifying artifacts in GCS..."
gsutil ls -lh gs://plotpointe-artifacts/embeddings/ | grep -E "(txt|img|fused)\.(npy|json|parquet)"

# Download and display metadata
echo "[EMBEDDINGS] Downloading metadata..."
mkdir -p tmp
gsutil cp gs://plotpointe-artifacts/embeddings/txt_meta.json tmp/ 2>/dev/null || true
gsutil cp gs://plotpointe-artifacts/embeddings/img_meta.json tmp/ 2>/dev/null || true
gsutil cp gs://plotpointe-artifacts/embeddings/fusion_config.json tmp/ 2>/dev/null || true

echo ""
echo "[EMBEDDINGS] ✅ Pipeline complete"
echo ""
echo "=== Text Embeddings ==="
cat tmp/txt_meta.json 2>/dev/null || echo "  (metadata not found)"
echo ""
echo "=== Image Embeddings ==="
cat tmp/img_meta.json 2>/dev/null || echo "  (metadata not found)"
echo ""
echo "=== Fusion Config ==="
cat tmp/fusion_config.json 2>/dev/null || echo "  (metadata not found)"
echo ""
echo "Job IDs:"
echo "  Text: $JOB_TEXT"
echo "  Image: $JOB_IMAGE"
echo "  Fusion: $JOB_FUSE"
echo ""
echo "Artifacts:"
echo "  gs://plotpointe-artifacts/embeddings/txt.npy"
echo "  gs://plotpointe-artifacts/embeddings/txt_meta.json"
echo "  gs://plotpointe-artifacts/embeddings/img.npy"
echo "  gs://plotpointe-artifacts/embeddings/img_meta.json"
echo "  gs://plotpointe-artifacts/embeddings/img_items.parquet"
echo "  gs://plotpointe-artifacts/embeddings/fused.npy"
echo "  gs://plotpointe-artifacts/embeddings/fusion_config.json"

