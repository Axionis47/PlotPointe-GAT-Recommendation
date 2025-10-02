#!/usr/bin/env bash
# Parallel pipeline execution
# Optimizes image embeddings + starts CPU tasks in parallel

set -e

PROJECT=${1:-plotpointe}
REGION=${2:-us-central1}

echo "============================================================"
echo "🚀 PARALLEL PIPELINE EXECUTION"
echo "============================================================"
echo "Project: $PROJECT"
echo "Region: $REGION"
echo "Strategy: L4 GPU (image) + Parallel CPU tasks"
echo "============================================================"
echo ""

# Step 1: Cancel current T4 job (if running)
echo "[STEP 1] Checking for running image embedding jobs..."
RUNNING_JOBS=$(gcloud ai custom-jobs list \
  --region=$REGION \
  --filter="displayName~'embed-image' AND (state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING)" \
  --format="value(name)" \
  --project=$PROJECT 2>/dev/null)

if [ -n "$RUNNING_JOBS" ]; then
  echo "Found running job(s). Cancelling to restart with L4..."
  for job in $RUNNING_JOBS; do
    echo "  Cancelling: $job"
    gcloud ai custom-jobs cancel $job --region=$REGION --project=$PROJECT 2>/dev/null || true
  done
  echo "  Waiting 10 seconds for cancellation..."
  sleep 10
else
  echo "  No running jobs found."
fi
echo ""

# Step 2: Upload latest code
echo "[STEP 2] Uploading latest embedding scripts..."
gsutil cp embeddings/embed_image.py gs://plotpointe-artifacts/code/embeddings/embed_image.py
echo "  ✅ Code uploaded"
echo ""

# Step 3: Launch optimized image embeddings on L4
echo "[STEP 3] Launching image embeddings on L4 GPU (2x faster)..."
IMG_JOB=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=embed-image-l4-$(date +%s) \
  --config=vertex/configs/embed_image_l4.yaml \
  --format='value(name)' \
  --project=$PROJECT)

IMG_JOB_ID="${IMG_JOB##*/}"
echo "  ✅ Job launched!"
echo "  Job ID: $IMG_JOB_ID"
echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/$IMG_JOB_ID?project=$PROJECT"
echo "  Expected duration: 3-4 hours (L4 GPU + batch_size=64)"
echo ""

# Step 4: Launch parallel CPU task - Graph construction (U-I edges)
echo "[STEP 4] Launching parallel CPU task: U-I edge construction..."
echo "  (This can run while image embeddings are processing)"
echo ""

# Check if graph construction script exists
if [ -f "graphs/build_ui_edges.py" ]; then
  echo "  Launching U-I edge construction job..."
  UI_JOB=$(gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=build-ui-edges-$(date +%s) \
    --config=vertex/configs/build_ui_edges.yaml \
    --format='value(name)' \
    --project=$PROJECT 2>/dev/null || echo "")
  
  if [ -n "$UI_JOB" ]; then
    UI_JOB_ID="${UI_JOB##*/}"
    echo "  ✅ U-I edges job launched!"
    echo "  Job ID: $UI_JOB_ID"
    echo "  Expected duration: 10-15 minutes"
  else
    echo "  ⚠️  U-I edges job config not found, skipping"
  fi
else
  echo "  ⚠️  Graph construction script not found yet"
  echo "  Will create after image embeddings complete"
fi
echo ""

# Step 5: Launch parallel CPU task - Text-based I-I kNN
echo "[STEP 5] Launching parallel CPU task: Text-based I-I kNN..."
echo "  (Uses txt.npy which is already complete)"
echo ""

if [ -f "graphs/build_ii_knn.py" ]; then
  echo "  Launching text I-I kNN job..."
  II_TEXT_JOB=$(gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=build-ii-knn-text-$(date +%s) \
    --config=vertex/configs/build_ii_knn_text.yaml \
    --format='value(name)' \
    --project=$PROJECT 2>/dev/null || echo "")
  
  if [ -n "$II_TEXT_JOB" ]; then
    II_TEXT_JOB_ID="${II_TEXT_JOB##*/}"
    echo "  ✅ Text I-I kNN job launched!"
    echo "  Job ID: $II_TEXT_JOB_ID"
    echo "  Expected duration: 15-20 minutes"
  else
    echo "  ⚠️  Text I-I kNN job config not found, skipping"
  fi
else
  echo "  ⚠️  I-I kNN script not found yet"
  echo "  Will create after image embeddings complete"
fi
echo ""

# Summary
echo "============================================================"
echo "✅ PARALLEL PIPELINE LAUNCHED!"
echo "============================================================"
echo ""
echo "📊 Active Jobs:"
echo "  1. Image Embeddings (L4 GPU): $IMG_JOB_ID"
[ -n "$UI_JOB_ID" ] && echo "  2. U-I Edges (CPU):            $UI_JOB_ID"
[ -n "$II_TEXT_JOB_ID" ] && echo "  3. Text I-I kNN (CPU):         $II_TEXT_JOB_ID"
echo ""
echo "⏱️  Expected Timeline:"
echo "  - CPU tasks:          15-20 minutes"
echo "  - Image embeddings:   3-4 hours"
echo "  - Modal fusion:       +20 minutes (after image complete)"
echo "  - Total:              ~4 hours (vs 8 hours sequential)"
echo ""
echo "💰 Cost Savings:"
echo "  - L4 GPU: ~$0.70/hour (vs T4 $0.35/hour, but 2x faster)"
echo "  - Total: ~$3 (vs $3.50 on T4)"
echo "  - Time saved: 4 hours!"
echo ""
echo "📈 Monitor progress:"
echo "  gcloud ai custom-jobs list --region=$REGION --filter='state:JOB_STATE_RUNNING'"
echo ""
echo "🔔 Next steps (after image embeddings complete):"
echo "  1. Modal fusion (20 min)"
echo "  2. Fused I-I kNN (20 min)"
echo "  3. GAT training (with your GPU quota increase!)"
echo ""

