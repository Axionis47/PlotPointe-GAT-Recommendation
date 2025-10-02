#!/usr/bin/env bash
# Automatically continue pipeline after image embeddings complete
# Launches modal fusion and fused I-I kNN

set -e

PROJECT=${1:-plotpointe}
REGION=${2:-us-central1}
IMG_JOB_ID="2841805123313729536"

echo "============================================================"
echo "🤖 AUTO-CONTINUE PIPELINE"
echo "============================================================"
echo "Waiting for image embeddings to complete..."
echo "Job ID: $IMG_JOB_ID"
echo "============================================================"
echo ""

# Wait for image embeddings to complete
while true; do
  STATE=$(gcloud ai custom-jobs describe \
    projects/359145045403/locations/$REGION/customJobs/$IMG_JOB_ID \
    --region=$REGION \
    --format='value(state)' \
    --project=$PROJECT 2>/dev/null || echo "UNKNOWN")
  
  echo "[$(date +%H:%M:%S)] Image embeddings state: $STATE"
  
  if [[ "$STATE" == "JOB_STATE_SUCCEEDED" ]]; then
    echo ""
    echo "✅ Image embeddings complete!"
    break
  elif [[ "$STATE" == "JOB_STATE_FAILED" ]] || [[ "$STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo ""
    echo "❌ Image embeddings failed or cancelled. Cannot continue."
    exit 1
  fi
  
  sleep 60
done

echo ""
echo "============================================================"
echo "🚀 LAUNCHING MODAL FUSION"
echo "============================================================"
echo ""

# Upload fuse_modal.py if needed
gsutil cp embeddings/fuse_modal.py gs://plotpointe-artifacts/code/embeddings/fuse_modal.py 2>/dev/null || true

# Launch modal fusion
FUSION_JOB=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=fuse-modal-$(date +%s) \
  --config=vertex/configs/fuse_modal.yaml \
  --format='value(name)' \
  --project=$PROJECT)

FUSION_JOB_ID="${FUSION_JOB##*/}"
echo "✅ Modal fusion launched!"
echo "Job ID: $FUSION_JOB_ID"
echo "Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/$FUSION_JOB_ID?project=$PROJECT"
echo ""

# Wait for modal fusion to complete
echo "Waiting for modal fusion to complete..."
while true; do
  STATE=$(gcloud ai custom-jobs describe \
    projects/359145045403/locations/$REGION/customJobs/$FUSION_JOB_ID \
    --region=$REGION \
    --format='value(state)' \
    --project=$PROJECT 2>/dev/null || echo "UNKNOWN")
  
  echo "[$(date +%H:%M:%S)] Modal fusion state: $STATE"
  
  if [[ "$STATE" == "JOB_STATE_SUCCEEDED" ]]; then
    echo ""
    echo "✅ Modal fusion complete!"
    break
  elif [[ "$STATE" == "JOB_STATE_FAILED" ]] || [[ "$STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo ""
    echo "❌ Modal fusion failed or cancelled. Cannot continue."
    exit 1
  fi
  
  sleep 30
done

echo ""
echo "============================================================"
echo "🚀 LAUNCHING FUSED I-I KNN"
echo "============================================================"
echo ""

# Launch fused I-I kNN
FUSED_KNN_JOB=$(gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=build-ii-knn-fused-$(date +%s) \
  --config=vertex/configs/build_ii_knn_fused.yaml \
  --format='value(name)' \
  --project=$PROJECT)

FUSED_KNN_JOB_ID="${FUSED_KNN_JOB##*/}"
echo "✅ Fused I-I kNN launched!"
echo "Job ID: $FUSED_KNN_JOB_ID"
echo "Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/$FUSED_KNN_JOB_ID?project=$PROJECT"
echo ""

# Wait for fused I-I kNN to complete
echo "Waiting for fused I-I kNN to complete..."
while true; do
  STATE=$(gcloud ai custom-jobs describe \
    projects/359145045403/locations/$REGION/customJobs/$FUSED_KNN_JOB_ID \
    --region=$REGION \
    --format='value(state)' \
    --project=$PROJECT 2>/dev/null || echo "UNKNOWN")
  
  echo "[$(date +%H:%M:%S)] Fused I-I kNN state: $STATE"
  
  if [[ "$STATE" == "JOB_STATE_SUCCEEDED" ]]; then
    echo ""
    echo "✅ Fused I-I kNN complete!"
    break
  elif [[ "$STATE" == "JOB_STATE_FAILED" ]] || [[ "$STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo ""
    echo "❌ Fused I-I kNN failed or cancelled."
    exit 1
  fi
  
  sleep 30
done

echo ""
echo "============================================================"
echo "🎉 PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "✅ Completed steps:"
echo "  1. Text embeddings (498K items)"
echo "  2. Image embeddings (150K items)"
echo "  3. Modal fusion (498K items)"
echo "  4. U-I edges"
echo "  5. Text I-I kNN"
echo "  6. Fused I-I kNN"
echo ""
echo "📦 Outputs:"
echo ""
echo "Embeddings:"
gsutil ls -lh gs://plotpointe-artifacts/embeddings/*.npy 2>/dev/null
echo ""
echo "Graphs:"
gsutil ls -lh gs://plotpointe-artifacts/graphs/*.npz 2>/dev/null
echo ""
echo "🚀 READY FOR GAT TRAINING!"
echo ""
echo "Next steps:"
echo "  1. Review graph statistics"
echo "  2. Set up baseline models (MF/BPR, LightGCN)"
echo "  3. Train GAT models"
echo ""
echo "Completed at: $(date)"
echo "============================================================"

