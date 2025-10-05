#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-plotpointe}
REGION=us-central1

echo "[UPLOAD] Sync train_lightgcn.py to GCS..."
gsutil cp scripts/train_lightgcn.py gs://plotpointe-artifacts/code/train/train_lightgcn.py

echo "[LAUNCH] Submitting LightGCN d=64 to $REGION..."
JOB64=$(gcloud ai custom-jobs create --region=$REGION --project=$PROJECT \
  --display-name=train-lightgcn-d64-$(date +%s) \
  --config=vertex/configs/train_lightgcn_l4_central.yaml \
  --format='value(name)')
ID64=${JOB64##*/}
echo "  Job64: $ID64"; echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/$ID64?project=$PROJECT"

sleep 3

echo "[LAUNCH] Submitting LightGCN d=128 to $REGION..."
JOB128=$(gcloud ai custom-jobs create --region=$REGION --project=$PROJECT \
  --display-name=train-lightgcn-d128-$(date +%s) \
  --config=vertex/configs/train_lightgcn_l4_128_central.yaml \
  --format='value(name)')
ID128=${JOB128##*/}
echo "  Job128: $ID128"; echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$REGION/training/customJobs/$ID128?project=$PROJECT"

