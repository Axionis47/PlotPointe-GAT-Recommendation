#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-plotpointe}

echo "[UPLOAD] Sync GAT trainers to GCS..."
gsutil cp scripts/train_gat_custom.py gs://plotpointe-artifacts/code/train/train_gat_custom.py
gsutil cp scripts/train_gat_pyg.py gs://plotpointe-artifacts/code/train/train_gat_pyg.py

launch_job() {
  local YAML="$1"; local NAME="$2"; local REGION="us-central1"
  echo "[LAUNCH] $NAME in $REGION"
  gcloud ai custom-jobs create --region="$REGION" --project=$PROJECT --display-name="$NAME-$(date +%s)" --config="$YAML" --format='value(name)'
}

CID1=$(launch_job vertex/configs/train_gat_custom_l4.yaml train-gat-custom || true)
echo "[INFO] custom: ${CID1:-submit-failed}"
CID2=$(launch_job vertex/configs/train_gat_pyg_l4.yaml train-gat-pyg || true)
echo "[INFO] pyg: ${CID2:-submit-failed}"

echo "[DONE] Submitted (if accepted). Use console to monitor."

