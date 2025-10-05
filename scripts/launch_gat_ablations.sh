#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-plotpointe}
REGION=${REGION:-us-central1}

echo "[UPLOAD] Sync GAT trainers to GCS..."
gsutil cp scripts/train_gat_custom.py gs://plotpointe-artifacts/code/train/train_gat_custom.py >/dev/null 2>&1 || true
gsutil cp scripts/train_gat_pyg.py gs://plotpointe-artifacts/code/train/train_gat_pyg.py >/dev/null 2>&1 || true

declare -a JOBS=()

submit() {
  local yaml="$1"; local name="$2"
  echo "[LAUNCH] $name"
  local id
  id=$(gcloud ai custom-jobs create --region="$REGION" --project="$PROJECT" \
        --display-name="$name-$(date +%s)" --config="$yaml" --format='value(name)' || true)
  echo "[INFO] $name -> ${id:-submit-failed}"
  JOBS+=("$id")
}

# Custom GAT ablations (single-variable)
submit vertex/configs/train_gat_custom_l4_txt.yaml       train-gat-custom-txt
submit vertex/configs/train_gat_custom_l4_layers1.yaml   train-gat-custom-l1
submit vertex/configs/train_gat_custom_l4_bce.yaml       train-gat-custom-bce

# PyG GAT ablations (single-variable)
submit vertex/configs/train_gat_pyg_l4_heads2.yaml       train-gat-pyg-h2
submit vertex/configs/train_gat_pyg_l4_txt.yaml          train-gat-pyg-txt
submit vertex/configs/train_gat_pyg_l4_layers1.yaml      train-gat-pyg-l1
submit vertex/configs/train_gat_pyg_l4_bce.yaml          train-gat-pyg-bce

printf "[SUMMARY] Submitted %d jobs\n" "${#JOBS[@]}"
for j in "${JOBS[@]}"; do
  [[ -n "$j" ]] && echo " - $j"
done

