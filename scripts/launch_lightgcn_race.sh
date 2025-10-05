#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-plotpointe}

echo "[UPLOAD] Sync train_lightgcn.py to GCS..."
gsutil cp scripts/train_lightgcn.py gs://plotpointe-artifacts/code/train/train_lightgcn.py

echo "[FUNC] Defining race_train()"
race_train() {
  local YAML="$1"; local NAME_PREFIX="$2";
  local TS=$(date +%s)
  local REGIONS=(us-west4 us-east5 us-central1)
  local JOBS=()
  local IDS=()
  local JREG=()

  echo "[LAUNCH] Submitting ${NAME_PREFIX} to: ${REGIONS[*]}"
  for R in "${REGIONS[@]}"; do
    local DISP=${NAME_PREFIX}-${R}-${TS}
    set +e
    J=$(gcloud ai custom-jobs create --region="$R" --display-name="$DISP" --config="$YAML" --format='value(name)' --project=$PROJECT 2>/dev/null)
    RC=$?
    set -e
    if [ $RC -eq 0 ] && [ -n "$J" ]; then
      JOBS+=("$J"); IDS+=("${J##*/}"); JREG+=("$R")
      echo "  $R: ${J##*/}"
      echo "  Console: https://console.cloud.google.com/vertex-ai/locations/$R/training/customJobs/${J##*/}?project=$PROJECT"
    else
      echo "  $R: submit failed (skipping)"
    fi
  done

  if [ ${#JOBS[@]} -eq 0 ]; then
    echo "[RACE] No regions accepted the job at submit time. Exiting."
    return 1
  fi

  local WIN_REGION=""; local WIN_ID=""; local WIN_JOB=""
  for i in {1..60}; do
    for idx in "${!JOBS[@]}"; do
      R=${JREG[$idx]}
      J=${JOBS[$idx]}
      S=$(gcloud ai custom-jobs describe "$J" --region="$R" --project=$PROJECT --format='value(state)' || true)
      echo "  [POLL $i][$R] state=$S"
      if [ "$S" = "JOB_STATE_RUNNING" ]; then
        WIN_REGION=$R; WIN_JOB=$J; WIN_ID=${IDS[$idx]}; break 2
      fi
    done
    sleep 20
  done

  if [ -n "$WIN_ID" ]; then
    echo "[RACE] Winner: $WIN_REGION/$WIN_ID. Cancelling other regions..."
    for idx in "${!JOBS[@]}"; do
      R=${JREG[$idx]}
      J=${JOBS[$idx]}
      if [ "$J" != "$WIN_JOB" ]; then
        gcloud ai custom-jobs cancel "$J" --project=$PROJECT || true
      fi
    done
  else
    echo "[RACE] No region reached RUNNING within poll window. Proceeding with first submitted job."
    WIN_REGION=${REGIONS[0]}; WIN_JOB=${JOBS[0]}; WIN_ID=${IDS[0]}
  fi

  echo "[WAIT] Waiting for $WIN_REGION/$WIN_ID to complete..."
  for j in {1..180}; do
    S=$(gcloud ai custom-jobs describe "$WIN_JOB" --region="$WIN_REGION" --format='value(state)' --project=$PROJECT || true)
    echo "  [WAIT $j][$WIN_REGION] state=$S"
    if [ "$S" = "JOB_STATE_SUCCEEDED" ]; then echo "[DONE] $NAME_PREFIX SUCCEEDED"; break; fi
    if [ "$S" = "JOB_STATE_FAILED" ] || [ "$S" = "JOB_STATE_CANCELLED" ]; then echo "[DONE] $NAME_PREFIX TERMINATED: $S"; break; fi
    sleep 30
  done
}

race_train vertex/configs/train_lightgcn_l4.yaml train-lightgcn-d64 &
PID1=$!
race_train vertex/configs/train_lightgcn_l4_128.yaml train-lightgcn-d128 &
PID2=$!

echo "[INFO] Race monitors PIDs: $PID1, $PID2"
wait $PID1 || true
wait $PID2 || true

echo "[ALL DONE] Race monitors exited."

