#!/usr/bin/env bash
set -euo pipefail

# Submit Vertex AI GPU smoke test jobs (T4 and/or L4) and stream logs.
# Usage:
#   scripts/vertex_smoke_test.sh [t4|l4|both] [REGION]
# Defaults: both us-central1

KIND=${1:-both}
REGION=${2:-us-central1}

create_and_stream() {
  local cfg="$1"; local name="$2";
  echo "Submitting: $name with $cfg"
  JOB_NAME=$(gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$name" \
    --config="$cfg" \
    --format="value(name)")
  echo "Job: $JOB_NAME"
  echo "Streaming logs... (Ctrl+C to stop)"
  gcloud ai custom-jobs stream-logs "$JOB_NAME" --region="$REGION" || true
  echo "Finished streaming logs for $name"
}

case "$KIND" in
  t4)
    create_and_stream vertex/configs/smoke_t4.yaml gpu-smoke-t4
    ;;
  l4)
    create_and_stream vertex/configs/smoke_l4.yaml gpu-smoke-l4
    ;;
  both)
    create_and_stream vertex/configs/smoke_t4.yaml gpu-smoke-t4
    create_and_stream vertex/configs/smoke_l4.yaml gpu-smoke-l4
    ;;
  *)
    echo "Unknown argument: $KIND"; exit 1;
    ;;
esac

