#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a GCP project for a cloud-native recommender
# - Sets default region
# - Optionally enables required APIs
# - Prints a one-page checklist of API enablement status
#
# Usage:
#   ./scripts/bootstrap_gcp.sh -p <PROJECT_ID> -r <REGION> [--enable]
# Example:
#   ./scripts/bootstrap_gcp.sh -p my-proj-123 -r us-central1 --enable

PROJECT_ID=""
REGION=""
ENABLE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project)
      PROJECT_ID="$2"; shift 2;;
    -r|--region)
      REGION="$2"; shift 2;;
    --enable)
      ENABLE=true; shift;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install" >&2
  exit 1
fi

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: Missing -p|--project <PROJECT_ID>" >&2; exit 1
fi
if [[ -z "${REGION}" ]]; then
  echo "ERROR: Missing -r|--region <REGION> (e.g., us-central1)" >&2; exit 1
fi

# Required services
SERVICES=(
  run.googleapis.com
  artifactregistry.googleapis.com
  aiplatform.googleapis.com
  bigquery.googleapis.com
  pubsub.googleapis.com
  redis.googleapis.com
  cloudbuild.googleapis.com
  logging.googleapis.com
  monitoring.googleapis.com
  cloudscheduler.googleapis.com
)

# Configure project and region
echo "Setting gcloud project to: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "Setting default region to: ${REGION}"
gcloud config set compute/region "${REGION}" >/dev/null || true
# Cloud Run also keeps its own region setting
(gcloud config set run/region "${REGION}" >/dev/null) || true

if ${ENABLE}; then
  echo "Enabling required APIs (this may take a few minutes)..."
  gcloud services enable "${SERVICES[@]}"
else
  echo "Skipping API enablement (check-only mode). Use --enable to enable any missing APIs."
fi

echo
echo "==================== One-Page Checklist ===================="
echo "Project ID : ${PROJECT_ID}"
echo "Region     : ${REGION}"
echo "-----------------------------------------------------------"

PAD_TO=45
for SVC in "${SERVICES[@]}"; do
  NAME="$SVC"
  ENABLED=$(gcloud services list --enabled --filter="NAME=$SVC" --format="value(NAME)" || true)
  STATUS="DISABLED"
  if [[ -n "$ENABLED" ]]; then STATUS="ENABLED"; fi
  # Pretty align output
  printf "%-45s %s\n" "$NAME" "$STATUS"
done

echo "==========================================================="

# Exit non-zero if any service remains disabled (useful for CI)
MISSING=0
for SVC in "${SERVICES[@]}"; do
  if [[ -z $(gcloud services list --enabled --filter="NAME=$SVC" --format="value(NAME)" || true) ]]; then
    MISSING=$((MISSING+1))
  fi
done

if [[ $MISSING -gt 0 ]]; then
  echo "Some services are DISABLED. Re-run with --enable or enable them manually." >&2
  exit 2
fi

echo "All required services are ENABLED."

