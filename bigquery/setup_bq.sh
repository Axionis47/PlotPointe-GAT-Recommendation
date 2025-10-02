#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${1:-plotpointe}
LOCATION=us-central1

log(){ echo "[BQ] $*"; }

# Create datasets (idempotent)
mk_dataset(){
  local ds=$1
  if bq --project_id="$PROJECT_ID" --location="$LOCATION" ls -d | awk '{print $1}' | grep -qx "$PROJECT_ID:$ds"; then
    log "Dataset exists: $ds"
  else
    log "Creating dataset: $ds in $LOCATION"
    bq --project_id="$PROJECT_ID" --location="$LOCATION" mk -d "$ds"
  fi
}

mk_dataset recsys_logs
mk_dataset drift

# Create tables
# recsys_logs.requests: partition by request_ts (DAY), cluster by user_id, request_id, 90-day partition TTL
log "Creating/Updating table recsys_logs.requests"
bq --project_id="$PROJECT_ID" --location="$LOCATION" mk \
  --table \
  --time_partitioning_type=DAY \
  --time_partitioning_field=request_ts \
  --time_partitioning_expiration=7776000 \
  --clustering_fields=user_id,request_id \
  recsys_logs.requests \
  ./bigquery/schemas/recsys_logs.requests.json || true

# recsys_logs.feedback: partition by event_ts (DAY), cluster by user_id, asin, event_type, 180-day partition TTL
log "Creating/Updating table recsys_logs.feedback"
bq --project_id="$PROJECT_ID" --location="$LOCATION" mk \
  --table \
  --time_partitioning_type=DAY \
  --time_partitioning_field=event_ts \
  --time_partitioning_expiration=15552000 \
  --clustering_fields=user_id,asin,event_type \
  recsys_logs.feedback \
  ./bigquery/schemas/recsys_logs.feedback.json || true

# drift.hourly: partition by summary_ts (HOUR), cluster by metric,status, no expiration
log "Creating/Updating table drift.hourly"
bq --project_id="$PROJECT_ID" --location="$LOCATION" mk \
  --table \
  --time_partitioning_type=HOUR \
  --time_partitioning_field=summary_ts \
  --clustering_fields=metric,status \
  drift.hourly \
  ./bigquery/schemas/drift.hourly.json || true

log "Done."

