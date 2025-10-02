#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${1:-plotpointe}
REGION=us-central1

log(){ echo "[PUBSUB] $*"; }

gcloud config set project "$PROJECT_ID" >/dev/null

# Topics
for T in recsys-requests recsys-feedback recsys-requests-dlq recsys-feedback-dlq; do
  if gcloud pubsub topics describe "$T" >/dev/null 2>&1; then
    log "Topic exists: $T"
  else
    log "Creating topic: $T"
    gcloud pubsub topics create "$T"
  fi
done

# Subscriptions (pull) with DLQ policies
create_sub(){
  local sub=$1 topic=$2 dlq=$3
  if gcloud pubsub subscriptions describe "$sub" >/dev/null 2>&1; then
    log "Subscription exists: $sub"
  else
    log "Creating subscription: $sub -> $topic (DLQ=$dlq)"
    gcloud pubsub subscriptions create "$sub" \
      --topic="$topic" \
      --ack-deadline=60 \
      --message-retention-duration=604800s \
      --max-delivery-attempts=5 \
      --dead-letter-topic="projects/${PROJECT_ID}/topics/${dlq}"
  fi
}

create_sub recsys-requests-proc recsys-requests recsys-requests-dlq
create_sub recsys-feedback-proc recsys-feedback recsys-feedback-dlq

# DLQ viewer subscriptions (for ops triage)
if ! gcloud pubsub subscriptions describe recsys-requests-dlq-sub >/dev/null 2>&1; then
  gcloud pubsub subscriptions create recsys-requests-dlq-sub --topic=recsys-requests-dlq --ack-deadline=60
fi
if ! gcloud pubsub subscriptions describe recsys-feedback-dlq-sub >/dev/null 2>&1; then
  gcloud pubsub subscriptions create recsys-feedback-dlq-sub --topic=recsys-feedback-dlq --ack-deadline=60
fi

# Ensure Cloud Run processor SA can subscribe
PROC_SA="sa-serve@${PROJECT_ID}.iam.gserviceaccount.com"
log "Granting roles/pubsub.subscriber to ${PROC_SA}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROC_SA}" \
  --role=roles/pubsub.subscriber \
  --quiet >/dev/null

log "Done. Topics and subscriptions are ready."

