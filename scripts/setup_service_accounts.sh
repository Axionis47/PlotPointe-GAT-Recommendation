#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${1:-plotpointe}

eq() { echo "[IAM] $*"; }

# Create SAs (idempotent)
create_sa() {
  local sa_id=$1; local display=$2
  if gcloud iam service-accounts describe "${sa_id}@${PROJECT_ID}.iam.gserviceaccount.com" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    eq "Service account exists: ${sa_id}"
  else
    eq "Creating service account: ${sa_id}"
    gcloud iam service-accounts create "$sa_id" \
      --display-name="$display" \
      --project="$PROJECT_ID"
  fi
}

bind_role() {
  local sa_email=$1; local role=$2
  eq "Binding ${role} -> ${sa_email}"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${sa_email}" \
    --role="$role" \
    --quiet >/dev/null
}

# 1) sa-ci (ID must be >=6 chars; use 'sa-ci-svc' with display 'sa-ci')
create_sa sa-ci-svc "sa-ci"
SA_CI="sa-ci-svc@${PROJECT_ID}.iam.gserviceaccount.com"
bind_role "$SA_CI" roles/artifactregistry.writer
bind_role "$SA_CI" roles/run.admin
bind_role "$SA_CI" roles/cloudbuild.builds.editor

# 2) sa-pipeline
create_sa sa-pipeline "Vertex pipeline runner"
SA_PIPELINE="sa-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"
bind_role "$SA_PIPELINE" roles/aiplatform.user
bind_role "$SA_PIPELINE" roles/bigquery.dataEditor
bind_role "$SA_PIPELINE" roles/secretmanager.secretAccessor
# NOTE: GCS access should be bucket-scoped (objectAdmin on artifacts bucket). Add with:
# gsutil iam ch serviceAccount:${SA_PIPELINE}:objectAdmin gs://YOUR_ARTIFACTS_BUCKET

# 3) sa-serve
create_sa sa-serve "Cloud Run serving invoker"
SA_SERVE="sa-serve@${PROJECT_ID}.iam.gserviceaccount.com"
bind_role "$SA_SERVE" roles/run.invoker
bind_role "$SA_SERVE" roles/storage.objectViewer
bind_role "$SA_SERVE" roles/secretmanager.secretAccessor
bind_role "$SA_SERVE" roles/pubsub.publisher

# 4) sa-dataflow (optional)
create_sa sa-dataflow "Dataflow/stream processors"
SA_DATAFLOW="sa-dataflow@${PROJECT_ID}.iam.gserviceaccount.com"
bind_role "$SA_DATAFLOW" roles/pubsub.editor
bind_role "$SA_DATAFLOW" roles/bigquery.dataEditor
bind_role "$SA_DATAFLOW" roles/storage.admin

eq "Done. To verify roles per SA:"
eq "gcloud projects get-iam-policy ${PROJECT_ID} --flatten=bindings[].members --filter=bindings.members:${SA_CI} --format='table(bindings.role)'"
eq "gcloud projects get-iam-policy ${PROJECT_ID} --flatten=bindings[].members --filter=bindings.members:${SA_PIPELINE} --format='table(bindings.role)'"
eq "gcloud projects get-iam-policy ${PROJECT_ID} --flatten=bindings[].members --filter=bindings.members:${SA_SERVE} --format='table(bindings.role)'"
eq "gcloud projects get-iam-policy ${PROJECT_ID} --flatten=bindings[].members --filter=bindings.members:${SA_DATAFLOW} --format='table(bindings.role)'"

