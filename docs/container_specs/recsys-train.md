# recsys-train — Container Spec (Freeze)

Summary
- Purpose: Offline/Batch training jobs on Vertex AI (GPU capable)
- Image: us-docker.pkg.dev/plotpointe/recsys/recsys-train:latest
- Base: us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310
- Orchestration: Vertex AI CustomJob
- Service Account: sa-pipeline@plotpointe.iam.gserviceaccount.com

Runtime contract
- Entrypoint
  - /usr/bin/bash -lc "python -m train.main $TRAIN_ARGS"
- Health & Metrics
  - /healthz (HTTP 200) — optional lightweight probe when running long-lived workers
  - /metrics — Prometheus exposition (optional); in jobs, prefer structured logs
- Logging
  - JSON logs to stdout; include fields: {run_id, request_id, experiment, step, metric, value}

Resources
- Default machine: g2-standard-8 (8 vCPU, 32GB)
- GPU: 1× NVIDIA L4 (acceleratorCount=1)
- Disk: 100 GB (ephemeral) minimum
- Scheduling: On-demand by default; allow SPOT for cost-optimized runs with restartJobOnWorkerRestart

Environment variables (required unless noted)
- PROJECT_ID: plotpointe
- REGION: us-central1
- ARTIFACT_BUCKET: gs://plotpointe-artifacts
- STAGING_PREFIX: gs://plotpointe-artifacts/staging/amazon_electronics
- EXPERIMENT_NAME: recsys-dev
- DATASET_RECSYS: recsys_logs
- DATASET_DRIFT: drift
- REQUESTS_TABLE: recsys_logs.requests
- FEEDBACK_TABLE: recsys_logs.feedback
- DRIFT_TABLE: drift.hourly
- MODEL_DIR: gs://plotpointe-artifacts/models/recsys
- CHECKPOINT_DIR: gs://plotpointe-artifacts/checkpoints/recsys
- WANDB_ENABLED: false (optional)
- MLflow/Experiments
  - VERTEX_EXPERIMENT: recsys-dev (preferred)

I/O contract
- Read: staging parquet — ${STAGING_PREFIX}/interactions.parquet, ${STAGING_PREFIX}/items.parquet
- Read: BigQuery — ${PROJECT_ID}.${REQUESTS_TABLE}, ${PROJECT_ID}.${FEEDBACK_TABLE}
- Write: model artifacts to ${MODEL_DIR}/$RUN_ID
- Write: checkpoints to ${CHECKPOINT_DIR}/$RUN_ID
- Metrics: stdout JSON and Vertex Experiments params/metrics

Health and lifecycle
- Liveness: optional — GET /healthz returns 200; in pure batch jobs, rely on job status
- Readiness: not applicable for batch jobs
- PreStop: flush metrics/log buffers; upload last checkpoint

Security & IAM
- SA: sa-pipeline@plotpointe.iam.gserviceaccount.com
- Roles: roles/aiplatform.user, roles/storage.objectAdmin (scoped to artifacts bucket), roles/bigquery.dataViewer (for datasets), roles/bigquery.jobUser

Networking
- Egress to GCS, BigQuery, Vertex AI only; no inbound except optional /healthz when used

Build & CI
- Dockerfile (conceptual)
  - FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310
  - pip install -r requirements-train.txt
  - COPY src/ /app/src
  - ENV PYTHONPATH=/app
  - CMD ["python","-m","train.main"]

Observability
- Metrics: log training/validation metrics (loss, auc@k, ndcg@k)
- Traces: include run_id; optionally export to Vertex Experiments

Versioning
- Tag images per build: recsys-train:{git_sha}
- Tag models with run_id and git_sha in artifact metadata

Notes
- GPU smoke prerequisites validated (L4). Use same DLC family for runtime parity.

