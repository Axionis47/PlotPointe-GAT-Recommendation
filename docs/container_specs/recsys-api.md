# recsys-api — Container Spec (Freeze)

Summary
- Purpose: Online inference API (HTTP), emits logs/metrics and publishes request/feedback
- Image: us-docker.pkg.dev/plotpointe/recsys/recsys-api:latest
- Base: python:3.11-slim (or gcr.io/distroless/python3 for minimal)
- Platform: Cloud Run (fully managed)
- Service Account: sa-serve@plotpointe.iam.gserviceaccount.com

Runtime contract
- Entrypoint
  - uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
- Ports
  - ${PORT} (default 8080)
- Routes
  - GET /healthz → 200 OK (liveness)
  - GET /readyz → 200 OK when model loaded and BQ/PubSub clients initialized (readiness)
  - GET /metrics → Prometheus metrics (optional)
  - POST /v1/recommendations → request payload {user_id, context?, k?, filters?}; returns items [{asin, score, rank}]
  - POST /v1/feedback → payload {event_id, user_id, asin, event_type, request_id?, ts?}

Environment variables (required unless noted)
- PROJECT_ID: plotpointe
- REGION: us-central1
- ARTIFACT_BUCKET: gs://plotpointe-artifacts
- MODEL_DIR: gs://plotpointe-artifacts/models/recsys
- PUBSUB_TOPIC_REQUESTS: recsys-requests
- PUBSUB_TOPIC_FEEDBACK: recsys-feedback
- DATASET_RECSYS: recsys_logs
- REQUESTS_VIEW: recsys_logs.requests_v1
- FEEDBACK_VIEW: recsys_logs.feedback_v1
- DEFAULT_K: 20 (optional)
- AB_BUCKET: A (optional)

Behavioral contract
- On /v1/recommendations:
  - Generate request_id (uuid4) if not provided
  - Compute recommendations using the loaded model
  - Respond with items array [{asin, score, rank}] and echo request_id
  - Publish a request log to Pub/Sub topic ${PUBSUB_TOPIC_REQUESTS} with schema matching bigquery/schemas/recsys_logs.requests.json
  - Include fields: request_id, user_id, request_ts (UTC now), context, model{name,version,profile}, items[{asin,score,rank}], latency_ms, trace_id, experiment{name,variant}
- On /v1/feedback:
  - Ingest event and publish to ${PUBSUB_TOPIC_FEEDBACK} matching bigquery/schemas/recsys_logs.feedback.json

Idempotency
- Use request_id/event_id as insert_id when writing to downstream sinks (processors)
- API ensures same request_id/event_id yields no duplicate rows in BQ (at-least-once publish; downstream dedupe enforced)

Resources & autoscaling (Cloud Run)
- CPU: 2 vCPU, Memory: 4 GiB (tune via load testing)
- Concurrency: 16 (tune)
- Min instances: 0–1 (cold-start tradeoff); Max instances: 20 (guardrail)
- Request timeout: 60s

Security & IAM
- SA: sa-serve@plotpointe.iam.gserviceaccount.com
- Roles: roles/pubsub.publisher (to recsys-requests/recsys-feedback), roles/storage.objectViewer (to model artifacts)
- VPC egress: restricted as needed

Build & CI
- Dockerfile (conceptual)
  - FROM python:3.11-slim
  - RUN pip install -r requirements-api.txt
  - COPY api/ /app/api
  - ENV PYTHONPATH=/app
  - CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","${PORT}"]

Observability
- Structured JSON logs with request_id, user_id, route, latency_ms
- /metrics provides qps/latency histograms, model load latencies, publish success/fail counts

Readiness
- /readyz returns 200 only after: model artifact downloaded/loaded; Pub/Sub client warmed; BQ client reachable

Versioning
- Tag images per build: recsys-api:{git_sha}
- Expose model version in responses as model_version

Notes
- Keep response and logging schemas aligned with the BigQuery schemas and v1 compatibility views.

