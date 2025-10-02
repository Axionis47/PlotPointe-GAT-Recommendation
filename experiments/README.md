# Vertex AI Experiments: How to Track and View Runs

Why Vertex Experiments (vs. self-hosted MLflow)
- No server to operate; first-party in Vertex AI
- Unified UI to compare params, metrics, and link artifacts in GCS
- Works with Vertex training and local runs via the Python SDK

Prereqs
- APIs: Vertex AI API enabled (done)
- Project/location: plotpointe / us-central1
- Service Accounts: writers need roles/aiplatform.user (sa-pipeline already has it)

Quick start (local or Cloud Shell)
1) Python deps:
   pip install google-cloud-aiplatform
2) Create a placeholder artifact (already uploaded):
   gs://plotpointe-artifacts/experiments/recsys-dev/placeholder.txt
3) Log a sample run:
   python experiments/vertex_log_example.py --project plotpointe --location us-central1 --experiment recsys-dev \
     --artifact_uri gs://plotpointe-artifacts/experiments/recsys-dev/placeholder.txt

Where to view
- Console: https://console.cloud.google.com/vertex-ai/experiments?project=plotpointe
- Click experiment "recsys-dev"; select runs to compare, inspect params/metrics, and artifact_uri

Next steps
- Log time-series metrics each epoch; log model checkpoints to gs://plotpointe-artifacts/experiments/<exp>/<run>/
- From Vertex custom training jobs, call the same SDK and use the service account bindings already in place.

