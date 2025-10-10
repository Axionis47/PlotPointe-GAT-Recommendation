# Phase 7: Serving Packaging (baseline)

This phase adds a minimal serving bundle that can run on Cloud Run (or locally) and serve top-K recommendations using precomputed item embeddings. It is non-disruptive and works alongside your current training.

## Overview
- Exporter: `tools/export_item_embeddings.py` (runs in training image) to produce `item_embeddings.npy`
- Serving runtime: `serving/runtime.py` + FastAPI app `serving/app.py`
- Serving container: `docker/serving.Dockerfile` + GHCR workflow `.github/workflows/serving-container.yml`

Baseline recommendation logic: given a user's recent item_ids, compute the user vector as the mean of those items' vectors, then score all items by dot product and return top-K (excluding history). This provides immediate end-to-end serving without changing training.

## Build serving image (CI)
- Push to `main` or trigger manually the workflow `Container Image (Serving)`.
- Image name: `ghcr.io/<owner>/gat-serving:<tag>` (auto-tags: branch, sha, latest on main)

## Export item embeddings (one-time per model)
You can run as a Vertex job using the training image:

```
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=export-item-emb \
  --config=vertex/configs/export_item_embeddings.yaml \
  --args=""
```

Env vars for the YAML (typical):
- `CHECKPOINT_PT`: gs://.../checkpoints/<run>.pt
- `OUT_GCS`: gs://.../exports/item_embeddings.npy
- `MODEL_FAMILY`: gat_pyg|gat_custom (defaults to gat_pyg)
- `ITEM_FEATURES`: fused|txt
- `PROJECT_ID`, `STAGING_PREFIX`, `GRAPHS_PREFIX`, `EMBEDDINGS_PREFIX`

## Deploy serving on Cloud Run
Set environment variables at deploy time so the service loads on startup:

- `PROJECT_ID=plotpointe`
- `ITEM_EMBEDDINGS_URI=gs://.../exports/item_embeddings.npy`
- `TOPK=20` (optional)

Example deploy (replace image and URIs):

```
gcloud run deploy plotpointe-recsys --region=us-central1 \
  --image=ghcr.io/<owner>/gat-serving:latest \
  --allow-unauthenticated \
  --port=8080 \
  --set-env-vars=PROJECT_ID=plotpointe,ITEM_EMBEDDINGS_URI=gs://plotpointe-artifacts/models/gat/pyg/exports/item_embeddings.npy,TOPK=20
```

## API
- `GET /healthz` → `{status: ok}`
- `POST /startup` → load by body if not using env (JSON): `{project_id, item_embeddings_uri, topk?}`
- `POST /recommend` → body `{item_ids: [int], k?: int}` → returns `{indices: [], scores: []}`

## Notes
- This baseline does not use user embeddings from the model; it forms a fast heuristic user vector from history items. You can upgrade later to a proper user encoder or session model while keeping the same serving skeleton.
- The exporter runs in the training image and uses the original graph and features to compute item embeddings — no need to change training.

