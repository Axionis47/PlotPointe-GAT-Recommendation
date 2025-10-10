# GAT Recsys (Graph Attention based Recommendation)

A clean, end-to-end recommendation system built on PyTorch Geometric (GAT) and Google Cloud (Vertex AI). Simple to understand, easy to run, and production-ready.

> TL;DR (with numbers)
> - Best model (PyG GAT, fused, BPR): NDCG@20 ≈ 0.0160, Recall@20 ≈ 0.0433 (test)
> - Lift vs BCE (same setup): +~30% NDCG@20, +~23% Recall@20
> - Data scale: 1.69M interactions · 498k items · 192k users · L4 GPUs on Vertex AI
> - Baselines done well: stable setup, clean ablations, val≈test, reproducible runs


- Dataset: Amazon Electronics
  - 1,689,116 interactions
  - 498,196 items (63,001 with interactions)
  - 192,403 users
- Features: text, image, and fused (text+image)
- Graphs: User–Item bipartite, Item–Item cosine kNN
- Training: Vertex AI (L4 GPUs), containerized
- Serving: FastAPI on Cloud Run (top‑K recommendations using exported item embeddings)
- MLOps: Experiments, deterministic runs, structured logs, CI, Terraform IaC

---

## What we built (in simple words)
- Created item embeddings from text and images, and combined them for better signals.
- Built two graphs: one connects users to items (based on interactions), another connects items with similar items (cosine similarity).
- Trained Graph Attention Networks (GAT) to learn better item representations and ranking.
- Logged experiments and metrics in Vertex AI for clean comparisons.
- Packaged training and serving in Docker containers; CI publishes images automatically.
- Exported item embeddings and served a fast top‑K API on Cloud Run.

---

## Results and ablations (short and clear)

Results (test set, k=20)

| Model     | Features | Loss | NDCG@20 | Recall@20 |
|-----------|----------|------|---------|-----------|
| GAT (PyG) | fused    | BPR  | 0.0160  | 0.0433    |
| GAT (PyG) | fused    | BCE  | 0.0123  | 0.0354    |
| GAT (custom) | text-only | BPR | 0.0077 | 0.0219 |
| GAT (custom) | fused | BCE | 0.0045 | 0.0123 |

- Baselines done well: stable training, clean ablations, and val ≈ test behavior.
- Features: fused (text+image) > text-only; image-only trails text-only.
- Losses: BPR > BCE for ranking quality on this dataset/setup.
- Model shape: 2 layers, hidden_dim=128, 1 head is a solid sweet spot.
- Sampling: fewer epochs with larger samples per epoch converged faster.
- LightGCN: planned but blocked by L4 capacity in region; GAT met goals.

Tip: Keep these defaults for a strong start
- item_features=fused, loss=bpr, hidden_dim=128, layers=2, heads=1.


### Ablation numbers (evidence)
Source files in repo (JSON):
- tmp/metrics/pyg/metrics_gat_pyg_d128_1759651719.json (fused, BPR)
- tmp/metrics/pyg/metrics_gat_pyg_d128_1759665437.json (fused, BCE)
- tmp/metrics/custom/metrics_gat_custom_d128_1759654706.json (text-only, BPR)
- tmp/metrics/custom/metrics_gat_custom_d128_1759650827.json (fused, BCE)

Key test metrics (NDCG@20, Recall@20):
- PyG GAT, fused, BPR → ndcg@20 = 0.0160, recall@20 = 0.0433
- PyG GAT, fused, BCE → ndcg@20 = 0.0123, recall@20 = 0.0354
- Custom GAT, text-only, BPR → ndcg@20 = 0.0077, recall@20 = 0.0219
- Custom GAT, fused, BCE → ndcg@20 = 0.0045, recall@20 = 0.0123

Takeaway (example): On PyG with fused features, BPR vs BCE improves NDCG@20 by ~30% and Recall@20 by ~23% on test.

Note: Exact values vary run-to-run; see the JSONs above for full details (val/test splits and config). You can “bless” one metrics.json and gate regressions using tools/promotion_gate.py.

---

## Architecture overview
- Embeddings
  - Text: sentence-transformers (all‑MiniLM‑L6‑v2, 384d)
  - Image: CLIP ViT-B/32 (512d)
  - Fusion: small MLP to 128d
- Graphs
  - User–Item bipartite (weights from normalized ratings)
  - Item–Item kNN (cosine), k=20
- Training
  - Two flavors: custom PyTorch GAT and PyG GAT
  - Determinism and seeds for reproducibility
  - Structured JSON logs and optional Vertex Experiments logging
- Serving
  - Exported item embeddings
  - FastAPI runtime builds a quick user vector from recent items and returns top‑K items

---

## How to run (quick start)

### 1) Build/publish containers (CI does this on push)
- Training image: ghcr.io/<owner>/gat-recsys
- Serving image: ghcr.io/<owner>/gat-serving

GitHub Actions workflows:
- .github/workflows/container.yml (training)
- .github/workflows/serving-container.yml (serving)

### 2) Train on Vertex AI with prebuilt image
Use one of the provided configs (L4 GPU):
- vertex/configs/train_gat_pyg_l4_image.yaml
- vertex/configs/train_gat_custom_l4_image.yaml

Example:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=gat-pyg-image \
  --config=vertex/configs/train_gat_pyg_l4_image.yaml \
  --args=""
```

Optional Vertex Experiments (set before submit):

```bash
export USE_VERTEX_EXP=1
export EXP_NAME=GAT-Phase5
export RUN_NAME=gat_pyg_d128_$(date +%s)
```

### 3) Export item embeddings

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=export-item-emb \
  --config=vertex/configs/export_item_embeddings.yaml \
  --args=""
```

Key env vars for the YAML:
- CHECKPOINT_PT: gs://.../checkpoints/<run>.pt
- OUT_GCS: gs://.../exports/item_embeddings.npy
- MODEL_FAMILY: gat_pyg | gat_custom (default: gat_pyg)
- ITEM_FEATURES: fused | txt

### 4) Deploy serving on Cloud Run

```bash
gcloud run deploy gat-recsys --region=us-central1 \
  --image=ghcr.io/<owner>/gat-serving:latest \
  --allow-unauthenticated \
  --port=8080 \
  --set-env-vars=PROJECT_ID=plotpointe,ITEM_EMBEDDINGS_URI=gs://plotpointe-artifacts/models/gat/pyg/exports/item_embeddings.npy,TOPK=20
```

Endpoints
- GET /healthz
- POST /startup
- POST /recommend

---

## CI/CD and MLOps
- CI
  - Lint, type-check, tests on PRs and pushes
  - Packaging sanity (pip install .)
  - Terraform validate for IaC
- Containers
  - GH Actions publish to GHCR (training and serving images)
- Experiments and logs
  - Optional Vertex Experiments logging for runs
  - Structured JSON logging for metrics (e.g., ndcg@K, recall@K)
- Promotion gate
  - Simple tool to block regressions before promoting a model run
- Determinism
  - set_seeds() and enable_determinism() flags for reproducible results

---

## Tech stack
- Python, PyTorch, PyTorch Geometric, NumPy, pandas
- GCP: Vertex AI (Custom Jobs, Experiments), Cloud Storage, Cloud Run
- Infra: Docker, GitHub Actions, Terraform, pytest

---

## Why this project is strong for production
- Reproducible and containerized: train and serve consistently
- Clear experiment hygiene and simple ablation takeaways
- Ready serving path with minimal infra, easy to extend later

---

## Folder map (quick reference)
- plotpointe/ → library code (embeddings CLI, utils, registry)
- scripts/ → training scripts (custom GAT, PyG GAT)
- vertex/configs/ → Vertex AI job YAMLs
- docker/ → Dockerfiles for training and serving
- serving/ → FastAPI runtime
- tools/ → exporters, validators, promotion gate
- terraform/ → IaC (optional)
- tests/ → unit tests

---

## Notes
- GPU: NVIDIA L4 (Vertex AI)
- Region: us-central1
- Buckets/URIs use project_id "plotpointe" (can be changed later if required)

