# PlotPointe System Overview (v1)

Last updated: 2025-10-04
Owner: PlotPointe (GAT-Recommendation)
Scope: High-level architecture, flow, interfaces, rollout plan

---

## 1) Objective and success criteria

Build a modular, production-ready recommendation system on GCP that:
- Ingests/validates Amazon Electronics data
- Produces multi-modal item representations (text + image + fused)
- Constructs user–item and item–item graphs
- Trains GNN-based recommenders (LightGCN → GAT)
- Deploys and evaluates with strict compatibility on Vertex AI (L4 GPUs)

Success criteria:
- Reproducible Vertex pipelines; each stage is hermetic, versioned, and observable
- No compatibility issues on Vertex AI base images (Py310 + CUDA 12.1)
- Clear interfaces/artifacts between stages; easy to debug and re-run
- Stepwise approvals: each module passes tests and a smoke run before promotion

---

## 2) Environments and constraints

- Cloud: Google Cloud Platform (Vertex AI Custom Jobs, GCS, BigQuery)
- GPUs: NVIDIA L4 only (no T4). Machine type: g2-standard-8 for GPU jobs
- Language: Python 3.10
- Base images:
  - CPU: us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cpu.py310
  - GPU: us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-1.py310
- Key versions: torch==2.1.0, torchvision==0.16.0, transformers==4.35.2

---

## 3) End-to-end flow (data → training)

1. Data staging
   - Inputs: raw Amazon Electronics parquet in GCS staging prefix
   - Outputs: items.parquet, interactions.parquet under gs://…/staging/amazon_electronics
   - Gate: schema/quality checks pass (non-null, ranges, FK integrity)

2. Validation (quality gate)
   - Run lightweight checks (non-null, ranges, timestamp bounds, unique keys, FK)
   - Fail fast if any contract violated

3. Embeddings
   - Text embeddings (Sentence-Transformers) → txt.npy + txt_meta.json
   - Image embeddings (CLIP, GPU L4) → img.npy + img_meta.json + img_items.parquet
   - Fusion (MLP, GPU L4) → fused.npy + fused_meta.json (128d)

4. Graph Construction
   - U–I edges (weighted by normalized ratings) → ui_edges.npz + ui_stats.json + node_maps.json
   - I–I kNN (text, fused) → ii_edges_txt.npz / ii_edges_fused.npz + stats

5. Training
   - Baseline: LightGCN using U–I weights; ablations with txt vs fused
   - Target: GAT over multimodal item features and U–I edges
   - Outputs: model checkpoints, train/val metrics, config

6. Evaluation & Reporting
   - Offline metrics (e.g., Recall@K, NDCG@K)
   - Artifact consistency and cost/time summaries

7. Serving (later phase)
   - FAISS/ANN index for retrieval; model scoring service (Cloud Run)

8. Monitoring (deferred per plan)
   - Data/feature/graph drift to BigQuery, alerts in Cloud Monitoring

All steps write artifacts to GCS and log metrics to Vertex Experiments.

---

## 4) Planned modular structure (post-refactor)

- plotpointe.core
  - io: GCS I/O, path utils, artifact versioning
  - vertex: experiment/logging wrappers
  - logging: unified logs, timers
  - contracts: simple schema helpers
- plotpointe.data
  - validate: existing validation checks
  - features: text preparation (title + brand + categories)
- plotpointe.embeddings
  - text: text pipeline
  - image: image pipeline (L4)
  - fusion: fusion MLP (L4)
  - cli: thin entrypoints for Vertex jobs
- plotpointe.graphs
  - ui: user–item edges (weighted)
  - ii: item–item kNN (cosine)
  - stats: shared metrics export
- plotpointe.train
  - lightgcn: baseline
  - gat: main model
  - data: loaders from artifacts
  - cli: training entrypoints

Interfaces are file-based (GCS paths). Each module is independently runnable.

---

## 5) Artifacts and interfaces (contracts)

GCS Prefixes (example):
- Staging: gs://plotpointe-artifacts/staging/amazon_electronics
- Embeddings: gs://plotpointe-artifacts/embeddings
- Graphs: gs://plotpointe-artifacts/graphs
- Models: gs://plotpointe-artifacts/models

Key files:
- Embeddings: txt.npy, img.npy, fused.npy + *_meta.json, img_items.parquet
- Graphs: ui_edges.npz, node_maps.json, ui_stats.json, ii_edges_*.npz, *_stats.json
- Models: checkpoints/*.pt, metrics.json

---

## 6) Orchestration & submission strategy

- Vertex YAMLs per stage (CPU vs L4) with pinned base images and pip versions
- scripts/* helpers for uploading code, submitting jobs, polling status, and printing console URLs
- Experiments: aiplatform Experiments for run metadata and metrics
- Cost/time: capture throughput and wall time per stage; prefer parallel where safe

---

## 7) Approval gates (how we’ll proceed)

We will isolate and approve one component at a time:
- Gate A: Embeddings (text → image → fusion) produce expected shapes, metadata, and pass smoke runs on Vertex (L4 for GPU stages)
- Gate B: Graphs produce expected counts, sparsity, similarity stats
- Gate C: Training (LightGCN) trains end-to-end on produced artifacts; checkpoints and metrics logged
- Gate D: Training (GAT) with edge weights and multimodal features
- Gate E: Serving baseline (index + API) smoke passes
- Gate F: Monitoring layer (drift) activated

Each gate requires: config review, one Vertex run, metrics check, and artifact verification.

---

## 8) Testing & validation per stage

- Unit tests: deterministic feature logic, loaders, and utils
- Integration tests (selective, cost-safe): small-sample pipeline runs
- Smoke tests: GPU availability, CLIP load, minimal batch embed
- Validation: preflight schema checks before heavy jobs

---

## 9) Compatibility guardrails

- Use only NVIDIA L4 for GPU jobs (g2-standard-8); no T4
- Pin torch/transformers compatible versions (torch 2.1.0 + transformers 4.35.2)
- Use official Vertex Deep Learning Containers
- Avoid system-level changes; pip install only within job containers

---

## 10) Immediate next steps

1) Modularization Step 1 (Embeddings only): package layout and imports refactor without behavior changes; keep current Vertex job entrypoints working
2) L4 standardization: ensure fusion job YAML uses L4 variant consistently
3) Add training baseline (LightGCN) with Vertex config; verify end-to-end
4) Introduce GAT training with edge weights; add ablations
5) Only after core flow is sealed, add monitoring and dashboards

End of document.

