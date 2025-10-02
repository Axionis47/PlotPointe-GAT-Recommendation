# Project Progress — Cloud-Native GAT Recommendation System

**Project:** plotpointe  
**Region:** us-central1  
**Last Updated:** 2025-10-01

---

## Phase 0: Infrastructure (Steps 1-9) ✅ COMPLETE

All foundational infrastructure is deployed and validated:

- [x] **Step 1:** GCP Project & APIs (23 APIs enabled)
- [x] **Step 2:** Service Accounts (sa-pipeline, sa-serve, sa-processor)
- [x] **Step 3:** GCS Buckets (plotpointe-artifacts with lifecycle policies)
- [x] **Step 4:** Artifact Registry (recsys repository)
- [x] **Step 5:** BigQuery Datasets & Tables (recsys_logs, drift + compatibility views)
- [x] **Step 6:** Pub/Sub Topics & Subscriptions (with DLQs)
- [x] **Step 7:** Memorystore Redis (recsys-cache, 1GB)
- [x] **Step 8:** Vertex AI Experiments (recsys-dev)
- [x] **Step 9:** GPU Quotas & Smoke Tests (L4=2, T4 preemptible validated)

**Additional Validations:**
- [x] **Prompt C:** Amazon Electronics data staged (1.69M interactions, 498K items, all GE checks PASS)
- [x] **Prompt D:** BigQuery compatibility views created (requests_v1, feedback_v1, hourly_v1)
- [x] **Prompt E:** API contract & Pub/Sub dry-run (idempotency verified)
- [x] **Prompt F:** Container specs frozen (recsys-train, recsys-api)

**Documentation:**
- `docs/readiness_report_final.md` — Full infrastructure status report
- `docs/container_specs/recsys-train.md` — Training container spec
- `docs/container_specs/recsys-api.md` — Serving API container spec

---

## Phase 1: Embeddings & Graph (Steps 10-11) 🚧 READY TO EXECUTE

### Step 10: Text & Image Embeddings ⏳ READY
**Status:** Code complete, configs ready, awaiting execution

**Components:**
- `embeddings/embed_text.py` — Sentence-transformers text encoder (CPU)
- `embeddings/embed_image.py` — CLIP image encoder (T4 preemptible, capped at 150k)
- `embeddings/fuse_modal.py` — Multimodal fusion MLP (concat → 128d)
- `vertex/configs/embed_text.yaml` — Vertex AI job config (CPU)
- `vertex/configs/embed_image.yaml` — Vertex AI job config (T4 preemptible)
- `vertex/configs/fuse_modal.yaml` — Vertex AI job config (CPU)
- `scripts/run_embeddings_pipeline.sh` — Orchestration script

**To Execute:**
```bash
bash scripts/run_embeddings_pipeline.sh plotpointe us-central1
```

**Expected Outputs:**
- `gs://plotpointe-artifacts/embeddings/txt.npy` (498k × 384)
- `gs://plotpointe-artifacts/embeddings/img.npy` (~135k × 512)
- `gs://plotpointe-artifacts/embeddings/fused.npy` (498k × 128)
- Metadata JSON files for each
- 3 runs logged to Vertex Experiments

**Estimated Time:** ~2 hours (text: 15min, image: 90min, fusion: 20min)  
**Estimated Cost:** ~$2-4

**Documentation:** `docs/step10_embeddings_plan.md`

---

### Step 11: Graph Construction ⏸️ PENDING
**Status:** Not started (depends on Step 10)

**Planned Components:**
- U–I edges from interactions.parquet
- I–I kNN for text (k∈{15,20})
- I–I kNN for fused (k∈{15,20})
- Node maps (user/item ID → graph node index)
- QA report (edge counts, degree histograms, orphan rates, sample neighbors)

**Artifacts:**
- `gs://plotpointe-artifacts/graphs/ui_edges.npz`
- `gs://plotpointe-artifacts/graphs/ii_edges_txt.npz`
- `gs://plotpointe-artifacts/graphs/ii_edges_fused.npz`
- `gs://plotpointe-artifacts/graphs/node_maps.json`
- `gs://plotpointe-artifacts/graphs/qa_report.json`

---

## Phase 2: Baselines & GAT Training (Steps 12-13) ⏸️ PENDING

### Step 12: Baselines (MF, LightGCN)
**Status:** Not started (depends on Step 11)

**Planned:**
- MF/BPR baseline (CPU or brief GPU)
- LightGCN baseline (T4 preemptible, neighbor sampling)
- Metrics: NDCG/Recall@10/20, cold-start slices, long-tail tertiles
- Export: TorchScript/ONNX + eval.json

---

### Step 13: GATv2 Training
**Status:** Not started (depends on Step 11)

**Planned:**
- Single-modal GAT (text-only)
- Multi-modal GAT (fused embeddings)
- Ablations: heads {1,4,8}, layers {1,2,3}
- Export: TorchScript/ONNX + eval.json

---

## Phase 3: Serving & Monitoring (Steps 14-18) ⏸️ PENDING

### Step 14: FAISS Index
**Status:** Not started (depends on Step 13)

### Step 15: Serving API (Cloud Run)
**Status:** Container spec frozen, implementation pending

### Step 16: Pub/Sub → BigQuery Landing
**Status:** Schemas ready, processor implementation pending

### Step 17: Streaming Simulator
**Status:** Not started

### Step 18: Drift Monitoring
**Status:** Not started

---

## Phase 4: Evaluation & Production (Steps 19-23) ⏸️ PENDING

### Step 19: Serving Performance Ablations
**Status:** Not started

### Step 20: Model Ablations
**Status:** Not started

### Step 21: A/B & Shadow Testing
**Status:** Not started

### Step 22: Dashboards
**Status:** Not started

### Step 23: Docs, Model Card, Teardown
**Status:** Not started

---

## Current State Summary

**✅ Complete:**
- Infrastructure (Steps 1-9)
- Data staging & validation
- API contracts & schemas
- Container design

**🚧 Ready to Execute:**
- Step 10: Embeddings pipeline (code + configs ready)

**⏸️ Blocked/Pending:**
- Steps 11-23 (sequential dependencies)

---

## Next Action

**Execute Step 10:**
```bash
# Upload code and run full embeddings pipeline
bash scripts/run_embeddings_pipeline.sh plotpointe us-central1
```

**Expected Duration:** ~2 hours  
**Expected Cost:** ~$2-4  
**Success Criteria:** 3 runs in Vertex Experiments, 7 artifacts in GCS, counts match items

---

## Key Artifacts

### Infrastructure
- `scripts/setup_gcp.sh` — GCP bootstrap
- `scripts/setup_bigquery.sh` — BigQuery datasets/tables/views
- `scripts/setup_pubsub.sh` — Pub/Sub topics/subscriptions
- `scripts/test_pubsub_landing.sh` — Idempotency validation

### Data
- `data/pipelines/stage_amazon_electronics.py` — Data staging
- `data/validation/validate_amazon_electronics.py` — Great Expectations validation

### Embeddings (Step 10)
- `embeddings/embed_text.py` — Text encoder
- `embeddings/embed_image.py` — Image encoder (CLIP)
- `embeddings/fuse_modal.py` — Multimodal fusion
- `scripts/run_embeddings_pipeline.sh` — Pipeline orchestrator

### Documentation
- `docs/readiness_report_final.md` — Infrastructure status
- `docs/step10_embeddings_plan.md` — Embeddings pipeline plan
- `docs/container_specs/` — Container specifications
- `bigquery/schemas/` — BigQuery table schemas

---

## Cost Tracking

**Infrastructure (idle):**
- GCS storage: ~$0.02/GB/month (~$0.03/month for 1.5GB)
- BigQuery storage: ~$0.02/GB/month (minimal, partitioned with TTL)
- Memorystore Redis: ~$0.05/hour (~$36/month if always on, $0 when stopped)
- Cloud Run: $0 (min-instances=0)
- Artifact Registry: ~$0.10/GB/month (minimal)

**Compute (on-demand):**
- Step 10 embeddings: ~$2-4 per run
- Future training jobs: ~$5-20 per run (T4 preemptible)

**Total monthly (idle):** <$40 (mostly Memorystore if left running)

**Teardown checklist:** See `docs/readiness_report_final.md` (Step 23)

---

## Links

- **GCP Console:** https://console.cloud.google.com/?project=plotpointe
- **Vertex AI:** https://console.cloud.google.com/vertex-ai?project=plotpointe
- **BigQuery:** https://console.cloud.google.com/bigquery?project=plotpointe
- **GCS Bucket:** https://console.cloud.google.com/storage/browser/plotpointe-artifacts?project=plotpointe
- **Pub/Sub:** https://console.cloud.google.com/cloudpubsub?project=plotpointe
- **GPU Quotas:** https://console.cloud.google.com/iam-admin/quotas?project=plotpointe&service=aiplatform.googleapis.com&location=us-central1

