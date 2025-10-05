# PlotPointe Phased Plan (7 Phases)

Last updated: 2025-10-04
Owner: PlotPointe (GAT-Recommendation)
Scope: Phase-by-phase plan with inputs, outputs, gates; proceed only when each gate passes

---

## Phase 0 — CI/CD & Environment Guardrails
Purpose: Make runs reproducible and safe to promote
Inputs: Repository, pyproject tool configs, Vertex base images
Outputs: CI workflow, smoke checks, env policy docs
Gate (Pass to Phase 1 when): CI green on main/PRs, CLI smoke OK, env policy set (L4-only)

## Phase 1 — Data Readiness (Staging + Validation)
Purpose: Confirm inputs are clean, aligned, and documented
Inputs: GCS staging parquet (items.parquet, interactions.parquet)
Outputs: Validation report, updated DATA_BASELINE.md (schemas, datatypes, availability)
Gate: Validation PASS (non-null, bounds, FK, duplicates) and documentation updated

## Phase 2 — Embeddings (Text, Image, Fusion)
Purpose: Produce multimodal item representations
Inputs: items.parquet (+ optional interactions)
Outputs: embeddings/{txt.npy, img.npy, fused.npy, *_meta.json}, img_items.parquet
Gate: Vertex jobs succeed; coverage/histograms logged; artifacts versioned

## Phase 3 — Graphs (User–Item, Item–Item)
Purpose: Build training graphs from embeddings and interactions
Inputs: embeddings/*, interactions.parquet
Outputs: graphs/{ui_edges.npz, node_maps.json, ui_stats.json, ii_edges_*.npz, *_stats.json}
Gate: Sanity stats within range; node alignment correct; windowed outputs supported

- Decision (2025-10-04): Build Item–Item kNN over interactions-only items (≈63,001 items)
  - Rationale: Interacted items are a strong proxy for importance/popularity and align with training/evaluation targets; this reduces compute and emphasizes actionable neighbors.
  - Implications: Candidate recall focuses on active catalog; cold-start items remain reachable via User–Item edges and content features (text/fused) during ranking.


## Phase 4 — Training Baseline (LightGCN)
Purpose: Establish fast, reliable baseline with clear metrics
Inputs: ui_edges.npz (+ optional ii_edges), node_maps, item features (optional)
Outputs: models/lightgcn/checkpoints/*.pt, metrics.json (Recall@K/NDCG@K), run manifests
Gate: Training completes; metrics reproducible and above minimal thresholds

## Phase 5 — Training Target (GAT) + Evaluation
Purpose: Train target model with item features, run ablations and report
Inputs: ui_edges.npz, item features (txt/fused), (optional) ii_edges_*.npz
Outputs: models/gat/checkpoints/*.pt, metrics.json, ablation report
Gate: Target metrics >= baseline; ablations logged; evaluation report approved

## Phase 6 — Serving + Monitoring (Drift)
Purpose: Prove inference path and activate monitoring
Inputs: trained model, item embeddings, catalogs
Outputs: ANN index, minimal API (Cloud Run), drift metrics in BigQuery, alerts
Gate: API smoke passes with latency targets; drift pipeline logs metrics and alerts

