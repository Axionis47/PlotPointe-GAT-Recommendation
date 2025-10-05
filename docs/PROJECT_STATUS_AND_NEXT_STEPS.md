## Project status and next steps — PlotPointe GNN Recommender

Updated: 2025-10-05 (local)
Owner: PlotPointe
Artifacts root: gs://plotpointe-artifacts/

### Executive summary
- Data/embeddings/graphs are complete and validated (Phases 1–3).
- LightGCN baselines (Phase 4) failed repeatedly due to L4 capacity in us-central1; no baseline metrics captured.
- GAT training (Phase 5) executed with both a Custom PyTorch GAT and a PyG GAT, including ablations. Seven jobs succeeded; two earlier jobs failed due to capacity.
- Best model so far: PyG GAT, item_features=text-only (384d), loss=BPR, layers=2, heads=1.
  - Test NDCG@20 ≈ 0.0199, Recall@20 ≈ 0.0520 under 1-positive + 1000-negative evaluation.
- Next: add personalization and retrieval. Recommended production path is two-stage (retrieval → GNN rerank), with optional sequential model (SASRec) for further lift.

Links
- Phase plan: PHASES.md
- Data baseline and interpretation: DATA_BASELINE.md
- GAT architecture/design: docs/PHASE5_GAT_DESIGN.md
- Ablation results (detailed table): docs/PHASE5_ABLATIONS_REPORT.md

---

### Phase-by-phase status
- Phase 0 — CI/CD & Guardrails: COMPLETE
- Phase 1 — Data readiness: COMPLETE
  - Cardinalities (staging snapshot): 1,689,116 interactions; 498,196 items; 192,403 users; ~63k active items.
- Phase 2 — Embeddings: COMPLETE
  - Text (384d), Image (512d), Fused projection (128d) for interacted items and full catalog.
- Phase 3 — Graph construction: COMPLETE
  - User–Item bipartite and Item–Item kNN graphs (text/fused interacted subsets).
- Phase 4 — Training baseline (LightGCN): FAILED (capacity)
  - Jobs failed with “Resources are insufficient in region: us-central1.”
- Phase 5 — Training target (GAT): COMPLETE (metrics collected)
  - Trainers: scripts/train_gat_custom.py and scripts/train_gat_pyg.py
  - Vertex AI: g2-standard-4, 1× L4 GPU, us-central1
  - Results summarized below; full details in PHASE5_ABLATIONS_REPORT.md
- Phase 6 — Serving + Monitoring: NOT STARTED

---

### Data and evaluation protocol
- Split: Per-user chronological split (leave-last for val/test).
- Evaluation: For each user, 1 positive + 1000 sampled negatives; macro-averaged metrics.
- Metrics: Recall@K, NDCG@K (K=10,20). With 1 positive, HitRate@K = Recall@K.
- Note: Small absolute values are normal under sampled-negative protocols; relative differences across variants are the key signal.

---

### GAT results — key takeaways (Phase 5)
- Best overall variant: PyG GAT (layers=2, heads=1, BPR), item_features=text-only (384d)
  - Validation NDCG@20 ≈ 0.0196; Test NDCG@20 ≈ 0.0199; Test Recall@20 ≈ 0.0520
- Feature ablation: text-only > fused(128d)
  - 384d text preserves more semantic detail; the 128d fused projection likely under-capacity for this dataset.
- Loss ablation: BPR > BCE
  - Pairwise ranking aligns better with Recall/NDCG.
- Depth ablation: 2 layers >> 1 layer
  - Two-hop message passing captures co-engagement/co-purchase structure.
- Heads ablation: 1 head > 2 heads at same hidden size
  - With fixed hidden_dim, splitting into more heads reduces per-head capacity; scale hidden_dim with heads if increasing heads.
- Custom vs PyG: PyG outperformed Custom implementations in these runs (likely due to optimized kernels and battle-tested ops).

Artifacts
- Models/metrics: gs://plotpointe-artifacts/models/gat/{custom,pyg}/{checkpoints,metrics_*}
- Code uploaded for training: gs://plotpointe-artifacts/code/train/

---

### Interpreting the data baseline
- Extreme sparsity typical of retail; average ~8.8 items/user and ~26.8 users/item among ~63k active items.
- Cold start is significant (many catalog items have no interactions). Content features are critical at serve time.
- Implications for modeling: prioritize content-aware retrieval and two-hop graph/contextual signals in reranking.

---

### Production-minded recommendations

P0 — Quick wins (no new training required)
- Post-ranking personalization blend on top-K:
  - final_score = w1·GAT + w2·cos(user_text, item_text) + w3·novelty − w4·seen_recently
  - Compute user_text as a decayed average of the last N item text embeddings; grid-search weights on validation.
- Engineering hygiene: bake a custom container with pre-installed PyTorch/PyG to remove cold-start pip install time.
- Eval efficiency: reduce eval frequency or negatives for faster iteration during exploration; restore full eval for final runs.

P1 — Two-stage architecture (recommended path)
- Retrieval: Two-tower model (user sequence pooling → user vector; item text → item vector). Train with in-batch negatives, index with FAISS/ScaNN.
- Rerank: Use current best PyG GAT to rerank the top 200–1000 retrieved candidates per user.
- Score blend: α·(two-tower) + β·(GAT) + γ·(popularity prior); tune α,β,γ.
- Benefits: Strong candidate recall, robust cold-start coverage via content, efficient serving.

P2 — Sequential reranker (personalization lift)
- Train SASRec/BERT4Rec over user sequences (max len 50–100). Use as reranker or as the user tower for retrieval.
- Expect 5–20% relative gains in NDCG/Recall on Amazon-like datasets, particularly for users with ≥5 interactions.

P3 — Modeling and data improvements
- Fusion capacity: Increase fused dimension to 256–512 or switch to late fusion (concat text+image then project) so multimodal can beat text-only.
- Hard negatives: Mine negatives near positives using ANN to strengthen the BPR signal.
- Temporal weighting: Weight recent interactions higher; add edge dropout or time-aware edges in the graph.
- Deeper GNN with sampling: Consider PinSAGE/GraphSAGE with neighborhood sampling if exploring >2 layers.

P4 — Serving, monitoring, and safety
- Export a serving artifact for the chosen reranker; implement a minimal inference service.
- Candidate service: ANN index service for retrieval (ScaNN/FAISS), with periodic index refresh.
- Monitoring: Track request volumes, latency, coverage, popularity bias, drift in text distributions; add shadow A/B hooks for future online tests.

---

### Prioritized next steps (proposal)
1) P0 personalization blend experiment (1–2 days): compute user_text/novelty features offline, re-score current top-K, and measure lift on val/test.
2) P1 two-tower retrieval (3–5 days): train user-sequence/text two-tower, build FAISS index; measure candidate Recall@200/500; rerank with GAT and report NDCG/Recall deltas.
3) Optional P2 SASRec (5–7 days): train and integrate as reranker or user tower; compare against P1.
4) Fusion revisit (parallel/optional): try fused dim 256–512 or late fusion; re-run key ablations only.
5) Containerization (DevEx): bake a base image with torch/pyg deps to cut job startup time and cost.

Dependencies and notes
- Region remains us-central1 (L4). Preemptible L4 quota would unlock FLEX_START scheduling and lower cost; otherwise continue on-demand.
- Strict sequencing is possible for future runs; current Phase 5 batch has completed.

---

### Appendix — recent Vertex AI job outcomes
Succeeded
- train-gat-pyg-1759630810 — ID 2933173440070287360
- train-gat-custom-txt-1759633155 — ID 8497933729639956480
- train-gat-custom-bce-1759633157 — ID 4512248059417067520
- train-gat-pyg-h2-1759633158 — ID 1003943949695451136
- train-gat-pyg-txt-1759633159 — ID 5569186596965580800
- train-gat-pyg-l1-1759633160 — ID 5615629968122839040
- train-gat-pyg-bce-1759633161 — ID 1159318136839733248

Failed (capacity/scheduling)
- train-gat-custom-1759630809 — ID 4207692134616137728
- train-gat-custom-l1-1759633156 — ID 4309586076185395200

