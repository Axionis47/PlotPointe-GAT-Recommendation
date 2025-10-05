# Phase 5 — GAT Architecture and Decisions (Dataset‑specific)

Last updated: 2025-10-05
Owner: PlotPointe (GAT-Recommendation)
Scope: Document two complementary GAT implementations tailored to the Amazon Electronics dataset; justify choices, training strategy, and evaluation.

---

## Dataset context (key stats)
- Interactions: 1,689,116
- Users: 192,403
- Items with interactions: 63,001 (active catalog)
- Total items: 498,196 (for serving cold‑start)
- Item features: fused (128d), text (384d), image (512d); we standardize to fused 128d for Phase 5
- Graph: User–Item bipartite, symmetric edges with degree normalization (as in Phase 4)

Implications:
- Graph is moderately sparse (~3.4M directed edges after symmetrization), enabling full‑graph forward on a single L4 if epochs are tuned
- Item feature quality is high (multimodal fusion); users lack side features → learn user embeddings
- For training stability/performance, we avoid repeated backward on the same graph by using one backward per epoch with a large sampled BPR batch

## Objectives for Phase 5
- Train a target model (GAT) that matches/exceeds LightGCN baseline
- Provide two implementations:
  1) Custom GAT (pure PyTorch) — shows depth and tailoring
  2) Library GAT (PyTorch Geometric) — shows production readiness and reuse
- Evaluate with Recall@K and NDCG@K (sampled negatives, identical to Phase 4) for comparability

## Node features and model inputs
- Users: learned embeddings (d_user)
- Items: fused item embeddings (128d); projected to model hidden dim
- Unified space: concatenate users (0..n_users-1) and items (n_users..n_users+n_items-1) to share layers and simplify scoring

## Training strategy
- Loss: BPR (pairwise ranking) with uniform negatives
- Per‑epoch sampling: draw S triples (u, i_pos, i_neg) across all users
- Forward: compute node embeddings once per epoch (full graph), then compute loss over the sampled triples, and do exactly one backward() + optimizer step per epoch. This avoids autograd graph reuse while keeping memory bounded.
- Epochs: 20 (default), early stop on val NDCG@20 with patience 5

## Implementation A — Custom GAT (pure PyTorch)
- Layer: single‑head attention with LeakyReLU(0.2) and dropout on attention weights
- Normalization: attention softmax over incoming edges (per‑destination)
  - Efficient segment operations via scatter_add; logits are clamped to [-10, 10] to maintain numerical stability without extra deps
- Stack: 2 layers (hidden 128 → output 128)
- Pros: no external deps; transparent; tailored to our bipartite structure
- Cons: not as optimized as PyG; no built‑in neighbor sampling

## Implementation B — PyTorch Geometric GAT
- Uses torch_geometric.nn.GATConv on a homogeneous graph (users+items combined) with shared feature space
- Dependencies installed in the Vertex container via wheels matching torch==2.1.0+cu121
- Pros: optimized kernels; clear, concise code; easier to extend to neighbor sampling
- Cons: introduces dependencies (torch-geometric, torch-scatter, etc.)

## Hyperparameters (initial)
- Hidden dim: 128 (to match fused item features); output dim = 128
- Layers: 2
- Attention heads: 1 (keep memory bounded); can expand to 2–4 in ablations
- Dropout: 0.1 on attention weights
- Learning rate: 1e-3, weight decay: 1e-4
- Samples per epoch S: 200,000 BPR triples
- Batch for evaluation negatives: 1000 per user (same as Phase 4)

## Evaluation & Gate
- Metrics: Recall@10/20, NDCG@10/20, coverage (% users with ≥1 hit)
- Gate: Target >= LightGCN baseline; ablation table logged (features: fused vs text; heads: 1 vs 2)

## Risks and mitigations
- Capacity wait on L4: keep us-central1 (accepted region); DWS requires preemptible L4 quota
- Runtime with full‑graph forward per epoch: keep epochs modest (<=20), single head, hidden 128
- Numeric stability in custom attention: clamp logits; unit tests on small graphs (optional)

## Artifacts
- Checkpoints: gs://plotpointe-artifacts/models/gat/{custom,pyg}/checkpoints/*.pt
- Metrics JSON: gs://plotpointe-artifacts/models/gat/{custom,pyg}/metrics_*.json

## Next
- Implement trainers and Vertex configs, submit jobs, and report results against Phase 4 baseline.

