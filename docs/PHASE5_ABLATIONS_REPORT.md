## Phase 5 — GAT Ablation Results (Vertex AI us-central1)

Updated: 2025-10-05 (local)
Artifacts: gs://plotpointe-artifacts/models/gat/{custom,pyg}/{checkpoints,metrics_*}

### Summary
- Best overall (by val NDCG@20): PyG GAT, item_features=txt, loss=BPR, layers=2, heads=1
  - Val NDCG@20 = 0.0196; Test NDCG@20 = 0.0199; Test Recall@20 = 0.0520
- Feature ablation: text-only (384d) > fused (128d) on these runs — likely because 384d text preserves more semantic detail than the fused 128d projection.
- Loss ablation: BPR > BCE for ranking metrics, as expected.
- Depth ablation: 2 layers >> 1 layer (1-hop underfits co-purchase structure).
- Heads ablation (PyG): 1 head > 2 heads at same hidden size (2 heads underperformed here).
- Custom GAT variants underperformed PyG across the board for this setting; PyG kernels and optimizations likely help.

### Variant comparison (validation and test)

| family | item_features | loss | layers | heads | best val NDCG@20 | val R@10 | val R@20 | val N@10 | val N@20 | test R@10 | test R@20 | test N@10 | test N@20 |
|:------:|:-------------:|:----:|:------:|:-----:|------------------:|---------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|
| pyg    | txt           | bpr  | 2      | 1     | 0.0196           | 0.0331   | 0.0505   | 0.0152   | 0.0196   | 0.0338    | 0.0520    | 0.0153    | 0.0199    |
| pyg    | fused         | bpr  | 2      | 1     | 0.0161           | 0.0223   | 0.0432   | 0.0109   | 0.0161   | 0.0225    | 0.0433    | 0.0108    | 0.0160    |
| pyg    | fused         | bce  | 2      | 1     | 0.0125           | 0.0178   | 0.0350   | 0.0082   | 0.0125   | 0.0177    | 0.0354    | 0.0079    | 0.0123    |
| pyg    | fused         | bpr  | 2      | 2     | 0.0117           | 0.0161   | 0.0330   | 0.0074   | 0.0117   | 0.0156    | 0.0337    | 0.0070    | 0.0115    |
| pyg    | fused         | bpr  | 1      | 1     | 0.0027           | 0.0036   | 0.0071   | 0.0017   | 0.0026   | 0.0050    | 0.0086    | 0.0024    | 0.0033    |
| custom | txt           | bpr  | 2      | -     | 0.0079           | 0.0115   | 0.0229   | 0.0050   | 0.0079   | 0.0108    | 0.0219    | 0.0049    | 0.0077    |
| custom | fused         | bce  | 2      | -     | 0.0040           | 0.0053   | 0.0117   | 0.0024   | 0.0040   | 0.0060    | 0.0123    | 0.0030    | 0.0045    |

Notes:
- Metrics are taken from metrics_*.json emitted by each trainer at job completion; “best val NDCG@20” is the model-selected checkpoint criterion.
- Exact files (subset):
  - pyg: metrics_gat_pyg_d128_1759651719.json (fused,bpr,h1,l2), metrics_gat_pyg_d128_1759653808.json (txt,bpr,h1,l2), metrics_gat_pyg_d128_1759653014.json (fused,bpr,h2,l2), metrics_gat_pyg_d128_1759655611.json (fused,bpr,h1,l1), metrics_gat_pyg_d128_1759665437.json (fused,bce,h1,l2)
  - custom: metrics_gat_custom_d128_1759650827.json (fused,bce,l2), metrics_gat_custom_d128_1759654706.json (txt,bpr,l2)

### Interpretation (dataset-specific)
- Text signals dominate: In Amazon Electronics, titles/bullets are rich; the 384d text-only embedding outperformed the 128d fused projection. If we want fusion to win, increase fusion capacity (e.g., 256–512d) or use late fusion at the GNN layer (concatenate, then project).
- BPR aligns with ranking: Across both implementations, BPR yielded stronger ranking metrics than BCE.
- Two hops matter: 2 layers consistently outperformed 1; co-purchase/user co-engagement needs 2-hop neighborhoods.
- Attention heads: With fixed hidden size, adding heads can reduce per-head capacity; try scaling hidden_dim with heads if we revisit multi-heads.

### What to run next (optional)
- Re-run fusion with higher fused dim (e.g., 256 or 384) to see if multimodal beats text-only.
- Try eval optimizations (reduced negatives or eval frequency) to speed iteration.
- If we need a stronger custom baseline, profile attention normalization and sampling to close the gap to PyG.

