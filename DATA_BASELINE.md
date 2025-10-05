# Data Baseline and Training Fit (v1)

Last updated: 2025-10-04
Owner: PlotPointe (GAT-Recommendation)
Scope: What data we have, where it lives, how it maps to training; gates and next steps

---

## 1) Dataset overview

- Domain: Amazon Electronics
- Problem: Recommendation via user–item interactions and multi-modal (text, image) item content
- Cardinalities (exact, as of current staging snapshot):
  - Interactions (rows): 1,689,116
  - Items (unique asin): 498,196
  - Users (unique user_id): 192,403

We confirmed the staged data exists in GCS and passed validation checks locally (schema & integrity).

---

## 2) Data locations (GCS)

- Staging prefix (inputs to pipelines):
  - gs://plotpointe-artifacts/staging/amazon_electronics/items.parquet
  - gs://plotpointe-artifacts/staging/amazon_electronics/interactions.parquet
- Downstream artifact prefixes (outputs):
  - Embeddings: gs://plotpointe-artifacts/embeddings/
  - Graphs: gs://plotpointe-artifacts/graphs/
  - Models: gs://plotpointe-artifacts/models/

---

## 3) Schemas and feature availability

- items.parquet
  - asin: string (primary key)
  - title: string (text feature)
  - brand: string (text feature)
  - categories: array<string> (text feature)
  - price: float (optional numeric)
  - image_url: string (optional; for CLIP image embeddings)

- interactions.parquet
  - user_id: string (primary key component)
  - asin: string (FK → items.asin)
  - rating: float in [1.0, 5.0] (edge weight)
  - ts: timestamp (UTC) or epoch seconds (recency; streaming windows)

Coverage notes (to be quantified in cloud smoke run):
- Title/brand/categories coverage typically high for title, lower for brand/categories depending on source.
- image_url coverage is partial; only a subset of items have valid HTTP(S) URLs.


### 3.1 Column-level datatypes and availability

Items table (items.parquet):
- asin
  - Type: string (Parquet: BYTE_ARRAY/UTF8; pandas: object/string)
  - Nullability: not null; unique key
  - Availability: 100% expected
  - Normalization: treated as opaque ID; mapping json written during graph build
  - Downstream: item node id, joins, feature alignment
- title
  - Type: string
  - Nullability: optional (empty allowed)
  - Availability: high; measured as non-empty coverage
  - Normalization: whitespace trim; used raw in text features
  - Downstream: text embeddings
- brand
  - Type: string
  - Nullability: optional
  - Availability: variable; measured as non-empty coverage
  - Normalization: prefixed as "Brand: {brand}" in text features
  - Downstream: text embeddings
- categories
  - Type: array<string> (may arrive JSON-encoded); normalized to list[str]
  - Nullability: optional
  - Availability: variable; measured as non-empty list coverage
  - Normalization: join tokens as space-delimited text for embeddings
  - Downstream: text embeddings
- price
  - Type: float32/float64 (>= 0 when present)
  - Nullability: optional
  - Availability: measured as non-null ratio
  - Downstream: not used in baseline training; reserved for future ablations
- image_url
  - Type: string (HTTP/HTTPS expected)
  - Nullability: optional
  - Availability: measured as valid http(s) ratio
  - Normalization: filtered by scheme; download with timeout/retry
  - Downstream: image embeddings (CLIP)

Interactions table (interactions.parquet):
- user_id
  - Type: string
  - Nullability: not null
  - Availability: 100% expected
  - Downstream: user node id; mapping json written during graph build
- asin
  - Type: string; FK to items.asin
  - Nullability: not null
  - Availability: 100% expected; alignment to items measured
  - Downstream: joins, edge construction
- rating
  - Type: float32/float64 in [1.0, 5.0]
  - Nullability: not null
  - Availability: 100% expected; histogram recorded
  - Normalization: scaled to [0,1] for edge weights ((r-1)/4)
  - Downstream: edge weights for UI graph; optional loss weighting
- ts (timestamp)
  - Type: timestamp (UTC) or int64 epoch seconds; normalized to UTC timestamp
  - Nullability: not null (invalid parse considered missing)
  - Availability: 100% expected; min/max and recent_90d computed
  - Downstream: streaming windows, potential time decay

Availability metrics source:
- Computed per run in embeddings smoke/evidence step; logged to Vertex Experiments and optionally written to BigQuery for drift baselines.

---

### 3.2 Phase 1 computed coverage and stats (staging snapshot)

- Interactions
  - Rows: 1,689,116; Users: 192,403; Items in interactions (unique asin): 63,001
  - Rating histogram (count | pct):
    - 1.0: 108,723 | 6.44%
    - 2.0: 82,133 | 4.86%
    - 3.0: 142,254 | 8.42%
    - 4.0: 347,029 | 20.55%
    - 5.0: 1,008,977 | 59.73%
  - Duplicates on (user_id, asin, ts): 0.00%
  - Timestamp range: 2000-01-01 → 2014-07-23; fraction in last 90d: 0.00%
  - FK integrity (asin present in items): 100.00%

- Items coverage
  - title non-empty: 98.59%
  - brand non-empty: 28.38%
  - categories non-empty list: 0.00%
  - image_url http(s): 99.96%
  - price non-null: 78.22% (min 0.01 | median 19.99 | max 999.99)


## 4) Data quality and readiness

Validation checks run locally (PASS):
- Non-null: user_id, asin, rating, ts (interactions)
- Rating bounds: [1.0, 5.0]
- Timestamp bounds: reasonable range (>= 2000-01-01, <= now)
- Uniqueness: no duplicates on (user_id, asin, ts)
- FK integrity: interactions.asin ⊆ items.asin
- Items: asin not-null, asin unique, price non-negative (when present)

Gate acceptance for Phase 1: PASS — Data is ready for embeddings.

---

## 5) How this data fits training

- Supervision signal:
  - From interactions: implicit/explicit feedback — we use rating as the user–item edge weight (normalized to [0,1]).
  - Optional time decay or windowing for streaming/micro-batches.

- Item features (for models):
  - Text embeddings from items.title (+ brand + categories)
  - Image embeddings from items.image_url (subset only)
  - Fusion embeddings (text+image) for items where both modalities exist; for text-only items, we will use text embeddings as fallback.

- Graphs for training:
  - User–Item bipartite graph (COO sparse): edges (user_id ↔ asin) weighted by normalized rating (and later optional time decay).
  - Item–Item kNN graphs:
    - ii_edges_txt.npz from cosine of text embeddings
    - ii_edges_fused.npz from cosine of fused embeddings (subset)

- Training datasets:
  - LightGCN baseline: uses user–item graph (weights), optionally augmented by item features for evaluation.
  - GAT target: consumes user–item graph and item node features (txt or fused), optionally includes item–item edges for structure.

---

## 6) Streaming readiness (micro-batches)

- Windowing:
  - We will add optional flags --window-start/--window-end (UTC ISO) to embeddings, graphs, and training CLIs.
  - For interactions, filtering by ts supports rolling/micro-batch ingestion in production.
- Idempotence:
  - Artifact versioning by run name (includes window) ensures safe re-runs and exactly-once semantics at the artifact level.
- Chunking:
  - embed_image already supports chunking; we’ll align text and fusion for consistent throughput.

---

## 7) Metrics to collect in the embeddings smoke run (to populate)

We will compute these in Vertex smoke runs and publish to Vertex Experiments:
- Items coverage: title, brand, categories, image_url HTTP coverage
- Price stats: non-null ratio, min/median/max
- Interactions: counts, rating histogram (1–5), per-user and per-item interaction stats (mean/median/p90/p99)
- Timestamp: min/max; fraction recent (last 90 days)
- Alignment: fraction of interactions.asin present in items (FK)
- Duplicates fraction on (user_id, asin, ts)

These populate the Data Card and support drift monitoring later.

---

## 8) Acceptance criteria for baseline

- Data exists in GCS staging and passes validation (DONE)
- Vertex smoke runs for embeddings complete and publish coverage/histograms
- Artifacts written with run names and window annotations (even if window unset)
- Metrics logged to Vertex Experiments under study=baseline

---

## 9) Risks and mitigations

- Partial image coverage → fusion yields subset-only embeddings
  - Mitigation: text fallback for items without images; ablation: fused-only vs fallback
- Skewed ratings distribution (e.g., 5-star heavy)
  - Mitigation: normalize to [0,1], consider time decay and/or implicit weighting
- Long-tail items and users
  - Mitigation: cap k in kNN, add ii graph sparsity thresholds; sampling for training steps

---

## 10) Next steps (towards training)

1) Run Vertex embeddings smoke (text CPU, image L4, fusion L4) and record metrics
2) Package graphs CLIs (no logic change) and run graph jobs on Vertex
3) Implement LightGCN baseline training with Vertex Experiment logging
4) Add evaluation harness (Recall@K, NDCG@K) over fixed splits
5) Proceed to GAT training and serving baseline; drift monitoring follows

End of document.

