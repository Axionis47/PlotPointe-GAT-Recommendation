# Data Contracts: Amazon Electronics 5-core (staged to gs://plotpointe-artifacts)

Scope
- interactions.parquet: user–item interactions derived from reviews (Electronics 5-core)
- items.parquet: product metadata for Electronics category

Storage layout
- gs://plotpointe-artifacts/staging/amazon_electronics/interactions.parquet
- gs://plotpointe-artifacts/staging/amazon_electronics/items.parquet

interactions.parquet
- Columns and types
  - user_id STRING (REQUIRED)
  - asin STRING (REQUIRED)
  - rating FLOAT64 (REQUIRED)  — 1.0–5.0 inclusive
  - ts TIMESTAMP (REQUIRED)    — review event time
  - verified BOOL (NULLABLE)
  - summary STRING (NULLABLE)
  - helpful_up INT64 (NULLABLE)
  - helpful_down INT64 (NULLABLE)
  - source STRING (NULLABLE, default="amazon-electronics-5core")
- Constraints
  - Non‑null: user_id, asin, rating, ts
  - Bounds: rating in [1,5]
  - Timestamp bounds: 2000‑01‑01 ≤ ts ≤ now()+1d
  - Uniqueness: (user_id, asin, ts) unique per file (dedup step on load)
  - Foreign key: asin must exist in items.parquet.asin (referential integrity)

items.parquet
- Columns and types
  - asin STRING (REQUIRED)
  - title STRING (NULLABLE)
  - brand STRING (NULLABLE)
  - price FLOAT64 (NULLABLE, ≥ 0 when present)
  - categories ARRAY<STRING> (NULLABLE)
  - image_url STRING (NULLABLE)
  - source STRING (NULLABLE, default="amazon-electronics-meta")
- Constraints
  - Non‑null: asin
  - Uniqueness: asin unique
  - Price bounds: price >= 0 when present

Idempotency and dedupe
- Landing job generates a stable row key for interactions as hash(user_id, asin, ts)
- Upserts into downstream stores use this key (insert_id for Pub/Sub→BQ or MERGE on key)
- Files are partitioned by load date; merges ensure eventual de‑duplication if re‑staged

Timestamp/timezone
- ts stored as UTC; derived from unixReviewTime (seconds)

Rejection handling
- Rows failing non‑null/bounds/fk checks are written to *_rejects.parquet with a reason column and summarized in validation reports.

