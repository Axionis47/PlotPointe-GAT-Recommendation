#!/usr/bin/env python3
import argparse
import os
from datetime import datetime, timezone

import pandas as pd

# Validation covering the GE-style checks without requiring GE at runtime


def validate_interactions(df: pd.DataFrame, items_df: pd.DataFrame):
    results = {}

    # Non-null checks
    results['not_null_user_id'] = df['user_id'].notna().all()
    results['not_null_asin'] = df['asin'].notna().all()
    results['not_null_rating'] = df['rating'].notna().all()
    results['not_null_ts'] = df['ts'].notna().all()

    # Rating bounds [1.0, 5.0]
    results['rating_range'] = ((df['rating'] >= 1.0) & (df['rating'] <= 5.0)).all()

    # Timestamp bounds [2000-01-01, now]
    tmin = pd.Timestamp(datetime(2000, 1, 1, tzinfo=timezone.utc))
    tmax = pd.Timestamp(datetime.now(tz=timezone.utc))
    ts_series = pd.to_datetime(df['ts'], utc=True, errors='coerce')
    results['ts_bounds'] = ((ts_series >= tmin) & (ts_series <= tmax)).all()

    # Uniqueness on (user_id, asin, ts)
    results['unique_triplet'] = not df.duplicated(subset=['user_id', 'asin', 'ts']).any()

    # Foreign key: asin must be in items.asin
    asin_set = set(items_df['asin'].dropna().unique().tolist())
    results['fk_asin_in_items'] = df['asin'].isin(asin_set).all()

    return results


def validate_items(df: pd.DataFrame):
    results = {}

    results['not_null_asin'] = df['asin'].notna().all()
    results['asin_unique'] = not df['asin'].duplicated().any()
    price_non_null = df['price'].dropna()
    results['price_non_negative'] = (price_non_null >= 0).all()

    return results


def summarize(results_dict):
    lines = []
    overall = True
    for name, res in results_dict.items():
        status = 'PASS' if res else 'FAIL'
        if not res:
            overall = False
        lines.append(f"- {name}: {status}")
    return overall, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local-dir', default='data/staging/amazon_electronics')
    ap.add_argument('--gcs-prefix', default='gs://plotpointe-artifacts/staging/amazon_electronics')
    args = ap.parse_args()

    inter_path = os.path.join(args.local_dir, 'interactions.parquet')
    items_path = os.path.join(args.local_dir, 'items.parquet')

    if not (os.path.exists(inter_path) and os.path.exists(items_path)):
        raise SystemExit("Staged parquet files not found; run the staging script first.")

    inter_df = pd.read_parquet(inter_path)
    items_df = pd.read_parquet(items_path)

    inter_res = validate_interactions(inter_df, items_df)
    items_res = validate_items(items_df)

    overall_inter, summary_inter = summarize(inter_res)
    overall_items, summary_items = summarize(items_res)

    overall = overall_inter and overall_items

    print("Validation summary (interactions):")
    print(summary_inter)
    print("\nValidation summary (items):")
    print(summary_items)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == '__main__':
    main()

