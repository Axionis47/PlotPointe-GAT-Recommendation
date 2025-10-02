#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import os
import sys
import time
import hashlib
from datetime import datetime, timezone
from urllib.request import urlopen

import pandas as pd

# Requires: pandas, pyarrow; gsutil available for upload

REVIEWS_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"


def parse_loose_json_gz(url_or_path):
    # Stream parse "loose" json lines (python literals) and convert to dict
    if url_or_path.startswith("http"):
        f = urlopen(url_or_path)
        gz = gzip.GzipFile(fileobj=io.BytesIO(f.read()))
    else:
        gz = gzip.open(url_or_path, 'rb')
    for line in gz:
        # eval to python dict then dumps+loads to ensure standard json types
        try:
            obj = eval(line)
        except Exception:
            # fallback to json.loads if strict json
            obj = json.loads(line)
        yield obj


def to_interactions_df(objs):
    rows = []
    for o in objs:
        user_id = o.get('reviewerID')
        asin = o.get('asin')
        rating = o.get('overall')
        uts = o.get('unixReviewTime')
        ts = None
        if uts is not None:
            ts = datetime.fromtimestamp(int(uts), tz=timezone.utc)
        helpful = o.get('helpful', [None, None])
        verified = o.get('verified')
        summary = o.get('summary')
        rows.append({
            'user_id': user_id,
            'asin': asin,
            'rating': float(rating) if rating is not None else None,
            'ts': ts,
            'verified': bool(verified) if isinstance(verified, bool) else None,
            'summary': summary,
            'helpful_up': int(helpful[0]) if isinstance(helpful, list) and len(helpful) >= 1 and helpful[0] is not None else None,
            'helpful_down': int(helpful[1]) if isinstance(helpful, list) and len(helpful) >= 2 and helpful[1] is not None else None,
            'source': 'amazon-electronics-5core'
        })
    df = pd.DataFrame(rows)
    # Basic cleaning/dedup within file
    df = df.drop_duplicates(subset=['user_id', 'asin', 'ts'])
    return df


def to_items_df(objs):
    rows = []
    for o in objs:
        asin = o.get('asin')
        title = o.get('title')
        brand = o.get('brand')
        price = o.get('price')
        img = o.get('imUrl')
        cats = o.get('categories')
        # Flatten categories as list of strings
        flat_cats = []
        if isinstance(cats, list):
            for c in cats:
                if isinstance(c, list):
                    flat_cats.extend([str(x) for x in c])
                elif c is not None:
                    flat_cats.append(str(c))
        rows.append({
            'asin': asin,
            'title': title,
            'brand': brand,
            'price': float(price) if isinstance(price, (int, float)) else None,
            'categories': flat_cats if flat_cats else None,
            'image_url': img,
            'source': 'amazon-electronics-meta'
        })
    df = pd.DataFrame(rows)
    # Deduplicate on asin, keeping first non-null title/brand/price
    df = df.sort_values(['asin']).drop_duplicates(subset=['asin'], keep='first')
    return df


def write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def upload_to_gcs(local_path, gcs_uri):
    rc = os.system(f"gsutil cp -n {local_path} {gcs_uri}")
    if rc != 0:
        raise SystemExit(f"Upload failed: {local_path} -> {gcs_uri}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', default=REVIEWS_URL)
    ap.add_argument('--meta', default=META_URL)
    ap.add_argument('--out-dir', default='data/staging/amazon_electronics')
    ap.add_argument('--gcs-prefix', default='gs://plotpointe-artifacts/staging/amazon_electronics')
    args = ap.parse_args()

    print("[STAGE] Reading reviews...")
    inter_df = to_interactions_df(parse_loose_json_gz(args.reviews))
    print(f"[STAGE] Interactions rows: {len(inter_df):,}")

    print("[STAGE] Reading metadata...")
    items_df = to_items_df(parse_loose_json_gz(args.meta))
    print(f"[STAGE] Items rows: {len(items_df):,}")

    # Basic contract-enforced filtering (non-nulls and bounds); rejects collected
    rejects = []
    def reject(reason, mask):
        if mask.any():
            rej = inter_df[mask].copy()
            rej['reject_reason'] = reason
            rejects.append(rej)

    # Interactions constraints
    null_mask = inter_df['user_id'].isna() | inter_df['asin'].isna() | inter_df['rating'].isna() | inter_df['ts'].isna()
    reject('null_required_field', null_mask)
    inter_df = inter_df[~null_mask]

    bounds_mask = (inter_df['rating'] < 1.0) | (inter_df['rating'] > 5.0)
    reject('rating_out_of_bounds', bounds_mask)
    inter_df = inter_df[~bounds_mask]

    tmin = datetime(2000,1,1, tzinfo=timezone.utc)
    tmax = datetime.now(tz=timezone.utc)
    ts_mask = (inter_df['ts'] < tmin) | (inter_df['ts'] > tmax)
    reject('timestamp_out_of_bounds', ts_mask)
    inter_df = inter_df[~ts_mask]

    # FK check: keep only rows where asin in items
    asin_set = set(items_df['asin'].dropna().unique().tolist())
    fk_mask = ~inter_df['asin'].isin(asin_set)
    reject('fk_missing_item', fk_mask)
    inter_df = inter_df[~fk_mask]

    # Save outputs
    inter_out = os.path.join(args.out_dir, 'interactions.parquet')
    items_out = os.path.join(args.out_dir, 'items.parquet')
    write_parquet(inter_df, inter_out)
    write_parquet(items_df, items_out)

    if rejects:
        rej_df = pd.concat(rejects, ignore_index=True)
        write_parquet(rej_df, os.path.join(args.out_dir, 'interactions_rejects.parquet'))

    # Upload
    upload_to_gcs(inter_out, f"{args.gcs_prefix}/interactions.parquet")
    upload_to_gcs(items_out, f"{args.gcs_prefix}/items.parquet")
    if os.path.exists(os.path.join(args.out_dir, 'interactions_rejects.parquet')):
        upload_to_gcs(os.path.join(args.out_dir, 'interactions_rejects.parquet'), f"{args.gcs_prefix}/interactions_rejects.parquet")

    print("[STAGE] Done.")


if __name__ == '__main__':
    main()

