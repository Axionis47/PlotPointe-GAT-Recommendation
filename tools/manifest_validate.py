#!/usr/bin/env python3
"""
Validate that a feature manifest references existing GCS prefixes and is minimally well-formed.
Safe, read-only checks. Requires GOOGLE_APPLICATION_CREDENTIALS or default ADC.
"""
from __future__ import annotations
import argparse
import json
from typing import Tuple
from urllib.parse import urlparse

from google.cloud import storage

from plotpointe.feature_registry import load_manifest, resolve_paths


def _split_gs_uri(gs_uri: str) -> Tuple[str, str]:
    assert gs_uri.startswith("gs://"), f"Not a GCS URI: {gs_uri}"
    p = urlparse(gs_uri)
    bucket = p.netloc
    # urlparse puts the path like '/path/to/prefix'
    prefix = p.path[1:] if p.path.startswith('/') else p.path
    return bucket, prefix.rstrip('/')


def _prefix_exists(client: storage.Client, bucket: str, prefix: str) -> bool:
    b = client.bucket(bucket)
    # list_blobs is efficient for prefix existence check (stops after first result)
    it = client.list_blobs(b, prefix=prefix, max_results=1)
    return any(True for _ in it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-id', required=True, help='Manifest id (from manifests/registry.json)')
    args = ap.parse_args()

    man = load_manifest(args.manifest_id)
    paths = resolve_paths(man)

    client = storage.Client()

    checks = {
        'staging_prefix': paths['staging_prefix'],
        'embeddings_prefix': paths['embeddings_prefix'],
        'graphs_prefix': paths['graphs_prefix'],
    }

    results = {}
    for k, gs in checks.items():
        bucket, prefix = _split_gs_uri(gs)
        ok = _prefix_exists(client, bucket, prefix)
        results[k] = {'uri': gs, 'exists': ok}

    print(json.dumps({'manifest_id': man.id, 'checks': results}, indent=2))

    # Non-zero exit on missing prefixes for CI gate usage
    if not all(v['exists'] for v in results.values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

