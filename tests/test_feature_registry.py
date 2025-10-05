#!/usr/bin/env python3
from pathlib import Path
from plotpointe.feature_registry import load_manifest, resolve_paths


def test_load_manifest_and_resolve_paths():
    man = load_manifest('amazon_electronics_v1')
    paths = resolve_paths(man)
    assert man.id == 'amazon_electronics_v1'
    for k in ['staging_prefix', 'embeddings_prefix', 'graphs_prefix']:
        assert k in paths and isinstance(paths[k], str) and paths[k].startswith('gs://')
    assert paths['item_features'] in {'fused', 'text', 'image'}

