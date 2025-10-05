# Data Manifests and Feature Registry (Phase 4)

Lightweight, read-only manifests that declare which dataset/feature/graph prefixes a training run should use. This enables clean separation between features and models and lets you compare runs against a declared feature set.

- No behavior changes to training scripts; all existing jobs continue to work.
- You may optionally use the registry to resolve prefixes and then pass them as flags to training.

## Files added
- `manifests/registry.json` – ids -> manifest file mapping
- `manifests/features/amazon_electronics_v1.json` – baseline manifest using existing, stable prefixes
- `plotpointe/feature_registry.py` – loader/validator utilities
- `tools/manifest_validate.py` – read-only validation against GCS (prefix existence)

## Manifest format (JSON)

```
{
  "id": "amazon_electronics_v1",
  "dataset": "amazon_electronics",
  "paths": {
    "staging_prefix": "gs://.../staging/amazon_electronics",
    "embeddings_prefix": "gs://.../embeddings",
    "graphs_prefix": "gs://.../graphs"
  },
  "defaults": { "item_features": "fused" }
}
```

## How to use

1) Resolve prefixes from a manifest id:

```
python -c "from plotpointe.feature_registry import load_manifest, resolve_paths; man=load_manifest('amazon_electronics_v1'); print(resolve_paths(man))"
```

2) Pass them to training (same flags you already use):

```
python -m scripts.train_gat_pyg \
  --project-id plotpointe --region us-central1 \
  --staging-prefix $(jq -r .paths.staging_prefix manifests/features/amazon_electronics_v1.json) \
  --embeddings-prefix $(jq -r .paths.embeddings_prefix manifests/features/amazon_electronics_v1.json) \
  --graphs-prefix $(jq -r .paths.graphs_prefix manifests/features/amazon_electronics_v1.json) \
  --item-features fused ...
```

3) Validate the manifest references exist in GCS (read-only):

```
python tools/manifest_validate.py --manifest-id amazon_electronics_v1
```

- Exit code 0 if all prefixes exist; 2 otherwise (so you can wire it into CI if desired).

## Notes
- We intentionally reference stable top-level prefixes that your jobs already use to avoid churn.
- You can add new manifests (e.g., `amazon_electronics_v2.json`) as features evolve, without changing training code.

