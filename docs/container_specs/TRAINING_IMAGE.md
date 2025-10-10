# Training Image (Vertex AI)

This container pre-installs PyTorch 2.1.0 (CUDA 12.1), PyG wheels, and core libs so Vertex jobs skip cold-start pip installs.

- Base: `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-1.py310`
- Image: `ghcr.io/<owner>/gat-recsys:<tag>` (built by GitHub Actions)
- Entrypoints (baked): `python -m scripts.train_gat_pyg` or `python -m scripts.train_gat_custom`

## Build & Push (CI)

- Workflow: `.github/workflows/container.yml`
- Trigger: push to main (Dockerfile/workflow changes) or manual `workflow_dispatch`
- Requires: no extra secrets for GHCR in most orgs (uses `GITHUB_TOKEN`). If your org restricts package publishing, create `GHCR_PAT` with `write:packages` and swap the login step.

## Use in Vertex AI

Two image-based configs are provided:
- `vertex/configs/train_gat_pyg_l4_image.yaml`
- `vertex/configs/train_gat_custom_l4_image.yaml`

They call the baked entrypoints and accept the same flags as before. Optional Experiment flags can be injected via env vars:

- `USE_VERTEX_EXP=1` → adds `--use-vertex-exp`
- `EXP_NAME=GAT-Phase5` → sets `--exp-name`
- `RUN_NAME=gat_pyg_d128_$(date +%s)` → sets `--run-name`

Example (gcloud):

```
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=gat-pyg-image \
  --config=vertex/configs/train_gat_pyg_l4_image.yaml \
  --args=""
```

To enable Vertex Experiments for the run:

```
export USE_VERTEX_EXP=1
export EXP_NAME=GAT-Phase5
export RUN_NAME=gat_pyg_d128_$(date +%s)
# then submit the same YAML
```

## Notes
- Existing YAMLs remain unchanged and keep working.
- Image contains `scripts/` and `plotpointe/` only; features/graphs are still read from GCS per flags.
- PyG wheels pinned to `torch==2.1.0+cu121` via official index (data.pyg.org).

