# Training image for GAT Recommendation (PyTorch + PyG + GCP SDK)
# Built to remove cold-start pip installs on Vertex AI jobs

FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-1.py310

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Core deps + GCP SDK clients + SciPy stack (aligned with requirements.txt)
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      torch==2.1.0 torchvision==0.16.0 \
      numpy>=1.24.0,<2.0.0 pandas>=2.0.0,<3.0.0 pyarrow>=12.0.0,<15.0.0 \
      scipy>=1.10.0,<2.0.0 scikit-learn>=1.3.0,<2.0.0 \
      google-cloud-aiplatform>=1.35.0,<2.0.0 google-cloud-storage>=2.10.0,<3.0.0 \
      tqdm>=4.65.0,<5.0.0 requests>=2.31.0,<3.0.0 pillow>=10.0.0,<11.0.0 networkx>=3.1,<4.0.0 && \
    pip install --no-cache-dir \
      pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
      -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy minimal repo code needed to run training entrypoints
WORKDIR /opt/app
COPY scripts/ /opt/app/scripts/
COPY plotpointe/ /opt/app/plotpointe/

ENV PYTHONPATH=/opt/app

# Default to bash shell; Vertex will override command
ENTRYPOINT ["/bin/bash", "-lc"]

