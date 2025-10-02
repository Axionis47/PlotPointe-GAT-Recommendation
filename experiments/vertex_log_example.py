#!/usr/bin/env python3
import argparse
import time
from datetime import datetime

from google.cloud import aiplatform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="plotpointe")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--experiment", default="recsys-dev")
    ap.add_argument("--run", default=None)
    ap.add_argument("--artifact_uri", default="gs://plotpointe-artifacts/experiments/recsys-dev/placeholder.txt")
    args = ap.parse_args()

    aiplatform.init(project=args.project, location=args.location, experiment=args.experiment)

    run_name = args.run or f"dryrun-{int(time.time())}"
    with aiplatform.start_run(run=run_name) as run:
        # Log a few example params and metrics
        run.log_params({
            "model": "gat-baseline",
            "embedding_dim": 64,
            "optimizer": "adam",
            "lr": 0.001,
            "batch_size": 512,
            "artifact_uri": args.artifact_uri,
        })
        run.log_metrics({
            "val/ndcg@10": 0.123,
            "val/hit@10": 0.456,
            "train/epoch": 1,
        })

    print(f"Created experiment run: {args.experiment}/{run_name}")
    print("Open Vertex AI Console → Experiments to view and compare runs.")


if __name__ == "__main__":
    main()

