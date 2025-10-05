# Phase 5: Observability and Promotion Gates

This phase adds:
- Structured JSON logs (opt-in) for key training events
- A small, local promotion gate that compares a run to a baseline metrics JSON

No default behavior changes. Everything is opt-in.

## Structured logs (opt-in)

Enable per run with `--structured-logs` on either trainer. Logs are emitted as JSON to stdout:

- `run_start`: includes config, feature selection, and environment
- `epoch_end`: epoch number, loss, and validation metrics
- `run_complete`: best val metric, test metrics, and uploaded artifact URIs

Example log line (one per event):

```
{"ts":"2025-01-01T00:00:00Z","event":"epoch_end","run_id":"gat_pyg_d128_1700000000","epoch":5,"loss":0.1234,"val":{"ndcg@20":0.0199,"recall@20":0.0520}}
```

## Promotion gate (read-only, local)

Use `tools/promotion_gate.py` to compare the current run's metrics JSON to a baseline metrics JSON file.

Example:

```
python tools/promotion_gate.py \
  --current /path/to/metrics_current.json \
  --baseline /path/to/metrics_baseline.json \
  --metric ndcg@20 --split test --mode improve_or_equal --tol 0.0 \
  --out /tmp/gate_result.json
```

- exit 0 if PASS; 3 if FAIL
- `--mode`:
  - `improve_or_equal`: require current >= baseline - tol
  - `no_regression`: require current + tol >= baseline

You can wire this into CI as a guard to prevent regressions from being promoted.

## Notes
- We intentionally avoided any coupling to Vertex Experiments in the gate so it remains simple and local. You can extend it later to fetch baselines from Vertex or GCS.
- Structured logs are best-effort and never fail the run if logging encounters issues.

