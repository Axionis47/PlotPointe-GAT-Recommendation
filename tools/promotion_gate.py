#!/usr/bin/env python3
"""
Promotion gate evaluator: compare current metrics JSON against a baseline metrics JSON.
Purely local, read-only. Use exit codes for CI wiring.

Example:
  python tools/promotion_gate.py \
    --current tmp/metrics_gat_pyg.json \
    --baseline gs://.../metrics_prev.json (download first) \
    --metric ndcg@20 --split test --mode improve_or_equal --tol 0.0 \
    --out tmp/gate_result.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def _extract_metric(metrics: Dict, split: str, metric: str) -> float:
    # metrics schema: {"val": {...}, "test": {...}}
    if split not in metrics:
        raise KeyError(f"Split not in metrics: {split}")
    if metric not in metrics[split]:
        raise KeyError(f"Metric not in metrics[{split}]: {metric}")
    return float(metrics[split][metric])


def evaluate(current_path: Path, baseline_path: Path, split: str, metric: str, mode: str, tol: float) -> Tuple[bool, Dict]:
    with open(current_path, "r") as f:
        cur = json.load(f)
    with open(baseline_path, "r") as f:
        base = json.load(f)

    cur_v = _extract_metric(cur, split, metric)
    base_v = _extract_metric(base, split, metric)

    ok: bool
    reason: str
    if mode == "improve_or_equal":
        ok = cur_v + 1e-12 >= base_v - tol
        reason = f"current {cur_v:.6f} >= baseline {base_v:.6f} - tol {tol}"
    elif mode == "no_regression":
        ok = cur_v + tol + 1e-12 >= base_v
        reason = f"current {cur_v:.6f} + tol {tol} >= baseline {base_v:.6f}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = {
        "metric": metric,
        "split": split,
        "mode": mode,
        "tolerance": tol,
        "current": cur_v,
        "baseline": base_v,
        "pass": ok,
        "reason": reason,
    }
    return ok, result


def main():
    ap = argparse.ArgumentParser(description="Promotion gate evaluator")
    ap.add_argument("--current", required=True, help="Path to current run metrics JSON")
    ap.add_argument("--baseline", required=True, help="Path to baseline metrics JSON")
    ap.add_argument("--split", default="test", help="Split to compare (val|test)")
    ap.add_argument("--metric", default="ndcg@20", help="Metric key to compare (e.g., ndcg@20)")
    ap.add_argument("--mode", choices=["improve_or_equal", "no_regression"], default="improve_or_equal")
    ap.add_argument("--tol", type=float, default=0.0, help="Tolerance for comparison")
    ap.add_argument("--out", default=None, help="Optional path to write gate result JSON")
    args = ap.parse_args()

    ok, res = evaluate(Path(args.current), Path(args.baseline), args.split, args.metric, args.mode, args.tol)
    print(json.dumps(res, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)

    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()

