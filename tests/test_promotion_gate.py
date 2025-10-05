#!/usr/bin/env python3
import json
from pathlib import Path

from tools.promotion_gate import evaluate


def _write_metrics(p: Path, ndcg: float, recall: float):
    p.write_text(json.dumps({
        "val": {"ndcg@20": ndcg, "recall@20": recall},
        "test": {"ndcg@20": ndcg, "recall@20": recall}
    }))


def test_gate_improve_or_equal_pass(tmp_path: Path):
    cur = tmp_path / "cur.json"
    base = tmp_path / "base.json"
    _write_metrics(cur, 0.0200, 0.0500)
    _write_metrics(base, 0.0199, 0.0490)
    ok, res = evaluate(cur, base, split="test", metric="ndcg@20", mode="improve_or_equal", tol=0.0)
    assert ok and res["pass"]


def test_gate_no_regression_fail(tmp_path: Path):
    cur = tmp_path / "cur.json"
    base = tmp_path / "base.json"
    _write_metrics(cur, 0.0190, 0.0480)
    _write_metrics(base, 0.0199, 0.0520)
    ok, res = evaluate(cur, base, split="test", metric="ndcg@20", mode="no_regression", tol=0.0001)
    assert (not ok) and (not res["pass"])  # too much regression

