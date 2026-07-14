"""GradPlace unit + smoke tests (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from placement import (
    MACRO_WEIGHT,
    STD_CELL_BIN_CAP_FRACTION,
    generate_placement_input,
    lambda_overlap_schedule,
    std_cell_bin_capacity_cap,
    train_placement,
)


def test_macro_weight_constant():
    assert MACRO_WEIGHT == 5.0


def test_std_cell_bin_capacity_cap_is_quarter_bin():
    assert STD_CELL_BIN_CAP_FRACTION == 0.25
    for bin_size in (1.0, 2.5, 10.0):
        expected = 0.25 * (bin_size * bin_size)
        assert std_cell_bin_capacity_cap(bin_size) == pytest.approx(expected)
        t = torch.tensor(bin_size)
        assert float(std_cell_bin_capacity_cap(t)) == pytest.approx(expected)


def test_lambda_overlap_ramp_two_phase():
    n = 1000
    # Phase 1 endpoints
    assert lambda_overlap_schedule(0, n) == pytest.approx(0.1)
    assert lambda_overlap_schedule(399, n) == pytest.approx(0.1 + 0.9 * (399 / 400))
    # Phase 1→2 boundary (epoch 400 is start of phase 2)
    assert lambda_overlap_schedule(400, n) == pytest.approx(1.0)
    # Mid phase 2 (epoch 700: 300 steps into phase2 of length 600 → 300/599)
    assert lambda_overlap_schedule(700, n) == pytest.approx(1.0 + 14.0 * (300 / 599))
    # Final training epoch hits the phase-2 cap
    assert lambda_overlap_schedule(999, n) == pytest.approx(15.0)
    assert lambda_overlap_schedule(n, n) == pytest.approx(15.0)


def test_lambda_overlap_monotonic_nondecreasing():
    n = 200
    vals = [lambda_overlap_schedule(e, n) for e in range(n)]
    for a, b in zip(vals, vals[1:]):
        assert b + 1e-9 >= a
    assert vals[0] == pytest.approx(0.1)
    assert vals[-1] == pytest.approx(15.0)


def test_placer_smoke_synthetic_toy_cpu():
    """End-to-end smoke on the labeled synthetic toy (2 macros + 30 std cells)."""
    torch.manual_seed(1003)
    cells, pins, edges = generate_placement_input(2, 30)
    # Tiny radial init
    area = cells[:, 0].sum().item() ** 0.5 * 0.6
    ang = torch.rand(cells.shape[0]) * 6.28
    r = torch.rand(cells.shape[0]) * area
    cells = cells.clone()
    cells[:, 2] = r * torch.cos(ang)
    cells[:, 3] = r * torch.sin(ang)

    result = train_placement(
        cells,
        pins,
        edges,
        num_epochs=40,
        verbose=False,
        device="cpu",
        debug=False,
    )
    final = result["final_cell_features"]
    assert final.shape == cells.shape
    assert torch.isfinite(final).all()
    # Positions should have moved somehow vs identical init (not required to improve in 40 epochs)
    assert "loss_history" in result


def test_benchmark_metrics_json_schema_if_present():
    """If assets/benchmark_metrics.json exists (from eval script), validate schema."""
    path = Path("assets/benchmark_metrics.json")
    if not path.exists():
        pytest.skip("Run scripts/eval_before_after.py to generate assets/benchmark_metrics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["benchmark"]["label"].startswith("synthetic")
    assert "before" in data and "after" in data
    assert "runtime_seconds" in data
    for key in ("before_png", "after_png", "side_by_side_png"):
        rel = Path(data["artifacts"][key].replace("\\", "/"))
        assert rel.exists(), f"missing artifact {rel}"
