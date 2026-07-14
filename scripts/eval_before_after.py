"""Run a labeled synthetic VLSI toy netlist and report BEFORE vs AFTER metrics.

Benchmark (NOT an industry ISPD contest case):
  QUICK_TEST_CASES #1 — 2 macros + 30 standard cells, seed=1003
  Random radial initialization → GradPlace Adam train_placement

Usage:
  python scripts/eval_before_after.py
  python scripts/eval_before_after.py --epochs 1000 --outdir assets
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from placement import (
    calculate_normalized_metrics,
    calculate_overlap_metrics,
    generate_placement_input,
    plot_placement,
    train_placement,
    wirelength_attraction_loss,
)


# Labeled synthetic toy case (matches test.py QUICK_TEST_CASES entry)
TOY_BENCHMARK = {
    "name": "synthetic_toy_2macro_30std",
    "label": "synthetic toy (not an ISPD contest benchmark)",
    "num_macros": 2,
    "num_std_cells": 30,
    "seed": 1003,
}


def radial_init(cell_features: torch.Tensor) -> torch.Tensor:
    """Naive baseline: random radial scatter (same init used by the test harness)."""
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells, device=cell_features.device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=cell_features.device) * spread_radius
    out = cell_features.clone()
    out[:, 2] = radii * torch.cos(angles)
    out[:, 3] = radii * torch.sin(angles)
    return out


def snapshot_metrics(cell_features, pin_features, edge_list) -> dict:
    ov = calculate_overlap_metrics(cell_features)
    norm = calculate_normalized_metrics(cell_features, pin_features, edge_list)
    wl_mean = wirelength_attraction_loss(cell_features, pin_features, edge_list).item()
    total_wl = wl_mean * edge_list.shape[0]
    return {
        "total_wirelength_smooth_manhattan": float(total_wl),
        "normalized_wl": float(norm["normalized_wl"]),
        "overlap_count": int(ov["overlap_count"]),
        "total_overlap_area": float(ov["total_overlap_area"]),
        "overlap_percentage_pairs_per_cell": float(ov["overlap_percentage"]),
        "num_cells_with_overlaps": int(norm["num_cells_with_overlaps"]),
        "overlap_ratio": float(norm["overlap_ratio"]),
        "total_cells": int(norm["total_cells"]),
        "num_nets": int(norm["num_nets"]),
    }


def render_panels(cell_features, title: str, filepath: Path, highlight_overlaps: bool = True) -> None:
    """Save a single placement panel with optional red overlap highlighting."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from placement import calculate_cells_with_overlaps

    cf = cell_features.detach().cpu()
    positions = cf[:, 2:4].numpy()
    widths = cf[:, 4].numpy()
    heights = cf[:, 5].numpy()
    overlaps = calculate_cells_with_overlaps(cf) if highlight_overlaps else set()
    metrics = calculate_overlap_metrics(cf)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for i in range(cf.shape[0]):
        is_macro = heights[i] > 1.0
        if i in overlaps:
            face, edge = "#f87171", "#991b1b"
        elif is_macro:
            face, edge = "#a78bfa", "#5b21b6"
        else:
            face, edge = "#93c5fd", "#1e3a8a"
        ax.add_patch(
            Rectangle(
                (positions[i, 0] - widths[i] / 2, positions[i, 1] - heights[i] / 2),
                widths[i],
                heights[i],
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8 if is_macro else 0.4,
                alpha=0.75,
            )
        )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"{title}\n"
        f"overlap pairs={metrics['overlap_count']} · "
        f"overlap area={metrics['total_overlap_area']:.2f} · "
        f"cells overlapping={len(overlaps)}/{cf.shape[0]}",
        fontsize=11,
    )
    margin = 8
    ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eval(epochs: int, outdir: Path, device: str = "cpu") -> dict:
    torch.manual_seed(TOY_BENCHMARK["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cell_features, pin_features, edge_list = generate_placement_input(
        TOY_BENCHMARK["num_macros"], TOY_BENCHMARK["num_std_cells"]
    )
    cell_features = radial_init(cell_features)
    cell_features = cell_features.to(device)
    pin_features = pin_features.to(device)
    edge_list = edge_list.to(device)

    before = snapshot_metrics(cell_features.cpu(), pin_features.cpu(), edge_list.cpu())
    render_panels(
        cell_features.cpu(),
        "BEFORE — naive radial init (synthetic toy)",
        outdir / "placement_before.png",
    )

    t0 = time.perf_counter()
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        num_epochs=epochs,
        verbose=False,
        device=device,
        debug=False,
    )
    runtime_s = time.perf_counter() - t0

    final = result["final_cell_features"].cpu()
    after = snapshot_metrics(final, pin_features.cpu(), edge_list.cpu())
    render_panels(
        final,
        "AFTER — GradPlace optimized (synthetic toy)",
        outdir / "placement_after.png",
    )
    # Side-by-side using existing helper
    plot_placement(
        result["initial_cell_features"].cpu(),
        final,
        pin_features.cpu(),
        edge_list.cpu(),
        filename="placement_before_after.png",
        output_dir={"images_dir": str(outdir)},
    )

    wl_before = before["total_wirelength_smooth_manhattan"]
    wl_after = after["total_wirelength_smooth_manhattan"]
    ov_before = before["total_overlap_area"]
    ov_after = after["total_overlap_area"]
    cells_before = before["num_cells_with_overlaps"]
    cells_after = after["num_cells_with_overlaps"]

    report = {
        "benchmark": TOY_BENCHMARK,
        "hardware": {
            "device": device,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "hyperparams": {
            "num_epochs": epochs,
            "macro_weight": 5.0,
            "std_cell_bin_cap_fraction": 0.25,
            "lambda_overlap_ramp": "0.1 → 1.0 → 15.0 (two-phase)",
        },
        "before": before,
        "after": after,
        "delta": {
            "wirelength_reduction_pct": round(
                100.0 * (wl_before - wl_after) / wl_before, 2
            )
            if wl_before
            else 0.0,
            "overlap_area_reduction_pct": round(
                100.0 * (ov_before - ov_after) / ov_before, 2
            )
            if ov_before
            else 0.0,
            "cells_with_overlaps_before": cells_before,
            "cells_with_overlaps_after": cells_after,
        },
        "runtime_seconds": round(runtime_s, 3),
        "artifacts": {
            "before_png": (outdir / "placement_before.png").as_posix(),
            "after_png": (outdir / "placement_after.png").as_posix(),
            "side_by_side_png": (outdir / "placement_before_after.png").as_posix(),
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "benchmark_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("benchmark", "delta", "runtime_seconds", "before", "after")}, indent=2))
    print(f"Wrote {metrics_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=Path("assets"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    run_eval(epochs=args.epochs, outdir=args.outdir, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
