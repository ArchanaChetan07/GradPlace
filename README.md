# GradPlace: VLSI Cell Placement Optimization

A PyTorch-based optimizer for VLSI cell placement that minimizes wirelength while eliminating cell overlaps using differentiable optimization techniques.

## Overview

This project implements a gradient-based placement optimizer that:
- Minimizes total wirelength (Manhattan distance between connected cells)
- Eliminates cell overlaps using differentiable penalty functions
- Supports both macro cells and standard cells with different handling
- Uses multi-resolution grid-based density loss for scalability
- Includes comprehensive debugging and visualization tools

## Key Features

### Optimization Techniques
- **Two-phase lambda scheduling**: Smooth ramp from 0.1 → 1.0 → 15.0 for overlap penalty
- **Superlinear overflow penalty**: Cubic scaling (overflow³) for stronger repulsion in crowded bins
- **Macro-aware area weighting**: Macros get 5x weight, standard cells capped at 25% bin capacity
- **Overflow blur**: 3x3 average blur for smoother pressure distribution (auto-enabled for N≥5000)
- **Early stopping**: Automatically stops when overlaps reach zero (debug mode only)

### Loss Functions
- **Wirelength Loss**: Manhattan distance between connected cells via pins
- **Overlap Loss**: Multi-resolution grid-based density overflow with pairwise fallback
- **Gradient Clipping**: Adaptive clipping based on loss magnitude

### Visualization & Debugging
- Placement snapshots at key epochs
- Density and overflow heatmaps
- Loss curves and overlap tracking
- Gradient/pressure vector fields
- Combined debug reports

## Project Structure

```
.
├── placement.py          # Core optimization engine
├── test.py               # Test suite with multiple test cases
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── outputs/              # All run outputs (timestamped folders)
│   ├── run_YYYYMMDD_HHMMSS/
│   │   ├── logs/         # Training logs
│   │   ├── images/       # All visualizations
│   │   └── metrics/      # JSON metrics files
├── configs/              # Configuration files
├── data/                 # Data files
├── models/               # Saved models
└── notebooks/            # Jupyter notebooks
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify CUDA (optional, for GPU acceleration):
```bash
python check_cuda.py
```

## Usage

### Basic Usage

Run a single test case with debug outputs:
```bash
python test.py --debug --device auto
```

Run quick test suite (3 test cases):
```bash
python test.py --quick --device auto
```

Run full test suite (12 test cases):
```bash
python test.py --device auto
```

### Debug Mode Features

When using `--debug` flag:
- Runs two experiments back-to-back (baseline vs improved)
- Generates comprehensive visualizations
- Saves comparison file with metrics
- Enables early stopping when overlaps reach zero
- Creates detailed debug reports

### Device Selection

- `--device cpu`: Force CPU execution
- `--device cuda`: Force CUDA (if available)
- `--device auto`: Auto-select (CUDA if available, else CPU)

## Output Structure

Each run creates a timestamped folder in `outputs/` containing:

- **logs/train.log**: Complete training log with all console output
- **images/**: All visualization images (snapshots, heatmaps, curves, reports)
- **metrics/**: JSON files with loss history, overlap metrics, and runtime data

See `outputs/README.md` for detailed structure information.

## Key Algorithms

### Grid-Based Density Loss
- Soft assignment of cell areas to grid bins using Gaussian weighting
- Computes overflow (density above bin capacity)
- Applies superlinear penalty (cubic) for stronger repulsion
- Supports multi-resolution (fine + coarse grids) for large designs

### Macro Handling
- Detects macros by height > 1.0
- Applies 5x area weight to macros
- Caps standard cell contributions to prevent flooding

### Lambda Scheduling
- Phase 1 (0-40% epochs): Slow ramp 0.1 → 1.0
- Phase 2 (40-100% epochs): Aggressive ramp 1.0 → 15.0
- Allows wirelength optimization early, then focuses on overlap resolution

## Metrics

The optimizer reports:
- **Overlap Ratio**: Number of cells with overlaps / total cells
- **Normalized Wirelength**: (Total wirelength / num_nets) / sqrt(total_area)
- **Runtime**: Total training time in seconds

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib (for visualization)

## References

This implementation includes techniques from:
- Differentiable placement optimization
- Multi-resolution density-based placement
- Macro-aware placement algorithms
- Gradient-based optimization with adaptive scheduling

## License

[Add license information here]
