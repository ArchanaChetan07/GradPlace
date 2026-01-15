# Output Directory

This folder contains all outputs from placement optimization runs.

## Structure

Each run creates a timestamped subfolder with the following structure:

```
outputs/
└── run_YYYYMMDD_HHMMSS/    # Timestamped folder for each run
    ├── logs/                # Log files
    │   └── train.log       # Training log with all console output
    ├── images/              # All visualization images
    │   ├── snap_epoch_XXXX.png           # Placement snapshots (rectangles)
    │   ├── density_epoch_XXXX.png        # Density grid heatmaps
    │   ├── overflow_epoch_XXXX.png       # Overflow grid heatmaps
    │   ├── density_epoch_XXXX.pt         # Raw density tensor files
    │   ├── overflow_epoch_XXXX.pt        # Raw overflow tensor files
    │   ├── bin_densities_heatmap.png     # Bin densities visualization
    │   ├── gradient_quiver_epoch_XXX.png # Gradient/pressure vector fields
    │   ├── loss_curves.png                # Loss curves over time
    │   ├── overlap_cells_over_time.png    # Overlap cell count over time
    │   ├── overlap_highlight.png          # Final placement with overlaps highlighted
    │   ├── report.png                     # Combined debug report (1x3 layout)
    │   └── placement_result.png           # Before/after placement comparison
    └── metrics/             # Scalar metrics in JSON format
        ├── loss_history.json      # Loss values per epoch
        ├── overlap_metrics.json   # Final overlap statistics
        ├── runtime.json            # Timing information
        ├── test_results.json       # Test suite results (if from test.py)
        └── comparison.txt          # Baseline vs Improved comparison (debug mode)
```

## When Images Are Saved

### Debug Mode (`debug=True`)
- **Placement snapshots** (`snap_epoch_XXXX.png`): 
  - Saved at epoch 0, every `debug_interval` epochs (default 200), and final epoch
  - Shows all cells as rectangles with downsampling for N > 20,000
  
- **Density/Overflow heatmaps** (`density_epoch_XXXX.png`, `overflow_epoch_XXXX.png`):
  - Saved at epoch 0, every `debug_interval` epochs, and final epoch
  - Includes raw `.pt` tensor files for post-processing
  
- **Gradient quiver plots** (`gradient_quiver_epoch_XXX.png`):
  - Saved every `debug_interval` epochs
  - Shows overlap pressure vectors scaled by lambda_overlap
  
- **Loss curves** (`loss_curves.png`):
  - Saved once at the end of training
  - Shows total_loss, wirelength_loss, and overlap_loss over epochs
  
- **Overlap cells over time** (`overlap_cells_over_time.png`):
  - Saved once at the end of training
  - Tracks number of cells with overlaps at each log_interval
  
- **Overlap highlight** (`overlap_highlight.png`):
  - Saved once at the end if overlaps remain
  - Highlights overlapping cells with red borders
  
- **Debug report** (`report.png`):
  - Saved once at the end
  - Combined 1x3 layout: final placement, overflow heatmap, loss curves

### Normal Mode (`debug=False`)
- Only final metrics are saved (no images)
- Logs still contain training progress

## Running Tests

### Single Debug Run
```bash
python test.py --debug --device auto
```
- Runs one test case (10 macros, 2000 std cells)
- Generates all debug visualizations
- Saves to `outputs/run_YYYYMMDD_HHMMSS/`

### Debug Comparison Mode
When using `--debug`, the test suite runs two experiments:
1. **Baseline**: Current settings with all improvements
2. **Improved**: Same settings (for comparison)
- Each experiment gets its own output directory
- Comparison file saved to improved run's metrics folder

### Quick Test Suite
```bash
python test.py --quick --device auto
```
- Runs 3 test cases of varying sizes
- No debug images (faster)

### Full Test Suite
```bash
python test.py --device auto
```
- Runs 12 test cases from small to very large (up to 100k cells)
- No debug images (faster)

## Metrics Files

### loss_history.json
Contains arrays of loss values per epoch:
- `total_loss`: Combined wirelength + overlap loss
- `wirelength_loss`: Manhattan distance between connected cells
- `overlap_loss`: Grid-based density overflow penalty

Saved incrementally during training (atomic writes).

### overlap_metrics.json
Final overlap statistics:
- `overlap_count`: Number of overlapping cell pairs
- `overlap_ratio`: num_cells_with_overlaps / total_cells
- `total_overlap_area`: Sum of all overlap areas

### runtime.json
Timing information:
- `elapsed_time_seconds`: Total training time
- `per_epoch_timing`: Wirelength and overlap loss computation times
- `cumulative_timing`: Total time spent in each loss function

Updated incrementally during training.

### comparison.txt (Debug Mode Only)
Text file comparing baseline vs improved experiments:
- Side-by-side metrics comparison
- Includes overlap counts, normalized wirelength, and runtime
- Shows differences between experiments

## Notes

- All images use non-interactive matplotlib backend (Agg)
- Images are saved in background threads to avoid blocking training
- Large plots (N > 20,000) are automatically downsampled for visualization
- All file writes are atomic (write to .tmp then rename) to prevent corruption
- Logs capture both console output and file output simultaneously
