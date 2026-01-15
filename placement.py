"""
VLSI Cell Placement Optimization
==========================================

OVERVIEW:
You are tasked with implementing a critical component of a chip placement optimizer.
Given a set of cells (circuit components) with fixed sizes and connectivity requirements,
you need to find positions for these cells that:
1. Minimize total wirelength (wiring cost between connected pins)
2. Eliminate all overlaps between cells

TASK:
Implement the `overlap_repulsion_loss()` function to prevent cells from overlapping.
The function must:
- Be differentiable (uses PyTorch operations for gradient descent)
- Detect when cells overlap in 2D space
- Apply increasing penalties for larger overlaps
- Work efficiently with vectorized operations

SUCCESS CRITERIA:
After running the optimizer with your implementation:
- overlap_count should be 0 (no overlapping cell pairs)
- total_overlap_area should be 0.0 (no overlap)
- wirelength should be minimized
- Visualization should show clean, non-overlapping placement

"""

import os
import time
from enum import IntEnum

# Force matplotlib to use non-interactive backend (global, safe)
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"

import torch
import torch.optim as optim


# Feature index enums for cleaner code access
class CellFeatureIdx(IntEnum):
    """Indices for cell feature tensor columns."""
    AREA = 0
    NUM_PINS = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5


class PinFeatureIdx(IntEnum):
    """Indices for pin feature tensor columns."""
    CELL_IDX = 0
    PIN_X = 1  # Relative to cell corner
    PIN_Y = 2  # Relative to cell corner
    X = 3  # Absolute position
    Y = 4  # Absolute position
    WIDTH = 5
    HEIGHT = 6


# Configuration constants
# Macro parameters
MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0

# Standard cell parameters (areas can be 1, 2, or 3)
STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0

# Pin count parameters
MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_output_directory(base_output_dir=None):
    """Create a unified output directory structure with timestamped run folder.
    
    Creates: outputs/run_YYYYMMDD_HHMMSS/{logs,images,metrics}/
    
    Args:
        base_output_dir: Base directory for outputs. If None, uses project root.
    
    Returns:
        Dictionary with paths: {
            'run_dir': path to timestamped run directory,
            'logs_dir': path to logs/ subdirectory,
            'images_dir': path to images/ subdirectory,
            'metrics_dir': path to metrics/ subdirectory
        }
    """
    if base_output_dir is None:
        base_output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create outputs/ directory at project root
    outputs_base = os.path.join(base_output_dir, "outputs")
    os.makedirs(outputs_base, exist_ok=True)
    
    # Create timestamped run directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(outputs_base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    logs_dir = os.path.join(run_dir, "logs")
    images_dir = os.path.join(run_dir, "images")
    metrics_dir = os.path.join(run_dir, "metrics")
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    return {
        'run_dir': run_dir,
        'logs_dir': logs_dir,
        'images_dir': images_dir,
        'metrics_dir': metrics_dir
    }


# Global log file handle for lightweight logging
_log_file_handle = None


def setup_logging(logs_dir):
    """Set up lightweight logging to both console and log file.
    
    Args:
        logs_dir: Directory where train.log will be created
    
    Returns:
        log function that writes to both stdout and file
    """
    global _log_file_handle
    
    log_file_path = os.path.join(logs_dir, "train.log")
    
    try:
        # Open log file in append mode
        _log_file_handle = open(log_file_path, 'a', encoding='utf-8')
        
        def log(msg, end='\n', flush=False):
            """Lightweight logging: print to console and append to log file."""
            # Print to console
            print(msg, end=end, flush=flush)
            # Write to log file
            if _log_file_handle is not None:
                try:
                    _log_file_handle.write(str(msg) + end)
                    if flush:
                        _log_file_handle.flush()
                except (IOError, OSError):
                    pass  # Silently fail if file write fails
        
        return log
    except (IOError, OSError) as e:
        # If we can't open the log file, just return a function that only prints
        def log(msg, end='\n', flush=False):
            print(msg, end=end, flush=flush)
        return log


def close_logging():
    """Close the log file handle."""
    global _log_file_handle
    if _log_file_handle is not None:
        try:
            _log_file_handle.close()
        except (IOError, OSError):
            pass
        _log_file_handle = None


def _log_plot_failure(filename, exception):
    """Safely log a plot failure to the run log file.
    
    Args:
        filename: Name of the file that failed to save
        exception: The exception that occurred
    """
    global _log_file_handle
    try:
        if _log_file_handle is not None:
            error_msg = f"plot_failed: {filename} {type(exception).__name__}: {str(exception)}\n"
            _log_file_handle.write(error_msg)
            _log_file_handle.flush()
    except:
        pass  # Silently fail - never crash training


def _atomic_write_json(data, filepath):
    """Atomically write JSON data to disk (write to .tmp then rename).
    
    This ensures metrics exist even if training stops early.
    
    Args:
        data: Dictionary to write as JSON
        filepath: Target file path
    """
    try:
        import json
        tmp_filepath = filepath + ".tmp"
        # Write to temporary file first
        with open(tmp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        # Atomic rename (works on both Unix and Windows)
        if os.path.exists(filepath):
            os.replace(tmp_filepath, filepath)
        else:
            os.rename(tmp_filepath, filepath)
    except Exception:
        pass  # Silently fail - never crash training


# ======= SETUP =======

def generate_placement_input(num_macros, num_std_cells):
    """Generate synthetic placement input data.

    Args:
        num_macros: Number of macros to generate
        num_std_cells: Number of standard cells to generate

    Returns:
        Tuple of (cell_features, pin_features, edge_list):
            - cell_features: torch.Tensor of shape [N, 6] with columns [area, num_pins, x, y, width, height]
            - pin_features: torch.Tensor of shape [total_pins, 7] with columns
              [cell_instance_index, pin_x, pin_y, x, y, pin_width, pin_height]
            - edge_list: torch.Tensor of shape [E, 2] with [src_pin_idx, tgt_pin_idx]
    """
    total_cells = num_macros + num_std_cells

    # Step 1: Generate macro areas (uniformly distributed between min and max)
    macro_areas = (
        torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA
    )

    # Step 2: Generate standard cell areas (randomly pick from 1, 2, or 3)
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[
        torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))
    ]

    # Combine all areas
    areas = torch.cat([macro_areas, std_cell_areas])

    # Step 3: Calculate cell dimensions
    # Macros are square
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)

    # Standard cells have fixed height = 1, width = area
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)

    # Combine dimensions
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    # Step 4: Calculate number of pins per cell
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)

    # Macros: between sqrt(area) and 2*sqrt(area) pins
    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()

    # Standard cells: between 3 and 6 pins
    num_pins_per_cell[num_macros:] = torch.randint(
        MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,)
    )

    # Step 5: Create cell features tensor [area, num_pins, x, y, width, height]
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.X] = 0.0  # x position (initialized to 0)
    cell_features[:, CellFeatureIdx.Y] = 0.0  # y position (initialized to 0)
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Step 6: Generate pins for each cell
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()

        # Generate random pin positions within the cell
        # Offset from edges to ensure pins are fully inside
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            # For very small cells, just center the pins
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)

        # Fill pin features
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_X] = (
            pin_x  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_Y] = (
            pin_y  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.X] = (
            pin_x  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.Y] = (
            pin_y  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE

        pin_idx += n_pins

    # Step 7: Generate edges with simple random connectivity
    # Each pin connects to 1-3 random pins (preferring different cells)
    edge_list = []
    avg_edges_per_pin = 2.0

    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx : pin_idx + n_pins] = cell_idx
        pin_idx += n_pins

    # Create adjacency set to avoid duplicate edges
    adjacency = [set() for _ in range(total_pins)]

    for pin_idx in range(total_pins):
        pin_cell = pin_to_cell[pin_idx].item()
        num_connections = torch.randint(1, 4, (1,)).item()  # 1-3 connections per pin

        # Try to connect to pins from different cells
        for _ in range(num_connections):
            # Random candidate
            other_pin = torch.randint(0, total_pins, (1,)).item()

            # Skip self-connections and existing connections
            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue

            # Add edge (always store smaller index first for consistency)
            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])

            # Update adjacency
            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)

    # Convert to tensor and remove duplicates
    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss computes the Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Update absolute pin positions based on cell positions
    cell_positions = cell_features[:, 2:4]  # [N, 2]
    cell_indices = pin_features[:, 0].long()

    # Calculate absolute pin positions
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, 1]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, 2]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Calculate smooth approximation of Manhattan distance
    # Using log-sum-exp approximation for differentiability
    alpha = 0.1  # Smoothing parameter
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)

    # Smooth L1 distance with numerical stability
    smooth_manhattan = alpha * torch.logsumexp(
        torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
    )

    # Total wirelength
    total_wirelength = torch.sum(smooth_manhattan)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


def _compute_grid_density_loss(cell_features, bin_size, blur=False, debug=False, return_debug=False, debug_output_dir=None, epoch=None):
    """Compute grid-based density overflow loss.
    
    Args:
        cell_features: [N, 6] tensor with cell properties
        bin_size: Size of each grid bin
        blur: If True, apply separable blur (1x3 then 3x1) to bin densities. Default: False
        debug: If True, return tuple of (loss, overflow_grid, density_grid). Default: False (deprecated, use return_debug)
        return_debug: If True, return tuple of (loss, overflow_grid, density_grid). Default: False
        debug_output_dir: Directory to save heatmaps and .pt files. Default: None
        epoch: Epoch number for filename. Default: None
        
    Returns:
        If return_debug=False and debug=False: Scalar loss value for density overflow
        If return_debug=True or debug=True: Tuple of (loss, overflow_grid, density_grid) where both grids are detached [num_bins_x, num_bins_y]
    """
    # Determine if we should return debug info
    should_return_debug = return_debug or debug
    
    N = cell_features.shape[0]
    if N <= 1:
        if should_return_debug:
            empty_grid = torch.zeros((0, 0), dtype=cell_features.dtype, device=cell_features.device)
            return (torch.tensor(0.0, dtype=cell_features.dtype, device=cell_features.device, requires_grad=True),
                    empty_grid, empty_grid)
        return torch.tensor(0.0, dtype=cell_features.dtype, device=cell_features.device, requires_grad=True)
    
    # Ensure bin_size is a tensor with proper device/dtype
    if not isinstance(bin_size, torch.Tensor):
        bin_size = torch.tensor(bin_size, dtype=cell_features.dtype, device=cell_features.device)
    
    positions = cell_features[:, 2:4]  # [N, 2]
    areas = cell_features[:, 0]  # [N] - cell areas
    heights = cell_features[:, 5]  # [N] - cell heights
    
    # Detect macros: macros have height > 1.0 in this challenge
    # Create boolean mask to identify macro cells
    is_macro = heights > 1.0  # [N] boolean tensor
    
    # Modify per-cell area contribution before splatting to bins
    # Macros get higher weight (macro_weight) to account for their larger impact on density
    # Standard cells get weight 1.0 (no change)
    macro_weight = 5.0
    effective_area = areas * torch.where(is_macro, macro_weight, 1.0)  # [N]
    
    # Cap standard-cell contribution to prevent flooding from many small cells
    # This prevents a large number of standard cells from overwhelming the density grid
    # Cap is set to 25% of bin capacity to allow reasonable density but prevent excessive accumulation
    bin_capacity = bin_size * bin_size
    cap = 0.25 * bin_capacity  # Cap for standard cells
    effective_area = torch.where(
        is_macro,
        effective_area,  # Macros: use weighted area (no cap)
        torch.clamp(effective_area, max=cap)  # Standard cells: clamp to cap
    )  # [N]
    
    # Compute grid bounds
    x_min = positions[:, 0].min() - cell_features[:, 4].max() / 2
    x_max = positions[:, 0].max() + cell_features[:, 4].max() / 2
    y_min = positions[:, 1].min() - cell_features[:, 5].max() / 2
    y_max = positions[:, 1].max() + cell_features[:, 5].max() / 2
    
    # Compute number of bins
    num_bins_x = int(torch.ceil((x_max - x_min) / bin_size).item()) + 1
    num_bins_y = int(torch.ceil((y_max - y_min) / bin_size).item()) + 1
    
    if num_bins_x <= 0 or num_bins_y <= 0:
        if should_return_debug:
            empty_grid = torch.zeros((0, 0), dtype=cell_features.dtype, device=cell_features.device)
            return (torch.tensor(0.0, dtype=cell_features.dtype, device=cell_features.device, requires_grad=True),
                    empty_grid, empty_grid)
        return torch.tensor(0.0, dtype=cell_features.dtype, device=cell_features.device, requires_grad=True)
    
    # Distribute cell area to bins using soft assignment (differentiable)
    # Use a smooth distribution based on distance from bin center
    bin_centers_x = torch.arange(num_bins_x, dtype=cell_features.dtype, device=cell_features.device) * bin_size + x_min + bin_size / 2
    bin_centers_y = torch.arange(num_bins_y, dtype=cell_features.dtype, device=cell_features.device) * bin_size + y_min + bin_size / 2
    
    # Compute distances from cell centers to bin centers
    cell_x = positions[:, 0].unsqueeze(1)  # [N, 1]
    cell_y = positions[:, 1].unsqueeze(1)  # [N, 1]
    bin_x_centers = bin_centers_x.unsqueeze(0)  # [1, num_bins_x]
    bin_y_centers = bin_centers_y.unsqueeze(0)  # [1, num_bins_y]
    
    # Soft assignment using Gaussian-like weighting
    sigma = bin_size / 2  # Smoothing parameter
    dx = (cell_x - bin_x_centers) / sigma  # [N, num_bins_x]
    dy = (cell_y - bin_y_centers) / sigma  # [N, num_bins_y]
    
    # Gaussian weights (normalized)
    weight_x = torch.exp(-0.5 * dx ** 2)  # [N, num_bins_x]
    weight_y = torch.exp(-0.5 * dy ** 2)  # [N, num_bins_y]
    
    # Normalize weights
    try:
        eps_value = torch.finfo(cell_features.dtype).eps
    except (TypeError, ValueError):
        eps_value = 1e-8
    eps = torch.tensor(eps_value, dtype=cell_features.dtype, device=cell_features.device, requires_grad=False)
    weight_x = weight_x / (weight_x.sum(dim=1, keepdim=True) + eps)
    weight_y = weight_y / (weight_y.sum(dim=1, keepdim=True) + eps)
    
    # Distribute effective area to bins (using modified area contributions)
    area_per_bin = (effective_area.unsqueeze(1) * weight_x).unsqueeze(2) * weight_y.unsqueeze(1)  # [N, num_bins_x, num_bins_y]
    bin_densities = area_per_bin.sum(dim=0)  # [num_bins_x, num_bins_y]
    
    # Debug prints: shapes, min/max, NaN checks
    if debug:
        print(f"  [DEBUG] bin_densities shape: {bin_densities.shape}")
        print(f"  [DEBUG] bin_densities min: {bin_densities.min().item():.6f}, max: {bin_densities.max().item():.6f}")
        print(f"  [DEBUG] bin_densities mean: {bin_densities.mean().item():.6f}, std: {bin_densities.std().item():.6f}")
        nan_count = torch.isnan(bin_densities).sum().item()
        inf_count = torch.isinf(bin_densities).sum().item()
        if nan_count > 0:
            print(f"  [DEBUG] WARNING: bin_densities has {nan_count} NaN values")
        if inf_count > 0:
            print(f"  [DEBUG] WARNING: bin_densities has {inf_count} Inf values")
    
    # Apply separable blur if enabled (equivalent to 3x3 average blur but more efficient)
    if blur:
        # Add batch and channel dimensions for conv2d: [1, 1, H, W]
        bin_densities_4d = bin_densities.unsqueeze(0).unsqueeze(0)  # [1, 1, num_bins_x, num_bins_y]
        
        # Create separable blur kernels (equivalent to 3x3 average blur)
        # 1x3 horizontal blur kernel: [1/3, 1/3, 1/3]
        kernel_1x3 = torch.tensor(
            [[1.0/3.0, 1.0/3.0, 1.0/3.0]],
            dtype=cell_features.dtype,
            device=cell_features.device
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 3]
        
        # 3x1 vertical blur kernel: [1/3; 1/3; 1/3]
        kernel_3x1 = torch.tensor(
            [[1.0/3.0], [1.0/3.0], [1.0/3.0]],
            dtype=cell_features.dtype,
            device=cell_features.device
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 1]
        
        # Apply horizontal blur (1x3) with padding to maintain size
        blurred_h = torch.nn.functional.conv2d(
            bin_densities_4d,
            kernel_1x3,
            padding=(0, 1),  # pad left/right
            groups=1
        )
        
        # Apply vertical blur (3x1) with padding to maintain size
        blurred = torch.nn.functional.conv2d(
            blurred_h,
            kernel_3x1,
            padding=(1, 0),  # pad top/bottom
            groups=1
        )
        
        # Remove batch and channel dimensions: [num_bins_x, num_bins_y]
        bin_densities = blurred.squeeze(0).squeeze(0)
        
        # Debug prints after blur
        if debug:
            print(f"  [DEBUG] bin_densities (after blur) shape: {bin_densities.shape}")
            print(f"  [DEBUG] bin_densities (after blur) min: {bin_densities.min().item():.6f}, max: {bin_densities.max().item():.6f}")
            nan_count = torch.isnan(bin_densities).sum().item()
            if nan_count > 0:
                print(f"  [DEBUG] WARNING: bin_densities (after blur) has {nan_count} NaN values")
    
    # Save bin_densities heatmap if debug is enabled
    if debug and should_return_debug and debug_output_dir is not None:
        heatmap_filename = None
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import os
            
            # Use provided debug_output_dir or fallback to default
            output_dir = debug_output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Detach and move to CPU for visualization
            bin_densities_vis = bin_densities.detach().cpu().numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(bin_densities_vis, cmap='viridis', interpolation='nearest', origin='lower')
            ax.set_title(f'Bin Densities Heatmap\nShape: {bin_densities.shape}, Min: {bin_densities_vis.min():.4f}, Max: {bin_densities_vis.max():.4f}')
            ax.set_xlabel('Bin X')
            ax.set_ylabel('Bin Y')
            plt.colorbar(im, ax=ax, label='Density')
            plt.tight_layout()
            
            # Save to images directory
            if debug_output_dir is not None:
                heatmap_filename = os.path.join(debug_output_dir, "bin_densities_heatmap.png")
            else:
                # Fallback to default location if debug_output_dir not set
                heatmap_filename = os.path.join(OUTPUT_DIR, "bin_densities_heatmap.png")
            plt.savefig(heatmap_filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [DEBUG] Saved bin_densities heatmap: {heatmap_filename}")
        except ImportError as e:
            # Matplotlib not available - log to file but continue training
            _log_plot_failure(heatmap_filename or "bin_densities_heatmap.png", e)
        except Exception as e:
            # Log failure but continue training
            _log_plot_failure(heatmap_filename or "bin_densities_heatmap.png", e)
            print(f"  [DEBUG] Warning: Could not save bin_densities heatmap: {e}")
    
    # Compute bin capacity (area of each bin)
    bin_capacity = bin_size * bin_size
    
    # Compute overflow (excess density above bin capacity)
    overflow = torch.relu(bin_densities - bin_capacity)  # [num_bins_x, num_bins_y]
    
    # Apply lightweight smoothing to overflow when blur is enabled
    # Blur helps distribute pressure more smoothly across neighboring bins, reducing jitter
    # and helping escape local traps by providing gradients to nearby underutilized regions
    if blur:
        # Add batch and channel dimensions for conv2d: [1, 1, H, W]
        overflow_4d = overflow.unsqueeze(0).unsqueeze(0)  # [1, 1, num_bins_x, num_bins_y]
        
        # Create 3x3 average blur kernel (all values = 1/9)
        kernel_3x3 = torch.ones(
            (1, 1, 3, 3),
            dtype=cell_features.dtype,
            device=cell_features.device
        ) / 9.0  # [1, 1, 3, 3]
        
        # Apply 3x3 average blur with padding to maintain size
        overflow_blurred = torch.nn.functional.conv2d(
            overflow_4d,
            kernel_3x3,
            padding=1,  # pad all sides to maintain size
            groups=1
        )
        
        # Remove batch and channel dimensions: [num_bins_x, num_bins_y]
        overflow = overflow_blurred.squeeze(0).squeeze(0)
    
    # Superlinear overflow penalty: cubic scaling for stronger repulsion in very crowded bins
    # Using overflow**3 instead of overflow**2 increases gradient magnitude more aggressively
    # for heavily crowded bins, helping resolve persistent hotspots that linear/quadratic penalties
    # struggle with. The cubic penalty grows much faster, providing stronger "pressure" to push
    # cells out of overcrowded regions.
    overflow_scale = 1.0  # Scale constant for easy tuning of overflow penalty strength
    # Clamp overflow to prevent numerical issues before computing cubic
    overflow = torch.clamp(overflow, min=0.0, max=1e4)  # Lower max to prevent overflow**3 from exploding
    # Cubic penalty: superlinear scaling for stronger repulsion in very crowded bins
    overflow_penalty = overflow_scale * (overflow ** 3)
    overflow_penalty = torch.clamp(overflow_penalty, min=0.0, max=1e6)  # Final clamp for numerical stability
    total_overflow = torch.sum(overflow_penalty)
    
    # Normalize by number of bins
    num_bins = torch.tensor(num_bins_x * num_bins_y, dtype=cell_features.dtype, device=cell_features.device, requires_grad=False)
    
    loss = total_overflow / (num_bins + eps)
    # Clamp final loss to prevent NaN/Inf
    loss = torch.clamp(loss, min=0.0, max=1e6)
    
    if should_return_debug:
        # Return detached grids for visualization
        overflow_detached = overflow.detach().cpu()
        density_detached = bin_densities.detach().cpu()
        
        # Save density and overflow heatmaps if debug_output_dir and epoch are provided
        if debug_output_dir is not None and epoch is not None:
            _save_density_overflow_heatmaps(
                density_detached,
                overflow_detached,
                debug_output_dir,
                epoch
            )
        
        return loss, overflow_detached, density_detached
    
    return loss


def overlap_repulsion_loss(cell_features, pin_features, edge_list, multi_res=None, bin_size=None, blur=False, debug=False, debug_output_dir=None, epoch=None):
    """Calculate loss to prevent cell overlaps.

    This function implements a differentiable loss function that penalizes overlapping cells.
    It can operate in two modes:
    - Default: Pairwise overlap detection (exact)
    - Multi-res: Grid-based density overflow on fine and coarse grids (faster for large designs)

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information (not used here)
        edge_list: [E, 2] tensor with edges (not used here)
        multi_res: If True, use multi-resolution grid-based approach. If None, auto-select based on N (True if N > 1000). Default: None
        bin_size: Grid bin size for multi-res mode. If None, auto-computed from cell sizes
        blur: If True and multi_res=True, apply separable blur (1x3 then 3x1) to bin densities. Default: False
        debug: If True and multi_res=True, return tuple of (loss, fine_overflow_grid, coarse_overflow_grid, fine_density_grid, coarse_density_grid). Default: False
        debug_output_dir: Directory to save heatmaps and .pt files. Default: None
        epoch: Epoch number for filename. Default: None

    Returns:
        If debug=False: Scalar loss value (should be 0 when no overlaps exist)
        If debug=True and multi_res=True: Tuple of (loss, fine_overflow_grid, coarse_overflow_grid, fine_density_grid, coarse_density_grid)
        If debug=True and multi_res=False: Scalar loss value (no grid available for pairwise mode)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, dtype=cell_features.dtype, device=cell_features.device, requires_grad=True)

    # Auto-select multi_res for large designs (pairwise is O(N^2), too slow for large N)
    if multi_res is None:
        multi_res = N > 1000
    
    # Enable blur by default for large designs (N >= 5000) to reduce jitter and local traps
    # Blur helps distribute pressure more smoothly across neighboring bins
    if blur is False and N >= 5000:
        blur = True

    if multi_res:
        # Multi-resolution mode: compute density overflow on fine and coarse grids
        if bin_size is None:
            # Auto-compute bin size from average cell size
            avg_width = cell_features[:, 4].mean()
            avg_height = cell_features[:, 5].mean()
            bin_size = (avg_width + avg_height) / 2
        
        # Ensure bin_size is a tensor with proper device/dtype
        if not isinstance(bin_size, torch.Tensor):
            bin_size = torch.tensor(bin_size, dtype=cell_features.dtype, device=cell_features.device)
        
        # Compute loss on fine grid (bin_size)
        fine_result = _compute_grid_density_loss(cell_features, bin_size, blur=blur, debug=debug, return_debug=debug, debug_output_dir=debug_output_dir, epoch=epoch)
        if debug:
            if len(fine_result) == 3:
                fine_loss, fine_overflow_grid, fine_density_grid = fine_result
            else:
                # Backward compatibility
                fine_loss, fine_overflow_grid = fine_result
                fine_density_grid = None
        else:
            fine_loss = fine_result
            fine_density_grid = None
        
        # Compute loss on coarse grid (2 * bin_size)
        coarse_result = _compute_grid_density_loss(cell_features, 2.0 * bin_size, blur=blur, debug=debug, return_debug=debug, debug_output_dir=debug_output_dir, epoch=epoch)
        if debug:
            if len(coarse_result) == 3:
                coarse_loss, coarse_overflow_grid, coarse_density_grid = coarse_result
            else:
                # Backward compatibility
                coarse_loss, coarse_overflow_grid = coarse_result
                coarse_density_grid = None
        else:
            coarse_loss = coarse_result
            coarse_density_grid = None
        
        # Sum the two losses
        total_loss = fine_loss + coarse_loss
        
        if debug:
            return total_loss, fine_overflow_grid, coarse_overflow_grid, fine_density_grid, coarse_density_grid
        return total_loss
    else:
        # Default mode: pairwise overlap detection
        # Extract cell positions, widths, and heights
        positions = cell_features[:, 2:4]  # [N, 2] - x and y positions
        widths = cell_features[:, 4]  # [N] - cell widths
        heights = cell_features[:, 5]  # [N] - cell heights

        # Compute all pairwise distances using broadcasting
        positions_i = positions.unsqueeze(1)  # [N, 1, 2]
        positions_j = positions.unsqueeze(0)  # [1, N, 2]
        distances = positions_i - positions_j  # [N, N, 2]
        dx = distances[:, :, 0]  # [N, N] - x differences
        dy = distances[:, :, 1]  # [N, N] - y differences

        # Calculate minimum separation distances for each pair
        widths_i = widths.unsqueeze(1)  # [N, 1]
        widths_j = widths.unsqueeze(0)  # [1, N]
        min_sep_x = (widths_i + widths_j) / 2  # [N, N] - minimum x separation

        heights_i = heights.unsqueeze(1)  # [N, 1]
        heights_j = heights.unsqueeze(0)  # [1, N]
        min_sep_y = (heights_i + heights_j) / 2  # [N, N] - minimum y separation

        # Use relu to get positive overlap amounts with numerical guards
        overlap_x = torch.relu(min_sep_x - torch.abs(dx))  # [N, N]
        overlap_y = torch.relu(min_sep_y - torch.abs(dy))  # [N, N]
        
        # Clamp to prevent numerical issues
        overlap_x = torch.clamp(overlap_x, min=0.0, max=1e6)
        overlap_y = torch.clamp(overlap_y, min=0.0, max=1e6)

        # Multiply overlaps in x and y to get overlap areas
        overlap_areas = overlap_x * overlap_y  # [N, N]
        overlap_areas = torch.clamp(overlap_areas, min=0.0, max=1e6)

        # Mask to only consider upper triangle (i < j) to avoid double-counting
        mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=cell_features.device, requires_grad=False),
            diagonal=1
        )
        masked_overlaps = overlap_areas[mask]  # [num_pairs]

        # Sum all overlap areas
        total_overlap = torch.sum(masked_overlaps)

        # Normalize by number of pairs: N * (N - 1) / 2
        # Use tensor with proper device/dtype and add eps for numerical stability
        try:
            eps_value = torch.finfo(cell_features.dtype).eps
        except (TypeError, ValueError):
            # Fallback for integer dtypes (shouldn't happen in practice)
            eps_value = 1e-8
        eps = torch.tensor(eps_value, dtype=cell_features.dtype, device=cell_features.device, requires_grad=False)
        num_pairs = torch.tensor(
            N * (N - 1) / 2.0,
            dtype=cell_features.dtype,
            device=cell_features.device,
            requires_grad=False
        )
        normalized_loss = total_overlap / (num_pairs + eps)
        # Clamp to prevent NaN/Inf
        normalized_loss = torch.clamp(normalized_loss, min=0.0, max=1e6)

        return normalized_loss


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.01,
    lambda_wirelength=1.0,
    lambda_overlap=10.0,
    verbose=True,
    log_interval=100,
    compile_overlap_loss=False,
    debug_overlap=False,
    device=None,
    debug=False,
    debug_interval=200,
    output_dir=None,
):
    """Train the placement optimization using gradient descent.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Weight for overlap loss
        verbose: Whether to print progress
        log_interval: How often to print progress
        compile_overlap_loss: If True and PyTorch supports it, compile the overlap loss function for faster execution. Default: False
        debug_overlap: If True and multi_res=True, save overflow heatmaps every debug_interval. Default: False
        device: Device to use ('cpu' or 'cuda'). If None, uses cell_features.device
        debug: If True, print extra diagnostics (gradient stats, position stats, etc.). Default: False
        debug_interval: How often to save debug images (snapshots, heatmaps). Prints still respect log_interval. Default: 200
        output_dir: Dictionary with output directory paths (from setup_output_directory()). If None, uses default debug_out/ directory for backward compatibility.

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    # Move to device if specified
    if device is not None:
        cell_features = cell_features.to(device)
        pin_features = pin_features.to(device)
        edge_list = edge_list.to(device)
    
    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    # Create optimizer
    optimizer = optim.Adam([cell_positions], lr=lr)

    # Debug: Print initial configuration
    if debug and verbose:
        print("\n[DEBUG] Training Configuration:")
        print(f"  Number of cells: {cell_features.shape[0]}")
        print(f"  Number of pins: {pin_features.shape[0]}")
        print(f"  Number of edges: {edge_list.shape[0]}")
        print(f"  Device: {cell_positions.device}")
        print(f"  Learning rate: {lr}")
        print(f"  Lambda wirelength: {lambda_wirelength}")
        print(f"  Lambda overlap: {lambda_overlap}")
        print(f"  Initial position spread: X=[{cell_positions[:, 0].min().item():.2f}, {cell_positions[:, 0].max().item():.2f}], Y=[{cell_positions[:, 1].min().item():.2f}, {cell_positions[:, 1].max().item():.2f}]")
        print()

    # Set up output directories
    if output_dir is not None:
        # Use provided output directory structure
        images_dir = output_dir.get('images_dir', None)
        logs_dir = output_dir.get('logs_dir', None)
        metrics_dir = output_dir.get('metrics_dir', None)
        debug_output_dir = images_dir  # Use images_dir for all debug images
        
        # Set up lightweight logging
        log = setup_logging(logs_dir) if logs_dir else print
    else:
        # Backward compatibility: use default debug_out directory
        images_dir = None
        logs_dir = None
        metrics_dir = None
        debug_output_dir = None
        log = print  # Use regular print for backward compatibility
        if debug:
            debug_output_dir = os.path.join(OUTPUT_DIR, "debug_out")
            try:
                os.makedirs(debug_output_dir, exist_ok=True)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not create debug output directory: {e}")

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }
    
    # Track cells with overlaps over time (for debug mode)
    overlap_cells_history = []  # List of (epoch, num_cells_with_overlaps) tuples
    
    # Track final overflow grid for debug report
    final_overflow_grid = None  # Will store the fine overflow grid from the last epoch

    # Track timing for loss calculations
    total_wl_time = 0.0
    total_overlap_time = 0.0
    training_start_time = time.perf_counter()  # Track total training time for incremental metrics
    
    # Track per-epoch timing for metrics
    per_epoch_timing = {
        "wirelength_time": [],  # milliseconds per epoch
        "overlap_time": [],     # milliseconds per epoch
    }

    # Optionally compile overlap loss function if supported
    # Note: Debug mode is incompatible with compilation, so we skip compilation if debug is enabled
    overlap_loss_fn = overlap_repulsion_loss
    if compile_overlap_loss and not debug_overlap:
        if hasattr(torch, 'compile'):
            try:
                overlap_loss_fn = torch.compile(overlap_repulsion_loss, mode='reduce-overhead')
                if verbose:
                    print("Compiled overlap loss function for faster execution")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compile overlap loss function: {e}")
                    print("Falling back to uncompiled version")
        else:
            if verbose:
                print("Warning: torch.compile not available in this PyTorch version")
                print("Falling back to uncompiled version")
    elif compile_overlap_loss and debug_overlap:
        if verbose:
            print("Warning: Debug mode is incompatible with compilation, using uncompiled version")

    # Print header for compact output
    if verbose:
        log(f"{'epoch':>6}  {'total_loss':>10}  {'wl_loss':>10}  {'overlap_loss':>10}  {'grad_norm':>12}  {'max_update':>12}")
        log("-" * 70)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Create cell_features with current positions
        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        # Calculate losses with timing
        start_time = time.perf_counter()
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        wl_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        total_wl_time += wl_time
        per_epoch_timing["wirelength_time"].append(wl_time)

        start_time = time.perf_counter()
        # Pass debug flag to get overflow grids when debug=True
        overlap_debug_flag = debug or debug_overlap
        # Pass epoch for heatmap saving (only at debug_interval epochs to avoid overhead)
        save_heatmaps_this_epoch = (debug and debug_output_dir is not None and 
                                    (epoch == 0 or epoch == num_epochs - 1 or (epoch % debug_interval == 0 and epoch > 0)))
        overlap_epoch = epoch if save_heatmaps_this_epoch else None
        
        overlap_result = overlap_loss_fn(
            cell_features_current, pin_features, edge_list,
            debug=overlap_debug_flag,
            debug_output_dir=debug_output_dir if debug else None,
            epoch=overlap_epoch
        )
        overlap_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        total_overlap_time += overlap_time
        per_epoch_timing["overlap_time"].append(overlap_time)
        
        # Handle debug mode return value
        fine_overflow_grid = None
        coarse_overflow_grid = None
        fine_density_grid = None
        coarse_density_grid = None
        if overlap_debug_flag and isinstance(overlap_result, tuple):
            if len(overlap_result) == 5:
                # Multi-res mode: (loss, fine_overflow, coarse_overflow, fine_density, coarse_density)
                overlap_loss, fine_overflow_grid, coarse_overflow_grid, fine_density_grid, coarse_density_grid = overlap_result
            elif len(overlap_result) == 3:
                # Multi-res mode (old format): (loss, fine_grid, coarse_grid)
                overlap_loss, fine_overflow_grid, coarse_overflow_grid = overlap_result
            elif len(overlap_result) == 2:
                # Single grid mode: (loss, grid)
                overlap_loss, fine_overflow_grid = overlap_result
            else:
                overlap_loss = overlap_result
        else:
            overlap_loss = overlap_result
        
        # Store final overflow grid for summary dashboard (from last epoch)
        # Capture overflow grid if available (works in both debug and non-debug modes when using grid-based loss)
        if fine_overflow_grid is not None and fine_overflow_grid.numel() > 0:
            final_overflow_grid = fine_overflow_grid.detach().cpu()

        # Check for NaN/Inf in losses before combining
        nan_detected = False
        if torch.isnan(wl_loss) or torch.isinf(wl_loss):
            if verbose and not nan_detected:
                log(f"WARNING: NaN/Inf detected in wirelength loss at epoch {epoch}")
                nan_detected = True
            wl_loss = torch.clamp(wl_loss, min=0.0, max=1e6)
        
        if torch.isnan(overlap_loss) or torch.isinf(overlap_loss):
            if verbose and not nan_detected:
                log(f"WARNING: NaN/Inf detected in overlap loss at epoch {epoch}")
                nan_detected = True
            overlap_loss = torch.clamp(overlap_loss, min=0.0, max=1e6)
        
        # Compute lambda_overlap schedule: two-phase smooth ramp
        # Phase 1 (0%–40%): slow ramp from 0.1 to 1.0
        # Phase 2 (40%–100%): aggressive ramp from 1.0 to 15.0
        # Clamp to 15.0 maximum
        phase1_end_epoch = int(0.4 * num_epochs)
        
        if epoch < phase1_end_epoch:
            # Phase 1: slow ramp from 0.1 to 1.0 over first 40% of training
            progress_phase1 = epoch / phase1_end_epoch  # 0.0 to 1.0
            lambda_overlap_current = 0.1 + (1.0 - 0.1) * progress_phase1
        else:
            # Phase 2: aggressive ramp from 1.0 to 15.0 over remaining 60% of training
            phase2_start_epoch = phase1_end_epoch
            phase2_duration = num_epochs - phase2_start_epoch
            progress_phase2 = (epoch - phase2_start_epoch) / phase2_duration  # 0.0 to 1.0
            lambda_overlap_current = 1.0 + (15.0 - 1.0) * progress_phase2
        
        # Clamp to 15.0 maximum (safety check)
        lambda_overlap_current = min(lambda_overlap_current, 15.0)
        
        # Combined loss with scheduled lambda_overlap
        total_loss = lambda_wirelength * wl_loss + lambda_overlap_current * overlap_loss
        
        # Clamp total loss to prevent NaN/Inf
        total_loss = torch.clamp(total_loss, min=0.0, max=1e10)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if verbose:
                log(f"ERROR: NaN/Inf in total loss at epoch {epoch}, skipping backward pass")
            continue

        # Compute overlap gradient component separately for visualization (before total backward pass)
        overlap_grad_scaled = None
        if debug and debug_output_dir is not None and (epoch % debug_interval == 0 or epoch == num_epochs - 1):
            try:
                # Zero any existing gradients
                if cell_positions.grad is not None:
                    cell_positions.grad.zero_()
                
                # Compute overlap loss gradient component (scaled by lambda_overlap)
                overlap_loss_scaled = lambda_overlap_current * overlap_loss
                overlap_loss_scaled.backward(retain_graph=True)
                
                # Store the scaled overlap gradient for visualization
                if cell_positions.grad is not None:
                    overlap_grad_scaled = cell_positions.grad.clone()
                    cell_positions.grad.zero_()  # Clear for total loss backward
            except Exception as e:
                if verbose:
                    print(f"  [DEBUG] Warning: Could not compute overlap gradient for visualization: {e}")
                overlap_grad_scaled = None

        # Backward pass for total loss (used for optimization)
        total_loss.backward()
        
        # Check for NaN gradients
        if cell_positions.grad is not None:
            if torch.isnan(cell_positions.grad).any() or torch.isinf(cell_positions.grad).any():
                if verbose:
                    log(f"WARNING: NaN/Inf gradients detected at epoch {epoch}, zeroing gradients")
                cell_positions.grad.zero_()
                continue

        # Gradient clipping to prevent extreme updates (adaptive based on loss magnitude)
        max_grad_norm = 5.0
        if total_loss.item() > 100.0:
            max_grad_norm = 2.0  # Tighter clipping when loss is high
        elif total_loss.item() > 10.0:
            max_grad_norm = 3.0
        
        # Debug: Compute gradient statistics before clipping
        grad_norm_before = None
        grad_norm_after = None
        grad_mean = None
        grad_std = None
        grad_max = None
        grad_min = None
        if debug and cell_positions.grad is not None:
            grad_norm_before = torch.norm(cell_positions.grad).item()
            grad_mean = cell_positions.grad.mean().item()
            grad_std = cell_positions.grad.std().item()
            grad_max = cell_positions.grad.abs().max().item()
            grad_min = cell_positions.grad.abs().min().item()
        
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=max_grad_norm)
        
        # Compute gradient norm for logging (needed for both debug and normal output)
        grad_norm = None
        if cell_positions.grad is not None:
            grad_norm = torch.norm(cell_positions.grad).item()
        
        # Save quiver plot for gradients/pressure visualization when debug is enabled
        # Visualize the scaled overlap gradient (pressure from overlap term) to match optimization forces
        if debug and overlap_grad_scaled is not None and debug_output_dir is not None and (epoch % debug_interval == 0 or epoch == num_epochs - 1):
            try:
                save_gradient_quiver_plot(
                    cell_positions,
                    overlap_grad_scaled,
                    debug_output_dir,
                    epoch,
                    lambda_overlap=lambda_overlap_current,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"  [DEBUG] Warning: Could not save gradient quiver plot: {e}")
        
        # Debug: Compute gradient statistics after clipping
        if debug and cell_positions.grad is not None:
            grad_norm_after = grad_norm

        # Store positions before update to compute step size
        positions_before = cell_positions.clone().detach() if (verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1)) else None

        # Update positions
        optimizer.step()
        
        # Compute max absolute update size
        max_update_size = None
        if positions_before is not None:
            update = cell_positions - positions_before
            max_update_size = update.abs().max().item()
        
        # Debug: Compute position statistics
        pos_x = None
        pos_y = None
        pos_spread_x = None
        pos_spread_y = None
        pos_mean_x = None
        pos_mean_y = None
        pos_std_x = None
        pos_std_y = None
        if debug:
            pos_x = cell_positions[:, 0]
            pos_y = cell_positions[:, 1]
            pos_spread_x = (pos_x.max() - pos_x.min()).item()
            pos_spread_y = (pos_y.max() - pos_y.min()).item()
            pos_mean_x = pos_x.mean().item()
            pos_mean_y = pos_y.mean().item()
            pos_std_x = pos_x.std().item()
            pos_std_y = pos_y.std().item()

        # Record losses
        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())
        
        # Save placement snapshots at debug_interval epochs when debug is enabled
        # Always save at epoch 0 and final epoch, plus at debug_interval intervals
        if debug and debug_output_dir is not None and (epoch == 0 or epoch == num_epochs - 1 or (epoch % debug_interval == 0 and epoch > 0)):
            try:
                # Clone cell features for snapshot (save_placement_snapshot runs in background thread)
                cell_features_snapshot = cell_features_current.clone()
                snapshot_filepath = os.path.join(debug_output_dir, f"snap_epoch_{epoch}.png")
                title = f"Epoch {epoch} - Loss: {total_loss.item():.6f}"
                thread = save_placement_snapshot(cell_features_snapshot, snapshot_filepath, title=title)
                if thread is not None and verbose:
                    log(f"  Saving snapshot: {snapshot_filepath} (background thread)")
            except Exception as e:
                if verbose:
                    log(f"  Warning: Could not save snapshot at epoch {epoch}: {e}")

        # Track cells with overlaps every log_interval (always track for overlap_cells_over_time plot)
        # Check for early stopping when debug=True and overlaps reach zero
        should_stop_early = False
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            try:
                # Move to CPU for calculate_cells_with_overlaps (uses numpy)
                cell_features_cpu = cell_features_current.cpu()
                cells_with_overlaps = calculate_cells_with_overlaps(cell_features_cpu)
                num_cells_with_overlaps = len(cells_with_overlaps)
                overlap_cells_history.append((epoch, num_cells_with_overlaps))
                
                # Early stopping: if debug=True and overlaps reach zero, stop training
                if debug and num_cells_with_overlaps == 0:
                    should_stop_early = True
                    if verbose:
                        log(f"  [EARLY STOP] All overlaps eliminated at epoch {epoch} ({num_cells_with_overlaps} cells with overlaps)")
            except Exception as e:
                pass  # Silently skip if calculation fails
        
        # Save metrics incrementally after each log_interval (atomic writes)
        if metrics_dir is not None and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            try:
                import json
                # Calculate elapsed time so far
                elapsed_time_seconds = time.perf_counter() - training_start_time
                
                # Save loss_history.json incrementally
                loss_history_data = {
                    "total_loss": [float(x) for x in loss_history["total_loss"]],
                    "wirelength_loss": [float(x) for x in loss_history["wirelength_loss"]],
                    "overlap_loss": [float(x) for x in loss_history["overlap_loss"]]
                }
                loss_history_file = os.path.join(metrics_dir, "loss_history.json")
                _atomic_write_json(loss_history_data, loss_history_file)
                
                # Save runtime.json with elapsed time so far
                runtime_data = {
                    "elapsed_time_seconds": float(elapsed_time_seconds),
                    "current_epoch": epoch,
                    "total_epochs": num_epochs,
                    "per_epoch_timing": {
                        "wirelength_time_ms": [float(x) for x in per_epoch_timing["wirelength_time"]],
                        "overlap_time_ms": [float(x) for x in per_epoch_timing["overlap_time"]]
                    },
                    "cumulative_timing": {
                        "total_wirelength_time_ms": float(total_wl_time),
                        "total_overlap_time_ms": float(total_overlap_time),
                        "total_training_time_ms": float(total_wl_time + total_overlap_time)
                    }
                }
                runtime_file = os.path.join(metrics_dir, "runtime.json")
                _atomic_write_json(runtime_data, runtime_file)
            except Exception:
                pass  # Silently fail - never crash training
        
        # Early stopping check: break out of training loop if overlaps reached zero in debug mode
        if should_stop_early:
            if verbose:
                log(f"Stopping training early at epoch {epoch} (all overlaps eliminated)")
            break  # Exit training loop early
        
        # Log progress and save debug visualizations
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            # Compact aligned output
            grad_norm_str = f"{grad_norm:12.4f}" if grad_norm is not None else f"{'N/A':>12}"
            max_update_str = f"{max_update_size:12.4f}" if max_update_size is not None else f"{'N/A':>12}"
            log(f"{epoch:6d}  {total_loss.item():10.6f}  {wl_loss.item():10.6f}  {overlap_loss.item():10.6f}  {grad_norm_str}  {max_update_str}")
            
            # Additional timing info (keep for compatibility)
            avg_wl_time = total_wl_time / (epoch + 1)
            avg_overlap_time = total_overlap_time / (epoch + 1)
            log(f"  Avg Wirelength Loss Time: {avg_wl_time:.2f} ms/epoch")
            log(f"  Avg Overlap Loss Time: {avg_overlap_time:.2f} ms/epoch")
            
            # Debug diagnostics (only when debug=True)
            if debug:
                log(f"  [DEBUG] Loss Components:")
                log(f"    Wirelength contribution: {lambda_wirelength * wl_loss.item():.6f}")
                log(f"    Overlap contribution: {lambda_overlap_current * overlap_loss.item():.6f}")
                log(f"    Lambda overlap (scheduled): {lambda_overlap_current:.4f}")
                if cell_positions.grad is not None and grad_norm_after is not None:
                    log(f"  [DEBUG] Gradient Stats (after clipping):")
                    log(f"    Norm: {grad_norm_after:.6f} (before: {grad_norm_before:.6f}, max_norm: {max_grad_norm:.2f})")
                    log(f"    Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")
                    log(f"    Max abs: {grad_max:.6f}, Min abs: {grad_min:.6f}")
                if pos_x is not None:
                    log(f"  [DEBUG] Position Stats:")
                    log(f"    X: mean={pos_mean_x:.2f}, std={pos_std_x:.2f}, spread={pos_spread_x:.2f}")
                    log(f"    Y: mean={pos_mean_y:.2f}, std={pos_std_y:.2f}, spread={pos_spread_y:.2f}")
                    log(f"    Bounds: X=[{pos_x.min().item():.2f}, {pos_x.max().item():.2f}], Y=[{pos_y.min().item():.2f}, {pos_y.max().item():.2f}]")
                
                # Calculate and print overlap metrics (monitoring only, not used in loss)
                try:
                    # Move to CPU for calculate_overlap_metrics (uses numpy)
                    cell_features_cpu = cell_features_current.cpu()
                    overlap_metrics = calculate_overlap_metrics(cell_features_cpu)
                    log(f"  [DEBUG] Overlap Metrics (ground truth):")
                    log(f"    Overlap count (pairs): {overlap_metrics['overlap_count']}")
                    log(f"    Total overlap area: {overlap_metrics['total_overlap_area']:.6f}")
                    log(f"    Max overlap area: {overlap_metrics['max_overlap_area']:.6f}")
                    # Note: num_cells_with_overlaps is already tracked above (independent of verbose)
                except Exception as e:
                    if verbose:
                        log(f"  [DEBUG] Warning: Could not calculate overlap metrics: {e}")
                
                # Print top 5 overflow values from density grids if available
                if fine_overflow_grid is not None and fine_overflow_grid.numel() > 0:
                    fine_flat = fine_overflow_grid.flatten()
                    fine_top5_values, fine_top5_indices = torch.topk(fine_flat, min(5, fine_flat.numel()))
                    fine_top5_values = fine_top5_values.numpy()
                    fine_top5_indices = fine_top5_indices.numpy()
                    
                    fine_shape = fine_overflow_grid.shape
                    log(f"  [DEBUG] Top 5 Fine Grid Overflow Bins:")
                    for i, (val, idx) in enumerate(zip(fine_top5_values, fine_top5_indices)):
                        if val > 0:
                            y_idx = idx // fine_shape[1]
                            x_idx = idx % fine_shape[1]
                            log(f"    {i+1}. Overflow: {val:.6f} at bin ({x_idx}, {y_idx})")
                
                if coarse_overflow_grid is not None and coarse_overflow_grid.numel() > 0:
                    coarse_flat = coarse_overflow_grid.flatten()
                    coarse_top5_values, coarse_top5_indices = torch.topk(coarse_flat, min(5, coarse_flat.numel()))
                    coarse_top5_values = coarse_top5_values.numpy()
                    coarse_top5_indices = coarse_top5_indices.numpy()
                    
                    coarse_shape = coarse_overflow_grid.shape
                    log(f"  [DEBUG] Top 5 Coarse Grid Overflow Bins:")
                    for i, (val, idx) in enumerate(zip(coarse_top5_values, coarse_top5_indices)):
                        if val > 0:
                            y_idx = idx // coarse_shape[1]
                            x_idx = idx % coarse_shape[1]
                            log(f"    {i+1}. Overflow: {val:.6f} at bin ({x_idx}, {y_idx})")
                
                current_device = cell_positions.device
                if current_device.type == 'cuda' and torch.cuda.is_available():
                    log(f"  [DEBUG] CUDA Memory: allocated={torch.cuda.memory_allocated(current_device)/1e6:.1f}MB, reserved={torch.cuda.memory_reserved(current_device)/1e6:.1f}MB")
            
            # Print top 5 bin overflow values if debug is enabled
            if debug_overlap and isinstance(overlap_result, tuple):
                # Find top 5 overflow values from fine grid
                if fine_overflow_grid.numel() > 0:
                    fine_flat = fine_overflow_grid.flatten()
                    fine_top5_values, fine_top5_indices = torch.topk(fine_flat, min(5, fine_flat.numel()))
                    fine_top5_values = fine_top5_values.numpy()
                    fine_top5_indices = fine_top5_indices.numpy()
                    
                    # Convert flat indices to (x, y) coordinates
                    fine_shape = fine_overflow_grid.shape
                    print("  Top 5 Fine Grid Overflow Bins:")
                    for i, (val, idx) in enumerate(zip(fine_top5_values, fine_top5_indices)):
                        if val > 0:
                            y_idx = idx // fine_shape[1]
                            x_idx = idx % fine_shape[1]
                            print(f"    {i+1}. Overflow: {val:.6f} at bin ({x_idx}, {y_idx})")
                
                # Find top 5 overflow values from coarse grid
                if coarse_overflow_grid.numel() > 0:
                    coarse_flat = coarse_overflow_grid.flatten()
                    coarse_top5_values, coarse_top5_indices = torch.topk(coarse_flat, min(5, coarse_flat.numel()))
                    coarse_top5_values = coarse_top5_values.numpy()
                    coarse_top5_indices = coarse_top5_indices.numpy()
                    
                    # Convert flat indices to (x, y) coordinates
                    coarse_shape = coarse_overflow_grid.shape
                    print("  Top 5 Coarse Grid Overflow Bins:")
                    for i, (val, idx) in enumerate(zip(coarse_top5_values, coarse_top5_indices)):
                        if val > 0:
                            y_idx = idx // coarse_shape[1]
                            x_idx = idx % coarse_shape[1]
                            print(f"    {i+1}. Overflow: {val:.6f} at bin ({x_idx}, {y_idx})")
            
            # Save overflow heatmaps at debug_interval epochs when debug is enabled
            # Always save at epoch 0 and final epoch, plus at debug_interval intervals
            if debug_overlap and isinstance(overlap_result, tuple) and (epoch == 0 or epoch == num_epochs - 1 or (epoch % debug_interval == 0 and epoch > 0)):
                filename = None
                try:
                    import matplotlib
                    matplotlib.use("Agg", force=True)
                    import matplotlib.pyplot as plt
                    
                    # Create figure with two subplots for fine and coarse grids
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Plot fine grid overflow
                    if fine_overflow_grid.numel() > 0:
                        im1 = ax1.imshow(fine_overflow_grid.numpy(), cmap='hot', interpolation='nearest', origin='lower')
                        ax1.set_title(f'Fine Grid Overflow (Epoch {epoch})')
                        ax1.set_xlabel('Bin X')
                        ax1.set_ylabel('Bin Y')
                        plt.colorbar(im1, ax=ax1, label='Overflow Area')
                    else:
                        ax1.text(0.5, 0.5, 'No overflow data', ha='center', va='center', transform=ax1.transAxes)
                        ax1.set_title(f'Fine Grid Overflow (Epoch {epoch})')
                    
                    # Plot coarse grid overflow
                    if coarse_overflow_grid.numel() > 0:
                        im2 = ax2.imshow(coarse_overflow_grid.numpy(), cmap='hot', interpolation='nearest', origin='lower')
                        ax2.set_title(f'Coarse Grid Overflow (Epoch {epoch})')
                        ax2.set_xlabel('Bin X')
                        ax2.set_ylabel('Bin Y')
                        plt.colorbar(im2, ax=ax2, label='Overflow Area')
                    else:
                        ax2.text(0.5, 0.5, 'No overflow data', ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title(f'Coarse Grid Overflow (Epoch {epoch})')
                    
                    plt.tight_layout()
                    if debug_output_dir is not None:
                        filename = os.path.join(debug_output_dir, f"overflow_heatmap_epoch_{epoch}.png")
                    else:
                        filename = f"overflow_heatmap_epoch_{epoch}.png"
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    if verbose:
                        print(f"  Saved overflow heatmap: {filename}")
                except ImportError as e:
                    _log_plot_failure(filename or f"overflow_heatmap_epoch_{epoch}.png", e)
                    if verbose:
                        print("  Warning: matplotlib not available, skipping overflow heatmap")
                except Exception as e:
                    _log_plot_failure(filename or f"overflow_heatmap_epoch_{epoch}.png", e)
                    if verbose:
                        print(f"  Warning: Could not save overflow heatmap: {e}")

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()
    
    # Save loss curves at the end of training (always save if output_dir is provided)
    if images_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            
            epochs = list(range(len(loss_history["total_loss"])))
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(epochs, loss_history["total_loss"], label="Total Loss", linewidth=2)
            ax.plot(epochs, loss_history["wirelength_loss"], label="Wirelength Loss", linewidth=2)
            ax.plot(epochs, loss_history["overlap_loss"], label="Overlap Loss", linewidth=2)
            
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Training Loss Curves", fontsize=14)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            loss_curves_filename = os.path.join(images_dir, "loss_curves.png")
            plt.savefig(loss_curves_filename, dpi=150, bbox_inches="tight")
            plt.close()
            
            if verbose:
                log(f"Saved loss curves: {loss_curves_filename}")
        except ImportError as e:
            _log_plot_failure("loss_curves.png", e)
            if verbose:
                log("Warning: matplotlib not available, skipping loss curves plot")
        except Exception as e:
            _log_plot_failure("loss_curves.png", e)
            if verbose:
                log(f"Warning: Could not save loss curves: {e}")
    
    # Save overlap cells over time plot at the end of training (always save if we have tracking data and output_dir is provided)
    if overlap_cells_history and images_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            
            epochs_tracked = [e for e, _ in overlap_cells_history]
            num_cells_tracked = [n for _, n in overlap_cells_history]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(epochs_tracked, num_cells_tracked, label="Cells with Overlaps", linewidth=2, marker='o', markersize=4)
            
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Number of Cells with Overlaps", fontsize=12)
            ax.set_title("Cells with Overlaps Over Time", fontsize=14)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            overlap_cells_filename = os.path.join(images_dir, "overlap_cells_over_time.png")
            plt.savefig(overlap_cells_filename, dpi=150, bbox_inches="tight")
            plt.close()
            
            if verbose:
                log(f"Saved overlap cells over time: {overlap_cells_filename}")
        except ImportError as e:
            _log_plot_failure("overlap_cells_over_time.png", e)
            if verbose:
                log("Warning: matplotlib not available, skipping overlap cells plot")
        except Exception as e:
            _log_plot_failure("overlap_cells_over_time.png", e)
            if verbose:
                log(f"Warning: Could not save overlap cells plot: {e}")
    
    # Save combined summary dashboard (always save at end of training)
    if images_dir is not None:
        save_debug_report(
            final_cell_features,
            final_overflow_grid,
            loss_history,
            images_dir,
            verbose=verbose
        )
        
        # If overlaps remain, save a plot highlighting overlapping cells
        if debug and debug_output_dir is not None:
            try:
                # Move to CPU for calculate_cells_with_overlaps (uses numpy)
                final_cell_features_cpu = final_cell_features.cpu() if final_cell_features.device.type == 'cuda' else final_cell_features
                cells_with_overlaps = calculate_cells_with_overlaps(final_cell_features_cpu)
                
                if len(cells_with_overlaps) > 0:
                    save_overlap_highlight_plot(
                        final_cell_features,
                        cells_with_overlaps,
                        debug_output_dir,
                        verbose=verbose
                    )
            except Exception as e:
                if verbose:
                    log(f"Warning: Could not save overlap highlight plot: {e}")

    # Save final metrics to JSON files (incremental saves already done, but update final values)
    if output_dir is not None and metrics_dir is not None:
        try:
            import json
            
            # Final save of loss history (already saved incrementally, but ensure final state)
            loss_history_data = {
                "total_loss": [float(x) for x in loss_history["total_loss"]],
                "wirelength_loss": [float(x) for x in loss_history["wirelength_loss"]],
                "overlap_loss": [float(x) for x in loss_history["overlap_loss"]]
            }
            loss_history_file = os.path.join(metrics_dir, "loss_history.json")
            _atomic_write_json(loss_history_data, loss_history_file)
            if verbose:
                log(f"Saved loss history to: {loss_history_file}")
            
            # Calculate and save overlap metrics
            try:
                final_cell_features_cpu = final_cell_features.cpu() if final_cell_features.device.type == 'cuda' else final_cell_features
                overlap_metrics = calculate_overlap_metrics(final_cell_features_cpu)
                # Calculate overlap_ratio: num_cells_with_overlaps / total_cells
                cells_with_overlaps = calculate_cells_with_overlaps(final_cell_features_cpu)
                num_cells_with_overlaps = len(cells_with_overlaps)
                N = final_cell_features_cpu.shape[0]
                overlap_ratio = num_cells_with_overlaps / N if N > 0 else 0.0
                
                overlap_metrics_data = {
                    "overlap_count": int(overlap_metrics['overlap_count']),
                    "overlap_ratio": float(overlap_ratio),
                    "total_overlap_area": float(overlap_metrics['total_overlap_area'])
                }
                overlap_metrics_file = os.path.join(metrics_dir, "overlap_metrics.json")
                with open(overlap_metrics_file, 'w') as f:
                    json.dump(overlap_metrics_data, f, indent=2)
                if verbose:
                    log(f"Saved overlap metrics to: {overlap_metrics_file}")
            except Exception as e:
                if verbose:
                    log(f"Warning: Could not save overlap metrics: {e}")
            
            # Final save of runtime metrics (already saved incrementally, but ensure final state)
            total_elapsed_time = time.perf_counter() - training_start_time
            runtime_data = {
                "elapsed_time_seconds": float(total_elapsed_time),
                "current_epoch": num_epochs - 1,
                "total_epochs": num_epochs,
                "per_epoch_timing": {
                    "wirelength_time_ms": [float(x) for x in per_epoch_timing["wirelength_time"]],
                    "overlap_time_ms": [float(x) for x in per_epoch_timing["overlap_time"]]
                },
                "cumulative_timing": {
                    "total_wirelength_time_ms": float(total_wl_time),
                    "total_overlap_time_ms": float(total_overlap_time),
                    "total_training_time_ms": float(total_wl_time + total_overlap_time)
                }
            }
            runtime_file = os.path.join(metrics_dir, "runtime.json")
            _atomic_write_json(runtime_data, runtime_file)
            if verbose:
                log(f"Saved runtime metrics to: {runtime_file}")
        except Exception as e:
            if verbose:
                log(f"Warning: Could not save metrics files: {e}")

    # Close log file at the end of training
    if output_dir is not None and logs_dir is not None:
        close_logging()
    
    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }


# ======= FINAL EVALUATION CODE (Don't edit this part) =======

def calculate_overlap_metrics(cell_features):
    """Calculate ground truth overlap statistics (non-differentiable).

    This function provides exact overlap measurements for evaluation and reporting.
    Unlike the loss function, this does NOT need to be differentiable.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]

    Returns:
        Dictionary with:
            - overlap_count: number of overlapping cell pairs (int)
            - total_overlap_area: sum of all overlap areas (float)
            - max_overlap_area: largest single overlap area (float)
            - overlap_percentage: percentage of total area that overlaps (float)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "overlap_percentage": 0.0,
        }

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()  # [N, 2]
    widths = cell_features[:, 4].detach().numpy()  # [N]
    heights = cell_features[:, 5].detach().numpy()  # [N]
    areas = cell_features[:, 0].detach().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    overlap_areas = []

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                overlap_areas.append(overlap_area)

    # Calculate percentage of total area
    total_area = sum(areas)
    overlap_percentage = (overlap_count / N * 100) if total_area > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "overlap_percentage": overlap_percentage,
    }


def calculate_cells_with_overlaps(cell_features):
    """Calculate number of cells involved in at least one overlap.

    This metric matches the test suite evaluation criteria.

    Args:
        cell_features: [N, 6] tensor with cell properties

    Returns:
        Set of cell indices that have overlaps with other cells
    """
    N = cell_features.shape[0]
    if N <= 1:
        return set()

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()
    widths = cell_features[:, 4].detach().numpy()
    heights = cell_features[:, 5].detach().numpy()

    cells_with_overlaps = set()

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                cells_with_overlaps.add(i)
                cells_with_overlaps.add(j)

    return cells_with_overlaps


def calculate_normalized_metrics(cell_features, pin_features, edge_list):
    """Calculate normalized overlap and wirelength metrics for test suite.

    These metrics match the evaluation criteria in the test suite.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity

    Returns:
        Dictionary with:
            - overlap_ratio: (num cells with overlaps / total cells)
            - normalized_wl: (wirelength / num nets) / sqrt(total area)
            - num_cells_with_overlaps: number of unique cells involved in overlaps
            - total_cells: total number of cells
            - num_nets: number of nets (edges)
    """
    N = cell_features.shape[0]

    # Calculate overlap metric: num cells with overlaps / total cells
    cells_with_overlaps = calculate_cells_with_overlaps(cell_features)
    num_cells_with_overlaps = len(cells_with_overlaps)
    overlap_ratio = num_cells_with_overlaps / N if N > 0 else 0.0

    # Calculate wirelength metric: (wirelength / num nets) / sqrt(total area)
    if edge_list.shape[0] == 0:
        normalized_wl = 0.0
        num_nets = 0
    else:
        # Calculate total wirelength using the loss function (unnormalized)
        wl_loss = wirelength_attraction_loss(cell_features, pin_features, edge_list)
        total_wirelength = wl_loss.item() * edge_list.shape[0]  # Undo normalization

        # Calculate total area
        total_area = cell_features[:, 0].sum().item()

        num_nets = edge_list.shape[0]

        # Normalize: (wirelength / net) / sqrt(area)
        # This gives a dimensionless quality metric independent of design size
        normalized_wl = (total_wirelength / num_nets) / (total_area ** 0.5) if total_area > 0 else 0.0

    return {
        "overlap_ratio": overlap_ratio,
        "normalized_wl": normalized_wl,
        "num_cells_with_overlaps": num_cells_with_overlaps,
        "total_cells": N,
        "num_nets": num_nets,
    }


def save_debug_report(final_cell_features, final_overflow_grid, loss_history, output_dir, verbose=False):
    """Create a combined summary dashboard with final placement, overflow heatmap, and loss curves.
    
    Creates a 1×3 layout dashboard saved as report.png:
    - Final placement snapshot
    - Final overflow heatmap (if available)
    - Loss curves
    
    Args:
        final_cell_features: [N, 6] tensor with final cell positions and properties
        final_overflow_grid: Overflow grid from the final epoch (can be None if not using grid-based loss)
        loss_history: Dictionary with 'total_loss', 'wirelength_loss', 'overlap_loss' lists
        output_dir: Directory to save the report (should be images_dir)
        verbose: If True, print status messages
    
    Returns:
        True if successful, False if matplotlib is not available or on error
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Create figure with 1x3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Subplot 1: Final placement snapshot
        N = final_cell_features.shape[0]
        if N > 0:
            # Move to CPU if needed
            cell_features_cpu = final_cell_features.cpu() if final_cell_features.device.type == 'cuda' else final_cell_features
            
            # Downsample for large designs to avoid performance issues when plotting many rectangles
            # If N > 20000, randomly sample 10000 cells for visualization
            # This keeps the visual representation manageable while preserving overall placement structure
            MAX_PLOT_CELLS = 10000
            DOWNSAMPLE_THRESHOLD = 20000
            if N > DOWNSAMPLE_THRESHOLD:
                import numpy as np
                # Use a fixed seed for reproducibility (seed based on N to get consistent sampling)
                rng = np.random.RandomState(seed=42)
                indices = rng.choice(N, size=MAX_PLOT_CELLS, replace=False)
                cell_features_plot = cell_features_cpu[indices]
                plot_note = f"\n(showing {MAX_PLOT_CELLS} of {N} cells)"
            else:
                cell_features_plot = cell_features_cpu
                indices = None
                plot_note = ""
            
            positions = cell_features_plot[:, 2:4].detach().numpy()
            widths = cell_features_plot[:, 4].detach().numpy()
            heights = cell_features_plot[:, 5].detach().numpy()
            
            # Draw cells (downsampled if N > DOWNSAMPLE_THRESHOLD)
            plot_N = cell_features_plot.shape[0]
            for i in range(plot_N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax1.add_patch(rect)
            
            # Calculate and display overlap metrics (use full dataset, not downsampled)
            metrics = calculate_overlap_metrics(cell_features_cpu)
            title1 = f"Final Placement\nOverlaps: {metrics['overlap_count']}, Area: {metrics['total_overlap_area']:.2f}{plot_note}"
            ax1.set_title(title1, fontsize=11)
            
            ax1.set_aspect("equal")
            ax1.grid(True, alpha=0.3)
            
            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax1.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax1.set_ylim(all_y.min() - margin, all_y.max() + margin)
            ax1.set_xlabel("X", fontsize=10)
            ax1.set_ylabel("Y", fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Final Placement", fontsize=11)
        
        # Subplot 2: Overflow heatmap
        # Note: Heatmaps use fixed grid size and are not downsampled - they represent density at bin resolution
        if final_overflow_grid is not None and final_overflow_grid.numel() > 0:
            im = ax2.imshow(final_overflow_grid.numpy(), cmap='hot', interpolation='nearest', origin='lower')
            ax2.set_title('Final Overflow Heatmap', fontsize=11)
            ax2.set_xlabel('Bin X', fontsize=10)
            ax2.set_ylabel('Bin Y', fontsize=10)
            plt.colorbar(im, ax=ax2, label='Overflow Area')
        else:
            ax2.text(0.5, 0.5, 'No overflow data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Final Overflow Heatmap', fontsize=11)
        
        # Subplot 3: Loss curves
        epochs = list(range(len(loss_history["total_loss"])))
        if epochs:
            ax3.plot(epochs, loss_history["total_loss"], label="Total Loss", linewidth=2)
            ax3.plot(epochs, loss_history["wirelength_loss"], label="Wirelength Loss", linewidth=2)
            ax3.plot(epochs, loss_history["overlap_loss"], label="Overlap Loss", linewidth=2)
            ax3.set_xlabel("Epoch", fontsize=10)
            ax3.set_ylabel("Loss", fontsize=10)
            ax3.set_title("Training Loss Curves", fontsize=11)
            ax3.legend(loc="best", fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Training Loss Curves", fontsize=11)
        
        plt.tight_layout()
        report_filename = os.path.join(output_dir, "report.png")
        plt.savefig(report_filename, dpi=150, bbox_inches="tight")
        plt.close()
        
        if verbose:
            print(f"Saved summary dashboard: {report_filename}")
        return True
        
    except ImportError as e:
        _log_plot_failure("report.png", e)
        if verbose:
            print("Warning: matplotlib not available, skipping summary dashboard")
        return False
    except Exception as e:
        _log_plot_failure("report.png", e)
        if verbose:
            print(f"Warning: Could not save summary dashboard: {e}")
        return False


def save_gradient_quiver_plot(cell_positions, gradients, output_dir, epoch, lambda_overlap=None, verbose=False):
    """Save a quiver plot visualizing gradients/pressure as force vectors on a 32x32 grid.
    
    This function visualizes the overlap gradient component scaled by lambda_overlap,
    showing the actual pressure forces used in optimization.
    
    Args:
        cell_positions: [N, 2] tensor with cell positions
        gradients: [N, 2] tensor with gradients (pressure/force vectors) - should be overlap gradient scaled by lambda_overlap
        output_dir: Directory to save the plot
        epoch: Current epoch number
        lambda_overlap: Current lambda_overlap value used for scaling (for title/annotation)
        verbose: If True, print status messages
    
    Returns:
        True if successful, False if matplotlib is not available or on error
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Move to CPU if needed
        positions_cpu = cell_positions.detach().cpu().numpy() if cell_positions.device.type == 'cuda' else cell_positions.detach().cpu().numpy()
        grads_cpu = gradients.detach().cpu().numpy() if gradients.device.type == 'cuda' else gradients.detach().cpu().numpy()
        
        N = positions_cpu.shape[0]
        if N == 0:
            return False
        
        # Compute grid bounds
        x_min, x_max = positions_cpu[:, 0].min(), positions_cpu[:, 0].max()
        y_min, y_max = positions_cpu[:, 1].min(), positions_cpu[:, 1].max()
        
        # Add margin
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
        
        # Create 32x32 grid
        grid_size = 32
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate gradients to grid points using inverse distance weighting
        # For each grid point, compute weighted average of nearby cell gradients
        U_grid = np.zeros((grid_size, grid_size))
        V_grid = np.zeros((grid_size, grid_size))
        
        # Use Gaussian weighting based on distance
        sigma = max((x_max - x_min) / grid_size, (y_max - y_min) / grid_size) * 2.0
        
        for i in range(grid_size):
            for j in range(grid_size):
                grid_x = X_grid[i, j]
                grid_y = Y_grid[i, j]
                
                # Compute distances from grid point to all cells
                dx = positions_cpu[:, 0] - grid_x
                dy = positions_cpu[:, 1] - grid_y
                distances_sq = dx**2 + dy**2
                
                # Gaussian weights
                weights = np.exp(-distances_sq / (2 * sigma**2))
                weights = weights / (weights.sum() + 1e-10)  # Normalize
                
                # Weighted average of gradients
                U_grid[i, j] = np.sum(weights * grads_cpu[:, 0])
                V_grid[i, j] = np.sum(weights * grads_cpu[:, 1])
        
        # Create quiver plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot cell positions as points
        ax.scatter(positions_cpu[:, 0], positions_cpu[:, 1], c='blue', s=10, alpha=0.5, label='Cells')
        
        # Plot quiver (gradient vectors)
        quiver = ax.quiver(X_grid, Y_grid, U_grid, V_grid, 
                          np.sqrt(U_grid**2 + V_grid**2),  # Color by magnitude
                          cmap='hot', scale=None, alpha=0.7, width=0.003)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        # Title includes lambda_overlap to show the scaling factor used
        title = f'Overlap Pressure Vectors (Epoch {epoch})\n32x32 Grid, {N} cells'
        if lambda_overlap is not None:
            title += f', λ_overlap={lambda_overlap:.3f}'
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar for gradient magnitude
        plt.colorbar(quiver, ax=ax, label='Gradient Magnitude')
        
        plt.tight_layout()
        quiver_filename = os.path.join(output_dir, f"gradient_quiver_epoch_{epoch}.png")
        plt.savefig(quiver_filename, dpi=150, bbox_inches="tight")
        plt.close()
        
        if verbose:
            print(f"  [DEBUG] Saved gradient quiver plot: {quiver_filename}")
        return True
        
    except ImportError as e:
        _log_plot_failure(f"gradient_quiver_epoch_{epoch}.png", e)
        if verbose:
            print("Warning: matplotlib not available, skipping gradient quiver plot")
        return False
    except Exception as e:
        _log_plot_failure(f"gradient_quiver_epoch_{epoch}.png", e)
        if verbose:
            print(f"Warning: Could not save gradient quiver plot: {e}")
        return False


def save_overlap_highlight_plot(cell_features, cells_with_overlaps, output_dir, verbose=False):
    """Save a plot of final placement with overlapping cells highlighted.
    
    Args:
        cell_features: [N, 6] tensor with final cell positions and properties
        cells_with_overlaps: Set of cell indices that have overlaps
        output_dir: Directory to save the plot
        verbose: If True, print status messages
    
    Returns:
        True if successful, False if matplotlib is not available or on error
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np
        
        N = cell_features.shape[0]
        if N == 0:
            return False
        
        # Move to CPU if needed for numpy conversion
        cell_features_cpu = cell_features.cpu() if cell_features.device.type == 'cuda' else cell_features
        
        # Downsample for large designs to avoid performance issues when plotting many rectangles
        # If N > 20000, randomly sample 10000 cells for visualization
        MAX_PLOT_CELLS = 10000
        DOWNSAMPLE_THRESHOLD = 20000
        if N > DOWNSAMPLE_THRESHOLD:
            # Use a fixed seed for reproducibility
            rng = np.random.RandomState(seed=42)
            indices = rng.choice(N, size=MAX_PLOT_CELLS, replace=False)
            cell_features_plot = cell_features_cpu[indices]
            # Map original indices to downsampled indices
            index_map = {orig_idx: plot_idx for plot_idx, orig_idx in enumerate(indices)}
            # Find overlapping cells in the downsampled set
            cells_with_overlaps_plot = {index_map[idx] for idx in cells_with_overlaps if idx in index_map}
            plot_note = f" (showing {MAX_PLOT_CELLS} of {N} cells)"
        else:
            cell_features_plot = cell_features_cpu
            cells_with_overlaps_plot = cells_with_overlaps
            plot_note = ""
        
        positions = cell_features_plot[:, 2:4].detach().numpy()
        widths = cell_features_plot[:, 4].detach().numpy()
        heights = cell_features_plot[:, 5].detach().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Draw all cells first (non-overlapping cells)
        plot_N = cell_features_plot.shape[0]
        for i in range(plot_N):
            if i not in cells_with_overlaps_plot:
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)
        
        # Draw overlapping cells with thicker edges and red color
        num_overlapping_plot = len(cells_with_overlaps_plot)
        for i in cells_with_overlaps_plot:
            x = positions[i, 0] - widths[i] / 2
            y = positions[i, 1] - heights[i] / 2
            rect = Rectangle(
                (x, y),
                widths[i],
                heights[i],
                fill=True,
                facecolor="lightcoral",  # Light red for overlapping cells
                edgecolor="red",
                linewidth=2.5,  # Thicker edge for overlapping cells
                alpha=0.8,
            )
            ax.add_patch(rect)
        
        # Annotate overlapping cell indices (up to 50 cells)
        MAX_ANNOTATIONS = 50
        if num_overlapping_plot > 0:
            # Annotate up to MAX_ANNOTATIONS cells
            cells_to_annotate = list(cells_with_overlaps_plot)[:MAX_ANNOTATIONS]
            for i in cells_to_annotate:
                center_x = positions[i, 0]
                center_y = positions[i, 1]
                # Get original index if downsampled
                if N > DOWNSAMPLE_THRESHOLD:
                    orig_idx = indices[i]
                else:
                    orig_idx = i
                ax.annotate(
                    str(orig_idx),
                    (center_x, center_y),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='darkred',
                    weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7)
                )
        
        # Calculate and display overlap metrics (use full dataset)
        metrics = calculate_overlap_metrics(cell_features_cpu)
        title = f"Final Placement - Overlapping Cells Highlighted\n"
        title += f"Overlaps: {metrics['overlap_count']} pairs, {len(cells_with_overlaps)} cells involved"
        if num_overlapping_plot > MAX_ANNOTATIONS:
            title += f"\n(Annotating first {MAX_ANNOTATIONS} of {num_overlapping_plot} overlapping cells in view)"
        title += plot_note
        
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        
        # Set axis limits with margin
        all_x = positions[:, 0]
        all_y = positions[:, 1]
        margin = 10
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        
        plt.tight_layout()
        overlap_highlight_filename = os.path.join(output_dir, "overlap_highlight.png")
        plt.savefig(overlap_highlight_filename, dpi=150, bbox_inches="tight")
        plt.close()
        
        if verbose:
            print(f"Saved overlap highlight plot: {overlap_highlight_filename}")
        return True
        
    except ImportError as e:
        _log_plot_failure("overlap_highlight.png", e)
        if verbose:
            print("Warning: matplotlib not available, skipping overlap highlight plot")
        return False
    except Exception as e:
        _log_plot_failure("overlap_highlight.png", e)
        if verbose:
            print(f"Warning: Could not save overlap highlight plot: {e}")
        return False


def _save_density_overflow_heatmaps(density_grid, overflow_grid, output_dir, epoch):
    """Save density and overflow heatmaps as PNG images and raw tensors as .pt files.
    
    This function runs in a background thread to avoid blocking training.
    
    Args:
        density_grid: [num_bins_x, num_bins_y] tensor with density values (mass_grid)
        overflow_grid: [num_bins_x, num_bins_y] tensor with overflow values
        output_dir: Directory to save images and .pt files
        epoch: Epoch number for filename
    """
    import threading
    
    def _save_worker():
        """Worker function that runs in background thread."""
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import os
            
            if density_grid.numel() == 0 or overflow_grid.numel() == 0:
                return
            
            # Convert to numpy for visualization
            density_np = density_grid.numpy()
            overflow_np = overflow_grid.numpy()
            
            # Save density heatmap
            density_max = float(density_np.max())
            density_filepath = os.path.join(output_dir, f"density_epoch_{epoch:04d}.png")
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(density_np, cmap='viridis', interpolation='nearest', origin='lower')
            ax.set_title(f'Density Grid (Epoch {epoch})\nMax: {density_max:.4f}')
            ax.set_xlabel('Bin X')
            ax.set_ylabel('Bin Y')
            plt.colorbar(im, ax=ax, label='Density')
            plt.tight_layout()
            plt.savefig(density_filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Save overflow heatmap
            overflow_max = float(overflow_np.max())
            overflow_filepath = os.path.join(output_dir, f"overflow_epoch_{epoch:04d}.png")
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(overflow_np, cmap='hot', interpolation='nearest', origin='lower')
            ax.set_title(f'Overflow Grid (Epoch {epoch})\nMax: {overflow_max:.4f}')
            ax.set_xlabel('Bin X')
            ax.set_ylabel('Bin Y')
            plt.colorbar(im, ax=ax, label='Overflow Area')
            plt.tight_layout()
            plt.savefig(overflow_filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Save raw tensors as .pt files
            density_pt_filepath = os.path.join(output_dir, f"density_epoch_{epoch:04d}.pt")
            overflow_pt_filepath = os.path.join(output_dir, f"overflow_epoch_{epoch:04d}.pt")
            torch.save(density_grid, density_pt_filepath)
            torch.save(overflow_grid, overflow_pt_filepath)
            
        except ImportError as e:
            # Log failures but never crash
            if density_filepath:
                _log_plot_failure(os.path.basename(density_filepath), e)
            if overflow_filepath:
                _log_plot_failure(os.path.basename(overflow_filepath), e)
        except Exception as e:
            # Log failures but never crash
            if density_filepath:
                _log_plot_failure(os.path.basename(density_filepath), e)
            if overflow_filepath:
                _log_plot_failure(os.path.basename(overflow_filepath), e)
    
    # Start background thread (non-blocking)
    try:
        thread = threading.Thread(target=_save_worker, daemon=True)
        thread.start()
    except Exception:
        pass  # Silently fail


def save_placement_snapshot(cell_features, filepath, title=None):
    """Save a snapshot of a single placement visualization.
    
    This function draws all rectangles, downsamples if N > 20k, saves PNG only,
    and never blocks training (runs in background thread).
    
    Args:
        cell_features: [N, 6] tensor with cell positions and properties
        filepath: Full file path for saving the PNG image
        title: Optional title for the plot. If None, uses overlap metrics.
    
    Returns:
        Thread object if started successfully, None otherwise
    """
    import threading
    
    def _save_snapshot_worker():
        """Worker function that runs in background thread to save snapshot."""
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            N = cell_features.shape[0]
            if N == 0:
                return
            
            # Move to CPU if needed for numpy conversion (detach to avoid gradient tracking)
            cell_features_cpu = cell_features.cpu().detach() if cell_features.device.type == 'cuda' else cell_features.detach()
            
            # Downsample for large designs to avoid performance issues when plotting many rectangles
            # If N > 20000, randomly sample 10000 cells for visualization
            # This keeps the visual representation manageable while preserving overall placement structure
            MAX_PLOT_CELLS = 10000
            DOWNSAMPLE_THRESHOLD = 20000
            if N > DOWNSAMPLE_THRESHOLD:
                import numpy as np
                # Use a fixed seed for reproducibility
                rng = np.random.RandomState(seed=42)
                indices = rng.choice(N, size=MAX_PLOT_CELLS, replace=False)
                cell_features_plot = cell_features_cpu[indices]
                plot_note = f" (showing {MAX_PLOT_CELLS} of {N} cells)"
            else:
                cell_features_plot = cell_features_cpu
                plot_note = ""
            
            positions = cell_features_plot[:, 2:4].numpy()
            widths = cell_features_plot[:, 4].numpy()
            heights = cell_features_plot[:, 5].numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Draw all rectangles (downsampled if N > DOWNSAMPLE_THRESHOLD)
            plot_N = cell_features_plot.shape[0]
            for i in range(plot_N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)
            
            # Calculate and display overlap metrics (use full dataset, not downsampled)
            metrics = calculate_overlap_metrics(cell_features_cpu)
            
            # Set title (include downsampling note if applicable)
            if title is None:
                title = f"Overlaps: {metrics['overlap_count']}, Total Overlap Area: {metrics['total_overlap_area']:.2f}{plot_note}"
            else:
                title = f"{title}{plot_note}"
            ax.set_title(title, fontsize=12)
            
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            
            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            
        except ImportError as e:
            _log_plot_failure(os.path.basename(filepath), e)
        except Exception as e:
            _log_plot_failure(os.path.basename(filepath), e)
    
    # Start background thread to save snapshot (non-blocking)
    try:
        thread = threading.Thread(target=_save_snapshot_worker, daemon=True)
        thread.start()
        return thread
    except Exception:
        return None


def plot_placement(
    initial_cell_features,
    final_cell_features,
    pin_features,
    edge_list,
    filename="placement_result.png",
    output_dir=None,
):
    """Create side-by-side visualization of initial vs final placement.

    Args:
        initial_cell_features: Initial cell positions and properties
        final_cell_features: Optimized cell positions and properties
        pin_features: Pin information
        edge_list: Edge connectivity
        filename: Output filename for the plot
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot both initial and final placements
        for ax, cell_features, title in [
            (ax1, initial_cell_features, "Initial Placement"),
            (ax2, final_cell_features, "Final Placement"),
        ]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().numpy()
            widths = cell_features[:, 4].detach().numpy()
            heights = cell_features[:, 5].detach().numpy()

            # Draw cells
            for i in range(N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)

            # Calculate and display overlap metrics
            metrics = calculate_overlap_metrics(cell_features)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{title}\n"
                f"Overlaps: {metrics['overlap_count']}, "
                f"Total Overlap Area: {metrics['total_overlap_area']:.2f}",
                fontsize=12,
            )

            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        plt.tight_layout()
        # Use images_dir if provided, otherwise fallback to OUTPUT_DIR
        if output_dir is not None and 'images_dir' in output_dir:
            output_path = os.path.join(output_dir['images_dir'], filename)
        else:
            output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    except ImportError as e:
        _log_plot_failure(filename, e)
        print(f"Could not create visualization: {e}")
        print("Install matplotlib to enable visualization: pip install matplotlib")
        return None
    except Exception as e:
        _log_plot_failure(filename, e)
        print(f"Could not create visualization: {e}")
        return None

# ======= MAIN FUNCTION =======

def main():
    """Main function demonstrating the placement optimization challenge."""
    print("=" * 70)
    print("VLSI CELL PLACEMENT OPTIMIZATION CHALLENGE")
    print("=" * 70)
    print("\nObjective: Implement overlap_repulsion_loss() to eliminate cell overlaps")
    print("while minimizing wirelength.\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate placement problem
    num_macros = 3
    num_std_cells = 50

    print(f"Generating placement problem:")
    print(f"  - {num_macros} macros")
    print(f"  - {num_std_cells} standard cells")

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Initialize positions with random spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Calculate initial metrics
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    initial_metrics = calculate_overlap_metrics(cell_features)
    print(f"Overlap count: {initial_metrics['overlap_count']}")
    print(f"Total overlap area: {initial_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {initial_metrics['max_overlap_area']:.2f}")
    print(f"Overlap percentage: {initial_metrics['overlap_percentage']:.2f}%")

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)

    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=True,
        log_interval=200,
    )

    # Calculate final metrics (both detailed and normalized)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_cell_features = result["final_cell_features"]

    # Detailed metrics
    final_metrics = calculate_overlap_metrics(final_cell_features)
    print(f"Overlap count (pairs): {final_metrics['overlap_count']}")
    print(f"Total overlap area: {final_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {final_metrics['max_overlap_area']:.2f}")

    # Normalized metrics (matching test suite)
    print("\n" + "-" * 70)
    print("TEST SUITE METRICS (for leaderboard)")
    print("-" * 70)
    normalized_metrics = calculate_normalized_metrics(
        final_cell_features, pin_features, edge_list
    )
    print(f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
          f"({normalized_metrics['num_cells_with_overlaps']}/{normalized_metrics['total_cells']} cells)")
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    if normalized_metrics["num_cells_with_overlaps"] == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print("\nCongratulations! Your implementation successfully eliminated all overlaps.")
        print(f"Your normalized wirelength: {normalized_metrics['normalized_wl']:.4f}")
    else:
        print("✗ FAIL: Overlaps still exist")
        print(f"  Need to eliminate overlaps in {normalized_metrics['num_cells_with_overlaps']} cells")
        print("\nSuggestions:")
        print("  1. Check your overlap_repulsion_loss() implementation")
        print("  2. Change lambdas (try increasing lambda_overlap)")
        print("  3. Change learning rate or number of epochs")

    # Generate visualization
    plot_placement(
        result["initial_cell_features"],
        result["final_cell_features"],
        pin_features,
        edge_list,
        filename="placement_result.png",
    )

if __name__ == "__main__":
    main()
