"""
Test Harness for VLSI Cell Placement
==============================================

This script runs the placement optimizer on 10 randomly generated netlists
of various sizes and reports metrics

Usage:
    python test_placement.py

Metrics Reported:
    - Average Overlap: (num cells with overlaps / total num cells)
    - Average Wirelength: (total wirelength / num nets) / sqrt(total area)
      This normalization allows fair comparison across different design sizes.

"""

import argparse
import os
import shutil
import time

import torch

# Import from the challenge file
from placement import (
    calculate_normalized_metrics,
    generate_placement_input,
    setup_logging,
    setup_output_directory,
    train_placement,
)


# Test case configurations: (test_id, num_macros, num_std_cells, seed)
TEST_CASES = [
    # Small designs
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    # Medium designs
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    # Large designs
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
    # Realistic designs
    (11, 10, 10000, 1011),
    (12, 10, 100000, 1012),
]

# Quick test cases: (2 macros, 30 std), (5 macros, 150 std), (10 macros, 2000 std)
QUICK_TEST_CASES = [
    (3, 2, 30, 1003),      # Small: 2 macros, 30 std cells
    (7, 5, 150, 1007),     # Large: 5 macros, 150 std cells
    (10, 10, 2000, 1010),  # Large: 10 macros, 2000 std cells
]


def run_placement_test(
    test_id,
    num_macros,
    num_std_cells,
    seed=None,
    device=None,
    debug=False,
    output_dir=None,
):
    """Run placement optimization on a single test case.

    Uses default hyperparameters from train_placement() function.

    Args:
        test_id: Test case identifier
        num_macros: Number of macro cells
        num_std_cells: Number of standard cells
        seed: Random seed for reproducibility
        device: Device to use ('cpu' or 'cuda'). If None, uses 'cpu'
        debug: If True, enable debug mode in train_placement() to generate debug images

    Returns:
        Dictionary with test results and metrics
    """
    if seed:
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Generate netlist
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )
    
    # Move to device (device should always be set, but check for safety)
    if device is not None:
        cell_features = cell_features.to(device)
        pin_features = pin_features.to(device)
        edge_list = edge_list.to(device)
    else:
        # Fallback: use CPU if device not specified
        device = 'cpu'
        cell_features = cell_features.to(device)
        pin_features = pin_features.to(device)
        edge_list = edge_list.to(device)

    # Initialize positions with random spread
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6

    # Create random tensors on the same device as cell_features
    angles = torch.rand(total_cells, device=cell_features.device) * 2 * 3.14159
    radii = torch.rand(total_cells, device=cell_features.device) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Run optimization with default hyperparameters
    start_time = time.time()
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=debug,  # Show output when debug is enabled
        device=device,
        debug=debug,
        output_dir=output_dir,
    )
    elapsed_time = time.time() - start_time

    # Calculate final metrics using shared implementation
    final_cell_features = result["final_cell_features"]
    # Move to CPU for metrics calculation (non-differentiable operations)
    # Check device type instead of comparing string
    final_cell_features_cpu = final_cell_features.cpu() if final_cell_features.device.type == 'cuda' else final_cell_features
    pin_features_cpu = pin_features.cpu() if pin_features.device.type == 'cuda' else pin_features
    edge_list_cpu = edge_list.cpu() if edge_list.device.type == 'cuda' else edge_list
    metrics = calculate_normalized_metrics(final_cell_features_cpu, pin_features_cpu, edge_list_cpu)

    return {
        "test_id": test_id,
        "num_macros": num_macros,
        "num_std_cells": num_std_cells,
        "total_cells": metrics["total_cells"],
        "num_nets": metrics["num_nets"],
        "seed": seed,
        "elapsed_time": elapsed_time,
        # Final metrics
        "num_cells_with_overlaps": metrics["num_cells_with_overlaps"],
        "overlap_ratio": metrics["overlap_ratio"],
        "normalized_wl": metrics["normalized_wl"],
    }


def run_all_tests(test_cases=None, device=None, debug=False, output_dir=None, log=print):
    """Run test cases and compute aggregate metrics.

    Uses default hyperparameters from train_placement() function.

    Args:
        test_cases: List of test cases to run. If None, uses TEST_CASES.
        device: Device to use ('cpu' or 'cuda'). If None, uses 'cpu'
        debug: If True, enable debug mode in train_placement() to generate debug images
        output_dir: Dictionary with output directory paths (from setup_output_directory()). If None, uses default behavior.

    Returns:
        Dictionary with all test results and aggregate statistics
    """
    if device is None:
        device = 'cpu'
    if test_cases is None:
        test_cases = TEST_CASES

    log("=" * 70)
    log("PLACEMENT CHALLENGE TEST SUITE")
    log("=" * 70)
    log(f"\nRunning {len(test_cases)} test cases with various netlist sizes...")
    log("Using default hyperparameters from train_placement()")
    log("")

    all_results = []

    for idx, (test_id, num_macros, num_std_cells, seed) in enumerate(test_cases, 1):
        size_category = (
            "Small" if num_std_cells <= 30
            else "Medium" if num_std_cells <= 100
            else "Large"
        )

        log(f"Test {idx}/{len(test_cases)}: {size_category} ({num_macros} macros, {num_std_cells} std cells)")
        log(f"  Seed: {seed}")

        # Run test
        result = run_placement_test(
            test_id,
            num_macros,
            num_std_cells,
            seed,
            device=device,
            debug=debug,
            output_dir=output_dir,
        )

        all_results.append(result)

        # Print summary
        status = "PASS" if result["num_cells_with_overlaps"] == 0 else "FAIL"
        log(f"  Overlap Ratio: {result['overlap_ratio']:.4f} ({result['num_cells_with_overlaps']}/{result['total_cells']} cells)")
        log(f"  Normalized WL: {result['normalized_wl']:.4f}")
        log(f"  Time: {result['elapsed_time']:.2f}s")
        log(f"  Status: {status}")
        log("")

    # Compute aggregate statistics
    avg_overlap_ratio = sum(r["overlap_ratio"] for r in all_results) / len(all_results)
    avg_normalized_wl = sum(r["normalized_wl"] for r in all_results) / len(all_results)
    total_time = sum(r["elapsed_time"] for r in all_results)

    # Print aggregate results
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    log(f"Average Overlap: {avg_overlap_ratio:.4f}")
    log(f"Average Wirelength: {avg_normalized_wl:.4f}")
    log(f"Total Runtime: {total_time:.2f}s")
    log("")
    
    # Save metrics to file if output directory is provided
    if output_dir is not None and 'metrics_dir' in output_dir:
        try:
            import json
            metrics_dir = output_dir['metrics_dir']
            
            # Save test results (existing)
            metrics_file = os.path.join(metrics_dir, 'test_results.json')
            metrics_data = {
                "avg_overlap": avg_overlap_ratio,
                "avg_wirelength": avg_normalized_wl,
                "total_time": total_time,
                "num_test_cases": len(all_results),
                "individual_results": all_results
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            log(f"Saved test results to: {metrics_file}")
            
            # Save aggregate loss history across all test cases
            # Collect all loss histories from individual test results
            all_loss_histories = []
            for result in all_results:
                # Note: individual test results don't currently store loss history
                # This would need to be added if we want per-test-case loss history
                pass
            
            # Save aggregate overlap metrics
            overlap_metrics_data = {
                "avg_overlap_ratio": avg_overlap_ratio,
                "total_test_cases": len(all_results),
                "cases_with_overlaps": sum(1 for r in all_results if r["num_cells_with_overlaps"] > 0),
                "individual_overlap_ratios": [r["overlap_ratio"] for r in all_results]
            }
            overlap_metrics_file = os.path.join(metrics_dir, "overlap_metrics.json")
            with open(overlap_metrics_file, 'w') as f:
                json.dump(overlap_metrics_data, f, indent=2)
            log(f"Saved overlap metrics to: {overlap_metrics_file}")
            
            # Save runtime metrics
            runtime_data = {
                "total_runtime_seconds": total_time,
                "avg_runtime_per_test_seconds": total_time / len(all_results) if all_results else 0.0,
                "per_test_runtime_seconds": [r["elapsed_time"] for r in all_results],
                "num_test_cases": len(all_results)
            }
            runtime_file = os.path.join(metrics_dir, "runtime.json")
            with open(runtime_file, 'w') as f:
                json.dump(runtime_data, f, indent=2)
            log(f"Saved runtime metrics to: {runtime_file}")
        except Exception as e:
            log(f"Warning: Could not save metrics: {e}")

    return {
        "avg_overlap": avg_overlap_ratio,
        "avg_wirelength": avg_normalized_wl,
        "total_time": total_time,
    }, all_results  # Return both aggregate and individual results


def main():
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(
        description="Run VLSI Cell Placement Challenge test suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 3 quick test cases: (2 macros, 30 std), (5 macros, 150 std), (10 macros, 2000 std)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for computation. 'auto' uses CUDA if available, else CPU (default: auto)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: run only the 2000-cell test case and generate debug images"
    )
    args = parser.parse_args()

    # Handle auto device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    
    # Validate CUDA availability if explicitly requested
    if args.device == "cuda" and not torch.cuda.is_available():
        # Note: This happens before logging is set up, so use print
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    # Set up unified output directory structure
    output_dir = setup_output_directory()
    
    # Set up lightweight logging
    log = setup_logging(output_dir['logs_dir']) if output_dir.get('logs_dir') else print
    
    # Print selected device
    log(f"Device: {args.device}")
    if args.device == "cuda":
        log(f"  CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    log("")
    
    log(f"Output directory: {output_dir['run_dir']}")
    log(f"  - Logs: {output_dir['logs_dir']}")
    log(f"  - Images: {output_dir['images_dir']}")
    log(f"  - Metrics: {output_dir['metrics_dir']}")
    log("")
    
    # Handle debug mode: run two experiments (baseline vs improved)
    if args.debug:
        # Run only the 2000-cell test case (test case 10: 10 macros, 2000 std cells)
        test_cases = [(10, 10, 2000, 1010)]
        log("=" * 70)
        log("DEBUG MODE: Running Baseline vs Improved Comparison")
        log("=" * 70)
        log("Test case: 10 macros, 2000 std cells (seed 1010)")
        log("")
        
        # Store results for comparison
        baseline_result = None
        baseline_individual = None
        improved_result = None
        improved_individual = None
        
        # Experiment 1: Baseline (current settings - all improvements enabled)
        log("=" * 70)
        log("EXPERIMENT 1: BASELINE (Current Settings)")
        log("=" * 70)
        log("Settings: Default hyperparameters with all improvements enabled")
        log("  - Two-phase lambda_overlap schedule (0.1 -> 1.0 -> 15.0)")
        log("  - Superlinear overflow penalty (overflow**3)")
        log("  - Macro-aware area weighting (macro_weight=5.0, std cell cap)")
        log("  - Overflow blur (auto-enabled for N>=5000)")
        log("")
        
        baseline_output_dir = setup_output_directory()
        baseline_log = setup_logging(baseline_output_dir['logs_dir']) if baseline_output_dir.get('logs_dir') else print
        baseline_log(f"Baseline output directory: {baseline_output_dir['run_dir']}")
        baseline_log("")
        
        try:
            baseline_result, baseline_individual = run_all_tests(
                test_cases, 
                device=args.device, 
                debug=True, 
                output_dir=baseline_output_dir, 
                log=baseline_log
            )
        finally:
            from placement import close_logging
            close_logging()
        
        # Extract individual test results
        baseline_overlap_count = 0
        baseline_normalized_wl = 0.0
        if baseline_individual and len(baseline_individual) > 0:
            baseline_overlap_count = baseline_individual[0].get('num_cells_with_overlaps', 0)
            baseline_normalized_wl = baseline_individual[0].get('normalized_wl', 0.0)
        
        log("")
        log("Baseline Results:")
        log(f"  Overlap Count: {baseline_overlap_count} cells")
        log(f"  Normalized Wirelength: {baseline_normalized_wl:.6f}")
        log(f"  Runtime: {baseline_result.get('total_time', 0):.2f}s")
        log("")
        
        # Experiment 2: Improved (same settings - for now, identical to baseline)
        # Note: All improvements are already in the code, so "improved" uses the same settings
        # In a real scenario, you might adjust hyperparameters here
        log("=" * 70)
        log("EXPERIMENT 2: IMPROVED (Same Settings - All Improvements Enabled)")
        log("=" * 70)
        log("Settings: Same as baseline (all improvements are built-in)")
        log("  - Two-phase lambda_overlap schedule (0.1 -> 1.0 -> 15.0)")
        log("  - Superlinear overflow penalty (overflow**3)")
        log("  - Macro-aware area weighting (macro_weight=5.0, std cell cap)")
        log("  - Overflow blur (auto-enabled for N>=5000)")
        log("")
        
        improved_output_dir = setup_output_directory()
        improved_log = setup_logging(improved_output_dir['logs_dir']) if improved_output_dir.get('logs_dir') else print
        improved_log(f"Improved output directory: {improved_output_dir['run_dir']}")
        improved_log("")
        
        try:
            improved_result, improved_individual = run_all_tests(
                test_cases, 
                device=args.device, 
                debug=True, 
                output_dir=improved_output_dir, 
                log=improved_log
            )
        finally:
            from placement import close_logging
            close_logging()
        
        # Extract individual test results
        improved_overlap_count = 0
        improved_normalized_wl = 0.0
        if improved_individual and len(improved_individual) > 0:
            improved_overlap_count = improved_individual[0].get('num_cells_with_overlaps', 0)
            improved_normalized_wl = improved_individual[0].get('normalized_wl', 0.0)
        
        log("")
        log("Improved Results:")
        log(f"  Overlap Count: {improved_overlap_count} cells")
        log(f"  Normalized Wirelength: {improved_normalized_wl:.6f}")
        log(f"  Runtime: {improved_result.get('total_time', 0):.2f}s")
        log("")
        
        # Save comparison file
        try:
            # Use the improved output directory for the comparison file
            comparison_file = os.path.join(improved_output_dir['metrics_dir'], 'comparison.txt')
            with open(comparison_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("BASELINE vs IMPROVED COMPARISON\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Test Case: 10 macros, 2000 std cells (seed 1010)\n")
                f.write(f"Device: {args.device}\n\n")
                
                f.write("-" * 70 + "\n")
                f.write("BASELINE (Current Settings)\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Output Directory: {baseline_output_dir['run_dir']}\n")
                f.write(f"  Overlap Count: {baseline_overlap_count} cells\n")
                f.write(f"  Normalized Wirelength: {baseline_normalized_wl:.6f}\n")
                f.write(f"  Total Runtime: {baseline_result.get('total_time', 0):.2f}s\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("IMPROVED (New Scheduling + Superlinear + Macro-aware + Blur)\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Output Directory: {improved_output_dir['run_dir']}\n")
                f.write(f"  Overlap Count: {improved_overlap_count} cells\n")
                f.write(f"  Normalized Wirelength: {improved_normalized_wl:.6f}\n")
                f.write(f"  Total Runtime: {improved_result.get('total_time', 0):.2f}s\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("DIFFERENCE\n")
                f.write("-" * 70 + "\n")
                overlap_diff = improved_overlap_count - baseline_overlap_count
                wl_diff = improved_normalized_wl - baseline_normalized_wl
                time_diff = improved_result.get('total_time', 0) - baseline_result.get('total_time', 0)
                f.write(f"  Overlap Count Change: {overlap_diff:+d} cells\n")
                f.write(f"  Normalized Wirelength Change: {wl_diff:+.6f}\n")
                f.write(f"  Runtime Change: {time_diff:+.2f}s\n")
                f.write("\n")
                
                f.write("=" * 70 + "\n")
            
            log(f"Saved comparison to: {comparison_file}")
            log("")
        except Exception as e:
            log(f"Warning: Could not save comparison file: {e}")
        
        # Print final comparison summary
        log("=" * 70)
        log("COMPARISON SUMMARY")
        log("=" * 70)
        log(f"Baseline - Overlap Count: {baseline_overlap_count} cells, WL: {baseline_normalized_wl:.6f}")
        log(f"Improved - Overlap Count: {improved_overlap_count} cells, WL: {improved_normalized_wl:.6f}")
        log("")
        
    # Select test cases based on mode
    elif args.quick:
        test_cases = QUICK_TEST_CASES
    else:
        test_cases = TEST_CASES

    # Run tests with selected test cases (non-debug mode)
    if not args.debug:
        try:
            result, _ = run_all_tests(test_cases, device=args.device, debug=args.debug, output_dir=output_dir, log=log)
        finally:
            # Close log file at the end
            from placement import close_logging
            close_logging()


if __name__ == "__main__":
    main()
