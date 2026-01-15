"""
Main entry point for VLSI Cell Placement Challenge.
"""

import argparse
import torch

from test import QUICK_TEST_CASES, TEST_CASES, run_all_tests


def set_seed(seed=42):
    """Set random seed for reproducibility.
    
    Only sets CUDA deterministic flags if CUDA is available and being used.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Only set deterministic flags if CUDA is actually being used
        if torch.cuda.is_initialized():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VLSI Cell Placement Challenge test suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 3 quick test cases"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite (default)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: cpu)"
    )
    args = parser.parse_args()
    
    # Validate CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    # Set seed once at the start
    set_seed(42)
    
    # Select test cases
    if args.quick:
        test_cases = QUICK_TEST_CASES
    else:
        test_cases = TEST_CASES
    
    # Run tests and get results
    results = run_all_tests(test_cases, device=args.device)
    
    # Print final metrics
    print(f"Average Overlap: {results['avg_overlap']:.4f}")
    print(f"Average Wirelength: {results['avg_wirelength']:.4f}")
    print(f"Total Runtime: {results['total_time']:.2f}s")


if __name__ == "__main__":
    main()
