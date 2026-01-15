#!/usr/bin/env python
"""Check CUDA availability"""
import sys

print(f"Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Device: CUDA")
    else:
        print("Device: CPU")
        print("Note: CUDA is installed on system but PyTorch cannot access it")
        
except Exception as e:
    print(f"Error loading PyTorch: {e}")
    print("Device: Unknown (PyTorch not working)")
