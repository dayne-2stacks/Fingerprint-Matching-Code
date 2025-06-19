#!/usr/bin/env python3
"""
Verify Pore Detection Setup
"""

import os
import torch
from pathlib import Path

def verify_setup():
    print("=== PORE DETECTION SETUP VERIFICATION ===\n")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Check model files
    model_path = "out_of_the_box_detect/models/40"
    model_exists = os.path.exists(model_path)
    print(f"✓ Model file exists: {model_exists}")
    if model_exists:
        size = os.path.getsize(model_path) / 1024  # KB
        print(f"  Model size: {size:.1f} KB")
    
    # Check dataset
    dataset_path = "../dataset/L3-SF"
    dataset_exists = os.path.exists(dataset_path)
    print(f"✓ Dataset exists: {dataset_exists}")
    if dataset_exists:
        image_count = len(list(Path(dataset_path).rglob("*.png")))
        print(f"  Total images: {image_count}")
    
    # Check output directories
    output_dirs = [
        "out_of_the_box_detect/Prediction/Coordinates",
        "out_of_the_box_detect/Prediction/Fingerprint", 
        "out_of_the_box_detect/Prediction/Pore"
    ]
    
    print("\n✓ Output directories:")
    for dir_path in output_dirs:
        exists = os.path.exists(dir_path)
        print(f"  {dir_path}: {'✓' if exists else '✗'}")
    
    # Check Python modules
    print("\n✓ Required modules:")
    modules = ['torch', 'cv2', 'numpy', 'torchvision', 'tqdm', 'pathlib']
    for module in modules:
        try:
            __import__(module)
            print(f"  {module}: ✓")
        except ImportError:
            print(f"  {module}: ✗ MISSING")
    
    print(f"\n{'='*50}")
    if cuda_available and model_exists and dataset_exists:
        print("🎉 SETUP IS READY FOR PORE DETECTION!")
    else:
        print("⚠️  SETUP ISSUES DETECTED - CHECK ABOVE")
    print(f"{'='*50}")

if __name__ == "__main__":
    verify_setup()
