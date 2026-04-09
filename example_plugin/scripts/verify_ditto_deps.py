#!/usr/bin/env python3
"""
Verification script for Ditto dependencies and installation.
Checks that all required packages are installed and importable.
"""

import sys
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, "Not found"
        
        # Try to actually import it
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Ditto Dependencies Verification")
    print("=" * 60)
    print()
    
    # Critical Ditto dependencies
    dependencies = [
        # TensorRT and CUDA
        ("tensorrt", "tensorrt"),
        ("cuda-python", "cuda"),
        
        # Audio Processing
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("soxr", "soxr"),
        ("audioread", "audioread"),
        
        # Scientific Computing
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("scikit-image", "skimage"),
        
        # Performance
        ("numba", "numba"),
        ("llvmlite", "llvmlite"),
        ("joblib", "joblib"),
        
        # Computer Vision
        ("opencv-python-headless", "cv2"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
        
        # Utilities
        ("polygraphy", "polygraphy"),
        ("colored", "colored"),
        ("cython", "Cython"),
        ("tqdm", "tqdm"),
        ("filetype", "filetype"),
        
        # Deep Learning
        ("torch", "torch"),
    ]
    
    print("Checking critical dependencies:")
    print("-" * 60)
    
    all_ok = True
    missing = []
    errors = []
    
    for package_name, import_name in dependencies:
        ok, version = check_package(package_name, import_name)
        status = "✓" if ok else "✗"
        
        if ok:
            print(f"{status} {package_name:30s} {version}")
        else:
            print(f"{status} {package_name:30s} MISSING")
            all_ok = False
            if "Not found" in version:
                missing.append(package_name)
            else:
                errors.append((package_name, version))
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All dependencies are installed correctly!")
        print()
        
        # Additional checks
        print("Additional Checks:")
        print("-" * 60)
        
        # Check NumPy version
        import numpy as np
        numpy_version = np.__version__
        numpy_major = int(numpy_version.split('.')[0])
        
        if numpy_major >= 2:
            print(f"⚠️  NumPy version {numpy_version} detected (2.x)")
            print("   This may cause issues with TensorFlow 2.15 (EchoMimicV3)")
            print("   Recommended: numpy<2.0.0")
        else:
            print(f"✓ NumPy version {numpy_version} (1.x) - Compatible")
        
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available in PyTorch")
        
        # Check TensorRT version
        import tensorrt as trt
        print(f"✓ TensorRT version: {trt.__version__}")
        
        print()
        return 0
    else:
        print("✗ Some dependencies are missing or have errors!")
        print()
        
        if missing:
            print("Missing packages:")
            for pkg in missing:
                print(f"  - {pkg}")
            print()
            print("Install missing packages with:")
            print("  uv sync")
        
        if errors:
            print()
            print("Packages with errors:")
            for pkg, error in errors:
                print(f"  - {pkg}: {error}")
        
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
