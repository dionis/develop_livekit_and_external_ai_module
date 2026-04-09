
import sys
import subprocess
import os

print("--- Python Info ---")
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

print("\n--- NVIDIA-SMI ---")
try:
    subprocess.run(["nvidia-smi"], check=True)
except Exception as e:
    print(f"Failed to run nvidia-smi: {e}")

print("\n--- PyTorch ---")
try:
    import torch
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA NOT Available for PyTorch.")
except ImportError:
    print("PyTorch not installed.")

print("\n--- ONNX Runtime ---")
try:
    import onnxruntime as ort
    print(f"ONNX Runtime Version: {ort.__version__}")
    print(f"Available Providers: {ort.get_available_providers()}")
    
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        print("WARNING: CUDAExecutionProvider not found in ONNX Runtime.")
        print("This usually means 'onnxruntime-gpu' is not installed or CUDA libs are missing.")
        print("Try running: pip list | grep onnxruntime")
except ImportError:
    print("onnxruntime not installed.")


print("\n--- ONNX Runtime Dependencies (CuDNN/CuBLAS) ---")
try:
    import nvidia.cudnn
    import nvidia.cublas
    import os
    
    try:
        if hasattr(nvidia.cudnn, '__file__') and nvidia.cudnn.__file__:
            cudnn_path = os.path.dirname(nvidia.cudnn.__file__)
        else:
            cudnn_path = nvidia.cudnn.__path__[0]
            
        if hasattr(nvidia.cublas, '__file__') and nvidia.cublas.__file__:
            cublas_path = os.path.dirname(nvidia.cublas.__file__)
        else:
            cublas_path = nvidia.cublas.__path__[0]
    except Exception as e:
        print(f"Error determining paths: {e}")
        cudnn_path = ""
        cublas_path = ""
    
    cudnn_lib = os.path.join(cudnn_path, 'lib')
    cublas_lib = os.path.join(cublas_path, 'lib')
    
    print(f"nvidia-cudnn found at: {cudnn_path}")
    print(f"nvidia-cublas found at: {cublas_path}")
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    missing_paths = []
    if cudnn_lib not in ld_library_path:
        missing_paths.append(cudnn_lib)
    if cublas_lib not in ld_library_path:
        missing_paths.append(cublas_lib)
        
    if missing_paths:
        print("\nCRITICAL WARNING: These NVIDIA library paths are NOT in your LD_LIBRARY_PATH:")
        for p in missing_paths:
            print(f"  - {p}")
        
        print("\nSUGGESTED FIX: Run this command in your terminal before running python:")
        print(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{':'.join(missing_paths)}")
        
except ImportError as e:
    print(f"Warning: Could not import nvidia.cudnn or nvidia.cublas: {e}")
    print("Please ensure you installed: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12")

print("\n--- Environment Variables (CUDA related) ---")
for k, v in os.environ.items():
    if "CUDA" in k or "NVIDIA" in k or "LD_LIBRARY_PATH" in k:
        print(f"{k}: {v}")
