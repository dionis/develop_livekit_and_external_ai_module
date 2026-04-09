#!/bin/bash
set -e

echo "🚀 Starting FRESH INSTALLATION on Python 3.10..."

# 1. Clean previous attempts
echo "Cleaning environment..."
rm -rf .venv uv.lock
uv cache clean

# 2. Setup Virtual Environment with Python 3.10
echo "Creating Virtual Environment with Python 3.10..."
uv venv --python 3.10
source .venv/bin/activate

# 3. Install core dependencies with NVIDIA index
# Using the same strategy as fix_tensorrt but for everything
echo "Installing build dependencies (pip, setuptools, wheel)..."
uv pip install pip setuptools wheel

echo "Installing dependencies (including TensorRT 8.6.1)..."

# Note: We use --index-strategy unsafe-best-match to favor nvidia index for trt/cudnn
uv pip install \
    "tensorrt==8.6.1" \
    "tensorrt_libs==8.6.1" \
    "tensorrt_bindings==8.6.1" \
    "nvidia-cudnn-cu12" \
    "nvidia-cublas-cu12" \
    "nvidia-cuda-runtime-cu12" \
    "numpy==1.26.4" \
    --index-url https://pypi.nvidia.com \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --no-build-isolation

# 4. Install other project dependencies
echo "Syncing all project dependencies..."
uv sync

# 5. AUTOMATIC PATH INJECTION
# We add the LD_LIBRARY_PATH to the .venv/bin/activate script so it's always set
echo "Injecting NVIDIA library paths into venv activation script..."

ACTIVATE_SCRIPT=".venv/bin/activate"
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

CUDNN_PATH="$SITE_PACKAGES/nvidia/cudnn/lib"
CUBLAS_PATH="$SITE_PACKAGES/nvidia/cublas/lib"
CUDA_RUNTIME_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/lib"
TRT_LIBS_PATH="$SITE_PACKAGES/tensorrt_libs"

# Create a block to append
PATH_BLOCK="
# NVIDIA/TensorRT Library Paths (Added by Fresh Install Script)
export LD_LIBRARY_PATH=\"$CUDNN_PATH:$CUBLAS_PATH:$CUDA_RUNTIME_PATH:$TRT_LIBS_PATH:\$LD_LIBRARY_PATH\"
"

if ! grep -q "NVIDIA/TensorRT Library Paths" "$ACTIVATE_SCRIPT"; then
    echo "$PATH_BLOCK" >> "$ACTIVATE_SCRIPT"
    echo "✅ Paths injected into $ACTIVATE_SCRIPT"
fi

# 6. Final Verification
echo "🔍 Verifying installation..."
export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:$CUDA_RUNTIME_PATH:$TRT_LIBS_PATH:$LD_LIBRARY_PATH"

python << EOF
try:
    import tensorrt as trt
    print(f"✅ TensorRT loaded successfully!")
    print(f"   Logger found: {hasattr(trt, 'Logger')}")
    import numpy
    print(f"✅ NumPy version: {numpy.__version__}")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    exit(1)
EOF

echo "=========================================="
echo "✅ FRESH INSTALLATION COMPLETE!"
echo "=========================================="
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Run benchmark: make test"
echo "=========================================="
