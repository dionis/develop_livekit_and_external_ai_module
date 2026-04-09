#!/bin/bash
# Install cuDNN and CUDA libraries for TensorRT on Ubuntu
# This fixes: libcudnn.so.8: cannot open shared object file

set -e

echo "🔧 Installing cuDNN and CUDA libraries for TensorRT..."
echo ""

# Check if running on Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    echo "❌ This script is for Ubuntu/Debian systems only"
    exit 1
fi

echo "Step 1: Adding NVIDIA CUDA repository..."
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

echo ""
echo "Step 2: Installing cuDNN 8..."
# Install cuDNN 8 (required by TensorRT 8.6.1)
sudo apt-get install -y libcudnn8 libcudnn8-dev

echo ""
echo "Step 3: Installing CUDA runtime libraries..."
# Install CUDA 12.1 runtime libraries
sudo apt-get install -y \
    cuda-cudart-12-1 \
    cuda-libraries-12-1 \
    libcublas-12-1

echo ""
echo "Step 4: Configuring library paths..."
# Add CUDA libraries to LD_LIBRARY_PATH
CUDA_LIB_PATH="/usr/local/cuda-12.1/lib64"
CUDNN_LIB_PATH="/usr/lib/x86_64-linux-gnu"

# Add to current session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH:$CUDNN_LIB_PATH

# Add to .bashrc for persistence
if ! grep -q "CUDA library paths" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA library paths for TensorRT" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CUDA_LIB_PATH:$CUDNN_LIB_PATH" >> ~/.bashrc
    echo "✓ Added CUDA paths to ~/.bashrc"
fi

echo ""
echo "Step 5: Verifying cuDNN installation..."
if [ -f "/usr/lib/x86_64-linux-gnu/libcudnn.so.8" ]; then
    echo "✓ libcudnn.so.8 found"
    ls -lh /usr/lib/x86_64-linux-gnu/libcudnn.so.8
else
    echo "⚠️  libcudnn.so.8 not found in expected location"
    echo "Searching for cuDNN libraries..."
    find /usr -name "libcudnn.so*" 2>/dev/null || echo "No cuDNN libraries found"
fi

echo ""
echo "Step 6: Verifying TensorRT can load..."
cd /home/ubuntu/video-tts-streamer
source .venv/bin/activate

python -c "
import tensorrt as trt
print(f'✓ TensorRT version: {trt.__version__}')
print(f'✓ TensorRT loaded successfully!')
if hasattr(trt, 'Logger'):
    logger = trt.Logger(trt.Logger.WARNING)
    print(f'✓ TensorRT Logger created successfully!')
" && echo "✅ TensorRT is working!" || echo "❌ TensorRT still has issues"

echo ""
echo "Done!"
echo ""
echo "⚠️  IMPORTANT: Reload your shell or run:"
echo "   source ~/.bashrc"
echo "   export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CUDA_LIB_PATH:$CUDNN_LIB_PATH"
