#!/bin/bash
set -e

echo "🔧 Fixing TensorRT installation..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment. Activating .venv..."
    source .venv/bin/activate
fi

# 1. Uninstall existing packages
echo "Step 1: Uninstalling existing TensorRT packages..."
uv pip uninstall -y tensorrt tensorrt-libs tensorrt-bindings 2>/dev/null || true

# 2. Ensure pip is available
echo ""
echo "Step 2: Installing pip in virtual environment..."
uv pip install pip setuptools wheel

# 3. Install TensorRT dependencies first (libs and bindings)
echo ""
echo "Step 3: Installing TensorRT dependencies (libs and bindings)..."
uv pip install tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 \
    --index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match

# 4. Install main TensorRT package
echo ""
echo "Step 4: Installing main TensorRT package..."
uv pip install tensorrt==8.6.1 \
    --index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --no-build-isolation

# 5. Verify installation
echo ""
echo "Step 5: Verifying TensorRT installation..."
python -c "
import tensorrt as trt
print(f'✓ TensorRT version: {trt.__version__}')
print(f'✓ TensorRT bindings loaded successfully')
if hasattr(trt, 'Logger'):
    print('✓ Logger class available')
else:
    print('⚠️  Logger class not found')
    exit(1)
" && echo "✅ TensorRT installation successful!" || echo "❌ TensorRT verification failed"

echo ""
echo "Done!"
