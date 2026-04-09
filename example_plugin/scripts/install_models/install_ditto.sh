#!/bin/bash
# Installation script for Ditto TalkingHead Model
# Source: https://github.com/antgroup/ditto-talkinghead

set -e

echo "=========================================="
echo "Installing Ditto TalkingHead Model"
echo "=========================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/external_models"
DITTO_DIR="$MODELS_DIR/ditto-talkinghead"

# Create directories
mkdir -p "$MODELS_DIR"

# Check if already installed
if [ -d "$DITTO_DIR" ]; then
    echo "Ditto already installed at $DITTO_DIR"
    read -p "Reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    rm -rf "$DITTO_DIR"
fi

# Clone repository
echo "Cloning Ditto repository..."
cd "$MODELS_DIR"
git clone https://github.com/antgroup/ditto-talkinghead.git
cd ditto-talkinghead

# Parse arguments
PYTHON_CMD="python3"
USE_CONDA=true

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --python) PYTHON_CMD="$2"; shift ;;
        --no-conda) USE_CONDA=false ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate Python version
echo ""
echo "Validating Python version using $PYTHON_CMD..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" == "3.10" ]] || [[ "$PYTHON_VERSION" == "3.11" ]]; then
    echo "✓ Compatible Python version found: $PYTHON_VERSION"
else
    echo "⚠️  Warning: Ditto is tested with Python 3.10/3.11, but found Python $PYTHON_VERSION"
    echo "   Continuing anyway, but you may encounter issues."
fi

# Check if conda is available and user wants to use it
if [ "$USE_CONDA" = true ] && command -v conda &> /dev/null; then
    echo "Installing with Conda..."
    conda env create -f environment.yaml || echo "⚠️  Conda environment creation failed or skipped."
    echo "✓ Conda setup checked"
else
    echo "Skipping Conda environment creation. Using environment from $PYTHON_CMD..."
    
    # Dependencies are now managed via pyproject.toml and main setup.sh
    echo "Ensuring Ditto dependencies are present in the current environment..."
    if command -v uv &> /dev/null; then
        # Use uv to install requirements.txt if it exists in the repo
        if [ -f "requirements.txt" ]; then
            uv pip install --python "$PYTHON_CMD" -r requirements.txt
        fi
    else
        if [ -f "requirements.txt" ]; then
            $PYTHON_CMD -m pip install -r requirements.txt
        fi
    fi
    
    echo "⚠️  Note: You may need to install ffmpeg separately"
    echo "  Visit: https://www.ffmpeg.org/download.html"
fi

# Download checkpoints from HuggingFace
echo ""
echo "Downloading model checkpoints from HuggingFace..."
# Download checkpoints from HuggingFace
echo ""
echo "Downloading model checkpoints from HuggingFace..."
echo "Using Python script (huggingface_hub)..."

# Ensure we use the validated python environment
RUN_CMD="$PYTHON_CMD"

# Ensure huggingface_hub is installed
$RUN_CMD -m pip install huggingface_hub

$RUN_CMD "$SCRIPT_DIR/download_ditto.py" "$DITTO_DIR/checkpoints"

if [ $? -ne 0 ]; then
   echo "❌ Download failed."
   exit 1
fi

echo "✓ Checkpoints downloaded"

# Validate checkpoint structure
echo ""
echo "Validating checkpoint structure..."
REQUIRED_DIRS=("ditto_cfg" "ditto_onnx" "ditto_trt_Ampere_Plus")
MISSING_DIRS=()

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$DITTO_DIR/checkpoints/$dir" ]; then
        MISSING_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DIRS[@]} -ne 0 ]; then
    echo "⚠️  Warning: Missing checkpoint directories: ${MISSING_DIRS[*]}"
else
    echo "✓ All checkpoint directories present"
fi

# Offer to convert ONNX to TensorRT for custom GPU
if [ -d "$DITTO_DIR/checkpoints/ditto_onnx" ] && [ ! -d "$DITTO_DIR/checkpoints/ditto_trt_custom" ]; then
    echo ""
    echo "Note: Pre-built TensorRT engines are for Ampere+ GPUs (RTX 30xx/40xx, A100, etc.)"
    
    # First, verify that TensorRT is working
    echo "Verifying TensorRT installation..."
    if $RUN_CMD -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>/dev/null; then
        echo "✓ TensorRT is working"
        read -p "Convert ONNX to TensorRT for your GPU? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Converting ONNX models to TensorRT..."
            
            $RUN_CMD "$DITTO_DIR/scripts/cvt_onnx_to_trt.py" \
                --onnx_dir "$DITTO_DIR/checkpoints/ditto_onnx" \
                --trt_dir "$DITTO_DIR/checkpoints/ditto_trt_custom"
            
            if [ $? -eq 0 ]; then
                echo "✓ TensorRT engines created for your GPU"
            else
                echo "⚠️  TensorRT conversion failed. You can run it manually later."
            fi
        fi
    else
        echo "⚠️  TensorRT is not working yet (this is normal during initial setup)"
        echo "⚠️  Skipping TensorRT conversion for now."
    fi
fi

# Fix NumPy 2.0 compatibility
echo ""
echo "Fixing NumPy 2.0 compatibility..."
echo "  → Ditto was written for NumPy 1.x, patching for NumPy 2.0+"

# Simple inline patch - replace np.atan2 with np.arctan2
if find "$DITTO_DIR" -name "*.py" -type f -exec grep -l "np\.atan2" {} + > /dev/null 2>&1; then
    echo "  → Found files with np.atan2, applying patches..."
    find "$DITTO_DIR" -name "*.py" -type f -exec sed -i 's/np\.atan2/np.arctan2/g' {} +
    echo "  ✓ Applied NumPy 2.0 compatibility patches"
else
    echo "  ✓ Already compatible with NumPy 2.0"
fi

# Configuration update skipped (handled by plugin)
echo ""
echo "Updating model configuration..."
echo "  ✓ Ditto path: $DITTO_DIR"

echo ""
echo "=========================================="
echo "Ditto installation complete!"
echo "=========================================="
echo "Location: $DITTO_DIR"
echo ""
echo "Next steps:"
echo "1. If using conda: conda activate ditto"
echo "2. Test inference:"
echo "   cd $DITTO_DIR"
echo "   python inference.py \\"
echo "     --data_root ./checkpoints/ditto_trt_Ampere_Plus \\"
echo "     --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \\"
echo "     --audio_path ./example/audio.wav \\"
echo "     --source_path ./example/image.png \\"
echo "     --output_path ./tmp/result.mp4"
echo ""
echo "Note: If your GPU doesn't support Ampere_Plus, convert ONNX to TensorRT:"
echo "   python scripts/cvt_onnx_to_trt.py \\"
echo "     --onnx_dir ./checkpoints/ditto_onnx \\"
echo "     --trt_dir ./checkpoints/ditto_trt_custom"
echo "=========================================="
