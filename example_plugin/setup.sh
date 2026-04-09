#!/bin/bash

# Automated installation script for Video TTS Streamer
# Usage: ./setup.sh

set -e


# Ensure we are not using an inherited virtual environment
unset VIRTUAL_ENV

echo "🚀 Installing Video TTS Streamer..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Verify Python
echo "Verifying Python..."

# List of candidates to check for compatible Python versions (3.10 or 3.11)
PYTHON_CMD=""
PYTHON_VERSION=""

# 1. Check system commands
CANDIDATES=("python3.10" "python3.11" "python3" "python")

# 2. Add uv-managed pythons if uv is available
if command -v uv &> /dev/null; then
    UV_310=$(uv python find 3.10 2>/dev/null || true)
    UV_311=$(uv python find 3.11 2>/dev/null || true)
    
    # If neither is found via uv, install 3.10 automatically
    if [ -z "$UV_310" ] && [ -z "$UV_311" ]; then
        print_info "Compatible Python not found via uv. Installing Python 3.10 via uv..."
        uv python install 3.10
        UV_310=$(uv python find 3.10 2>/dev/null || true)
    fi

    [ -n "$UV_310" ] && CANDIDATES=("$UV_310" "${CANDIDATES[@]}")
    [ -n "$UV_311" ] && CANDIDATES=("$UV_311" "${CANDIDATES[@]}")
fi

for cmd in "${CANDIDATES[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        VER_STR=$("$cmd" --version 2>&1)
        if [[ "$VER_STR" == *"Python 3.10"* ]] || [[ "$VER_STR" == *"Python 3.11"* ]]; then
            PYTHON_CMD="$cmd"
            PYTHON_VERSION="$VER_STR"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    CURRENT_PY=$(python3 --version 2>&1 || python --version 2>&1)
    if [[ "$CURRENT_PY" == *"Python 3.12"* ]]; then
        print_error "Python 3.12 detected. This version is NOT compatible with cuda-python 12.1.0 and TensorRT 8.6.1."
        print_info "Please use Python 3.10 or 3.11."
        print_info "You can create a compatible environment with Conda:"
        print_info "  conda create -n ditto python=3.10 && conda activate ditto"
        print_info "Or use uv to install it: uv python install 3.10"
        exit 1
    else
        print_error "Compatible Python (3.10 or 3.11) not found."
        print_info "Detected: $CURRENT_PY"
        print_info "Please install Python 3.10 or 3.11."
        exit 1
    fi
fi

print_success "Compatible Python found: $PYTHON_VERSION"

# Verify and Install FFmpeg with development libraries
echo "Verifying FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    print_error "FFmpeg is not installed"
    print_info "Installing FFmpeg and development libraries..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Ubuntu/Debian)
        sudo apt-get update
        sudo apt-get install -y \
            ffmpeg \
            libavformat-dev \
            libavcodec-dev \
            libavdevice-dev \
            libavutil-dev \
            libavfilter-dev \
            libswscale-dev \
            libswresample-dev \
            libfdk-aac-dev \
            libmp3lame-dev \
            librtmp-dev \
            libx264-dev \
            pkg-config
        print_success "FFmpeg and development libraries installed"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg pkg-config
            print_success "FFmpeg installed via Homebrew"
        else
            print_error "Homebrew not found. Please install from https://brew.sh"
            exit 1
        fi
    else
        print_error "Unsupported OS. Please install FFmpeg manually:"
        print_info "  Windows: https://ffmpeg.org/download.html"
        print_info "  Then install PyAV with: pip install av --only-binary av"
        exit 1
    fi
else
    print_success "FFmpeg found"
    
    # Verify FFmpeg development libraries are installed (Linux only)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! pkg-config --exists libavformat; then
            print_info "FFmpeg development libraries not found. Installing..."
            sudo apt-get update
            sudo apt-get install -y \
                libavformat-dev \
                libavcodec-dev \
                libavdevice-dev \
                libavutil-dev \
                libavfilter-dev \
                libswscale-dev \
                libswresample-dev \
                pkg-config
            print_success "FFmpeg development libraries installed"
        else
            print_success "FFmpeg development libraries found"
        fi
    fi
fi

# Install cuDNN and CUDA libraries (Linux only - required for TensorRT)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo ""
    echo "Verifying cuDNN and CUDA libraries for TensorRT..."
    
    if [ ! -f "/usr/lib/x86_64-linux-gnu/libcudnn.so.8" ]; then
        print_info "cuDNN 8 not found. Installing CUDA and cuDNN libraries..."
        
        # Add NVIDIA CUDA repository
        print_info "Adding NVIDIA CUDA repository..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        
        # Install cuDNN 8 (required by TensorRT 8.6.1)
        print_info "Installing cuDNN 8..."
        sudo apt-get install -y libcudnn8 libcudnn8-dev
        
        # Install CUDA 12.1 runtime libraries
        print_info "Installing CUDA 12.1 runtime libraries..."
        sudo apt-get install -y \
            cuda-cudart-12-1 \
            cuda-libraries-12-1 \
            libcublas-12-1
        
        print_success "cuDNN and CUDA libraries installed"
        
        # Configure LD_LIBRARY_PATH
        CUDA_LIB_PATH="/usr/local/cuda-12.1/lib64"
        CUDNN_LIB_PATH="/usr/lib/x86_64-linux-gnu"
        
        # Add to .bashrc if not already there
        if ! grep -q "CUDA library paths for TensorRT" ~/.bashrc; then
            echo "" >> ~/.bashrc
            echo "# CUDA library paths for TensorRT" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CUDA_LIB_PATH:$CUDNN_LIB_PATH" >> ~/.bashrc
            print_success "Added CUDA paths to ~/.bashrc"
        fi
        
        # Export for current session
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH:$CUDNN_LIB_PATH
    else
        print_success "cuDNN 8 found"
        
        # Ensure LD_LIBRARY_PATH is set
        CUDA_LIB_PATH="/usr/local/cuda-12.1/lib64"
        CUDNN_LIB_PATH="/usr/lib/x86_64-linux-gnu"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH:$CUDNN_LIB_PATH
    fi
fi

# Verify Git
echo "Verifying Git..."
if ! command -v git &> /dev/null; then
    print_error "Git is not installed"
    exit 1
fi
print_success "Git found"

# Install uv if not installed
echo "Verifying uv..."
if ! command -v uv &> /dev/null; then
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    print_success "uv installed"
else
    print_success "uv is already installed"
fi

# Configure HuggingFace authentication
echo ""
echo "Configuring HuggingFace authentication..."
if [ -f "scripts/configure_huggingface.sh" ]; then
    bash scripts/configure_huggingface.sh
else
    print_info "HuggingFace configuration script not found, skipping..."
fi


# Create directories
echo "Creating directories..."
mkdir -p uploads outputs chunks static modules external_models
print_success "Directories created"

# Install project dependencies
echo "Installing project dependencies..."
print_info "Using $PYTHON_CMD for virtual environment..."

# Ensure uv is installed and in PATH
if ! command -v uv &> /dev/null; then
    print_info "Installing uv..."
    pip install uv
    # Reload PATH to ensure uv is available
    export PATH="$HOME/.local/bin:$PATH"
fi

# Use the detected Python version
DETECTED_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_info "Syncing dependencies with Python $DETECTED_VER using $PYTHON_CMD..."
UV_HTTP_TIMEOUT=120 uv sync --python "$PYTHON_CMD"

echo "ditto-talkinghead Install special dependencies"
echo "Installing PyTorch 2.5.1 with CUDA 12.1 support..."
# Use --index-strategy unsafe-best-match to allow UV to consider packages from all indexes
# This is needed because nvidia-cudnn-cu12 appears on both NVIDIA and PyTorch indexes
UV_HTTP_TIMEOUT=120 uv pip install --python .venv torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match
UV_HTTP_TIMEOUT=120 uv pip install --python .venv mediapipe
UV_HTTP_TIMEOUT=120 uv pip install --python .venv einops
UV_HTTP_TIMEOUT=120 uv pip install --python .venv "onnxruntime-gpu<1.24"


UV_HTTP_TIMEOUT=120 uv pip install --python .venv \
    librosa \
    tqdm \
    filetype \
    imageio \
    scikit-image \
    cython \
    cuda-python==12.1.0 \
    imageio-ffmpeg \
    colored \
    polygraphy \
    "numpy>=1.23.0,<2.0.0"  # Compatible con TensorFlow 2.15

# Special installation for TensorRT (requires build insulation off due to internal pip calls)
# We also MUST ensure pip, setuptools and wheel are installed because TensorRT calls 'python -m pip'
# and requires setuptools/wheel for its legacy build process.
echo "Installing TensorRT 8.6.1 build dependencies..."
UV_HTTP_TIMEOUT=120 uv pip install --python .venv pip setuptools wheel

# First, install tensorrt-libs and tensorrt-bindings from NVIDIA index
# These are dependencies that must be installed before the main tensorrt package
echo "Installing TensorRT dependencies (libs and bindings)..."
UV_HTTP_TIMEOUT=120 uv pip install --python .venv tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 \
    --index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match

# Now install the main tensorrt package with --no-build-isolation
# This allows it to use the already-installed dependencies
echo "Installing main TensorRT package..."
UV_HTTP_TIMEOUT=120 uv pip install --python .venv tensorrt==8.6.1 \
    --index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --no-build-isolation || \
    print_warning "TensorRT installation failed. You may need to install it manually."

# Verify TensorRT installation
echo ""
echo "Verifying TensorRT installation..."
uv run python -c "
import tensorrt as trt
print(f'✓ TensorRT version: {trt.__version__}')
if hasattr(trt, 'Logger'):
    print('✓ TensorRT Logger available')
else:
    print('⚠️  TensorRT Logger not found')
" && print_success "TensorRT verified" || print_warning "TensorRT verification failed"

# Download LiveKit Plugin Models (VAD, Turn Detector)
echo ""
echo "------------------------------------------------"
echo "LiveKit Plugin Models Installation"
echo "------------------------------------------------"
print_info "Downloading models for Turn Detector and VAD..."
uv run python scripts/download_models.py || print_warning "Some models could not be downloaded automatically."

print_success "Dependencies and plugin models installed"

# Prompt for SOTA models installation
echo ""
echo "------------------------------------------------"
echo "SOTA Talking Head Models Installation"
echo "------------------------------------------------"
read -p "Would you like to install the Ditto model now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "scripts/install_models/install_ditto.sh" ]; then
        bash scripts/install_models/install_ditto.sh --python "$PYTHON_CMD" --no-conda
    else
        print_error "Model installation script (scripts/install_models/install_ditto.sh) not found"
    fi
else
    print_info "Skipping model installation. You can run it later with: bash scripts/install_models/install_ditto.sh --python $PYTHON_CMD --no-conda"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file from examples/.env.example..."
    if [ -f "examples/.env.example" ]; then
        cp examples/.env.example .env
        print_success ".env file created"
    else
        print_warning "examples/.env.example not found, could not create .env"
    fi
    print_info "⚠️  IMPORTANT: Edit .env with your credentials:"
    print_info "   - ELEVENLABS_API_KEY"
    print_info "   - ELEVENLABS_VOICE_ID (optional)"
    print_info "   - HUGGINGFACE_TOKEN (will be configured in next step)"
else
    print_success ".env file already exists"
fi

echo ""
print_success "Installation completed!"
echo ""
print_info "Next steps:"
echo "  1. Edit the .env file with your ElevenLabs API key"
echo "  2. Reload your shell: source ~/.bashrc"
echo "  3. Run: uv run python main.py"
echo "  4. Open http://localhost:8000 in your browser"
echo ""
print_info "To get an ElevenLabs API key:"
echo "  https://elevenlabs.io"
echo ""
print_warning "IMPORTANT: Run 'source ~/.bashrc' to load CUDA library paths!"
echo ""
