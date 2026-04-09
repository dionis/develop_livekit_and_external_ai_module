#!/usr/bin/env bash
# =============================================================================
#  ARTalk LiveKit Plugin — Automated Installation Script
#  Target: Linux with Miniconda/Anaconda/uv + CUDA GPU
#  Installs the livekit-plugins-artalk wrapper along with internal ARTalk
# =============================================================================

set -euo pipefail

# ── Terminal colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; \
                echo -e "${BOLD}${CYAN}  $*${NC}"; \
                echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"; }

# ── Configuration ────────────────────────────────────────────────────────────
ARTALK_REPO="https://github.com/xg-chu/ARTalk.git"
DIFF_GAUSS_REPO="https://github.com/xg-chu/diff-gaussian-rasterization.git"
INSTALL_DIR="${PWD}"

# =============================================================================
# STEP 0 — System Dependencies
# =============================================================================
log_section "STEP 0 — System Dependencies (FFmpeg)"

if ! command -v ffmpeg &> /dev/null; then
    log_info "FFmpeg not found. Required for torchaudio/torio."
    if command -v apt-get &> /dev/null; then
        log_info "Attempting to install ffmpeg via apt-get..."
        sudo apt-get update && sudo apt-get install -y ffmpeg || log_warn "Failed to install ffmpeg automatically. Please install it manually: sudo apt-get install ffmpeg"
    else
        log_warn "Package manager 'apt-get' not found. Please install ffmpeg manually for your distribution."
    fi
else
    log_ok "FFmpeg is already installed."
fi

# =============================================================================
# ENVIRONMENT SELECTION
# =============================================================================
log_section "Platform Selection"
echo -e "${YELLOW}Please select your target environment:${NC}"
echo "  1) Standard Linux (Local / VPS / Cloud VM)"
echo "  2) Lightning.ai Studio (Installs into 'cloudspace')"
echo ""
read -p "Enter choice [1-2]: " ENV_CHOICE

# =============================================================================
# INSTALLATION LOGIC - STANDARD LINUX
# =============================================================================
install_standard() {
    log_section "STEP 1 (Standard) — Python Verification"

    PYTHON_CMD=""
    CANDIDATES=("python3.10" "python3.11" "python3" "python")

    if command -v uv &> /dev/null; then
        UV_310=$(uv python find 3.10 2>/dev/null || true)
        
        if [ -z "$UV_310" ]; then
            log_info "Compatible Python not found via uv. Installing Python 3.10 via uv..."
            uv python install 3.10
            UV_310=$(uv python find 3.10 2>/dev/null || true)
        fi
        [ -n "$UV_310" ] && CANDIDATES=("$UV_310" "${CANDIDATES[@]}")
    fi

    for cmd in "${CANDIDATES[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            VER_STR=$("$cmd" --version 2>&1)
            if [[ "$VER_STR" == *"Python 3.10"* ]] || [[ "$VER_STR" == *"Python 3.11"* ]]; then
                PYTHON_CMD="$cmd"
                break
            fi
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        log_error "Requires Python 3.10 or 3.11 for Standard install. Found neither."
        exit 1
    fi
    log_ok "Using Python: $($PYTHON_CMD --version)"

    log_section "STEP 2 (Standard) — Syncing plugin dependencies in virtual environment"

    if ! command -v uv &> /dev/null; then
        log_info "Installing uv for faster dependency management."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi

    log_info "Syncing local LiveKit Plugin workspace into .venv"
    uv sync --python "$PYTHON_CMD"
    log_ok "Plugin core dependencies installed in .venv."

    log_section "STEP 3 (Standard) — Torch GPU Setup"

    # ARTalk relies on CUDA 12.1 for the diff-gaussian-rasterization package
    log_info "Installing PyTorch 2.4.1 + torchaudio 2.4.1 via cu121..."
    uv pip install --python .venv torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --index-strategy unsafe-best-match
    log_ok "PyTorch tools installed."

    log_section "STEP 4 (Standard) — Binding ARTalk core"

    mkdir -p "${INSTALL_DIR}/external_models"
    if [ -d "${INSTALL_DIR}/external_models/ARTalk" ]; then
        log_warn "Directory 'external_models/ARTalk' already exists. Skipping clone."
    else
        log_info "Cloning ARTalk with submodules..."
        git clone --recurse-submodules "${ARTALK_REPO}" "${INSTALL_DIR}/external_models/ARTalk"
        log_ok "ARTalk cloned successfully."
    fi

    ARTALK_PATH="${INSTALL_DIR}/external_models/ARTalk"

    # Install remaining ARTalk specific requirements inside the LiveKit plugin venv
    log_info "Installing ARTalk specific PIP requirements in .venv..."
    uv pip install --python .venv \
        trimesh \
        gdown \
        scipy \
        tqdm \
        opencv-python \
        soundfile \
        librosa

    log_section "STEP 5 (Standard) — diff-gaussian-rasterization Compilation"

    DIFF_GAUSS_DIR="${ARTALK_PATH}/diff-gaussian-rasterization"

    if [ -d "${DIFF_GAUSS_DIR}" ]; then
        log_warn "Directory 'diff-gaussian-rasterization' already exists. Skipping clone."
    else
        git clone --recurse-submodules "${DIFF_GAUSS_REPO}" "${DIFF_GAUSS_DIR}"
    fi

    RAST_HEADER="${DIFF_GAUSS_DIR}/cuda_rasterizer/rasterizer_impl.h"
    # Fixing cstdint issues for nvcc on Ubuntu 20/22
    if [ -f "${RAST_HEADER}" ]; then
        if grep -q '#pragma once' "${RAST_HEADER}"; then
            if ! grep -q '#include <cstdint>' "${RAST_HEADER}"; then
                sed -i '/#pragma once/a #include <cstdint>' "${RAST_HEADER}"
                log_ok "Fix applied successfully to rasterizer_impl.h"
            fi
        fi
    fi

    log_info "Compiling and installing diff-gaussian-rasterization (this will take a few minutes)..."
    cd "${DIFF_GAUSS_DIR}"
    # Disabling build isolation is key for PyTorch extension builds
    uv pip install --python "${INSTALL_DIR}/.venv" --no-build-isolation --force-reinstall .
    log_ok "CUDA Extension installed in .venv."
    cd "${INSTALL_DIR}"

    prepare_model_resources
    apply_watermark_patch

    # Create .env file for LiveKit Agents
    if [ ! -f ".env" ]; then
        echo "ARTALK_PATH=${ARTALK_PATH}" > .env
        echo "ARTALK_MODEL_STRATEGY=from_scratch" >> .env
        echo "# ARTALK_MODEL_STRATEGY options: from_scratch | example_models" >> .env
        echo "LIVEKIT_URL=" >> .env
        echo "LIVEKIT_API_KEY=" >> .env
        echo "LIVEKIT_API_SECRET=" >> .env
        echo "OPENAI_API_KEY=" >> .env
        echo "CARTESIA_API_KEY=" >> .env
        log_ok ".env template generated."
    fi

    log_section "✅  STANDARD INSTALLATION COMPLETE"
    echo -e "${GREEN}"
    echo "  Plugin mapped fully to : ${ARTALK_PATH}"
    echo "  Environment configured at: .venv"
    echo "  LiveKit Echo Example Test:"
    echo "    source .venv/bin/activate"
    echo "    python -m examples.basic_agent start"
    echo -e "${NC}"
}

# =============================================================================
# INSTALLATION LOGIC - LIGHTNING.AI STUDIO
# =============================================================================
install_lightning() {
    log_section "STEP 1 (Lightning) — Verify Lightning.ai environment"
    ACTIVE_PYTHON=$(which python)
    log_info "Active Python : ${ACTIVE_PYTHON}"
    
    if ! conda env list 2>&1 | grep -q "Conda environment management is not allowed"; then
        log_warn "Standard conda environment detected, but proceeding with Lightning.ai system-level installation path."
    else
        log_ok "Lightning.ai Studio confirmed."
    fi

    log_section "STEP 2 (Lightning) — Install Plugin Dependencies (System level)"
    # Install uv system-wide if not present
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi

    log_info "Installing local LiveKit Plugin workspace into system environment"
    uv pip install --system -e .
    log_ok "Base plugin packages installed."

    log_section "STEP 3 (Lightning) — Torch GPU Setup"
    log_info "Installing PyTorch 2.4.1 + torchaudio 2.4.1 via cu121..."
    uv pip install --system torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --index-strategy unsafe-best-match
    log_ok "PyTorch tools installed."


    log_section "STEP 4 (Lightning) — Binding ARTalk core"
    mkdir -p "${INSTALL_DIR}/external_models"
    if [ -d "${INSTALL_DIR}/external_models/ARTalk" ]; then
        log_warn "Directory 'external_models/ARTalk' already exists. Skipping clone."
    else
        log_info "Cloning ARTalk with submodules..."
        git clone --recurse-submodules "${ARTALK_REPO}" "${INSTALL_DIR}/external_models/ARTalk"
        log_ok "ARTalk cloned successfully."
    fi

    ARTALK_PATH="${INSTALL_DIR}/external_models/ARTalk"
    
    log_info "Installing ARTalk and GAGAvatar_track pip requirements..."
    # Core ARTalk deps
    uv pip install --system \
        trimesh \
        gdown \
        tqdm \
        soundfile \
        librosa \
        opencv-python-headless \
        imageio==2.35.1 \
        imageio-ffmpeg \
        lmdb==1.5.1
    log_ok "Core ARTalk packages installed."

    log_info "Installing GAGAvatar_track deps (onnx2torch, face-alignment, numba, scikit-image, transformers)..."
    # numpy must be >=2.0 to match GAGAvatar_track's own env, but <=2.0.1 for numba 0.60 compat
    uv pip install --system "numpy==2.0.1"
    uv pip install --system \
        "onnx==1.16.2" \
        "onnx2torch==1.5.15" \
        "llvmlite==0.43.0" \
        "numba==0.60.0" \
        "face-alignment==1.4.1" \
        "scikit-image==0.24.0" \
        "scipy==1.14.1" \
        "transformers==4.45.1" \
        "tokenizers==0.20.0" \
        "safetensors==0.4.5"
    log_ok "GAGAvatar_track packages installed."

    log_info "Cloning GAGAvatar_track (face tracking engine for custom avatars)..."
    GAGTRACK_DIR="${INSTALL_DIR}/external_models/GAGAvatar_track"
    if [ -d "${GAGTRACK_DIR}" ]; then
        log_warn "Directory 'external_models/GAGAvatar_track' already exists. Skipping clone."
    else
        git clone "https://github.com/xg-chu/GAGAvatar_track.git" "${GAGTRACK_DIR}"
        log_ok "GAGAvatar_track cloned."
    fi

    log_section "STEP 5 (Lightning) — diff-gaussian-rasterization Compilation"
    DIFF_GAUSS_DIR="${ARTALK_PATH}/diff-gaussian-rasterization"

    if [ -d "${DIFF_GAUSS_DIR}" ]; then
        log_warn "Directory 'diff-gaussian-rasterization' already exists. Skipping clone."
    else
        git clone --recurse-submodules "${DIFF_GAUSS_REPO}" "${DIFF_GAUSS_DIR}"
    fi

    RAST_HEADER="${DIFF_GAUSS_DIR}/cuda_rasterizer/rasterizer_impl.h"
    if [ -f "${RAST_HEADER}" ]; then
        if grep -q '#pragma once' "${RAST_HEADER}"; then
            if ! grep -q '#include <cstdint>' "${RAST_HEADER}"; then
                sed -i '/#pragma once/a #include <cstdint>' "${RAST_HEADER}"
                log_ok "Fix applied successfully to rasterizer_impl.h"
            fi
        fi
    fi

    log_info "Compiling and installing diff-gaussian-rasterization (this will take a few minutes)..."
    cd "${DIFF_GAUSS_DIR}"
    # --no-build-isolation is critical in Lightning due to isolated environments lacking torch metadata
    uv pip install --system --no-build-isolation --force-reinstall .
    log_ok "CUDA Extension installed globally."
    cd "${INSTALL_DIR}"

    prepare_model_resources
    apply_watermark_patch

    # Create .env file for LiveKit Agents
    if [ ! -f ".env" ]; then
        echo "ARTALK_PATH=${ARTALK_PATH}" > .env
        echo "ARTALK_MODEL_STRATEGY=from_scratch" >> .env
        echo "# ARTALK_MODEL_STRATEGY options: from_scratch | example_models" >> .env
        echo "LIVEKIT_URL=" >> .env
        echo "LIVEKIT_API_KEY=" >> .env
        echo "LIVEKIT_API_SECRET=" >> .env
        echo "OPENAI_API_KEY=" >> .env
        echo "CARTESIA_API_KEY=" >> .env
        log_ok ".env template generated."
    fi

    log_section "✅  LIGHTNING.AI INSTALLATION COMPLETE"
    echo -e "${GREEN}"
    echo "  Plugin mapped fully to : ${ARTALK_PATH}"
    echo "  Environment          : cloudspace (System)"
    echo "  LiveKit Echo Example Test (no virtual env activation required):"
    echo "    python -m examples.basic_agent start"
    echo -e "${NC}"
}

prepare_model_resources() {
    log_section "Downloading ARTalk Model Assets"
    
    ARTALK_PATH="${INSTALL_DIR}/external_models/ARTalk"
    if [ -f "${ARTALK_PATH}/build_resources.sh" ]; then
        log_info "Gathering model weights via ARTalk's builder script..."
        cd "${ARTALK_PATH}"
        yes | bash "build_resources.sh"
        cd "${INSTALL_DIR}"
        log_ok "Models collected successfully."
    else
        log_warn "build_resources.sh not found inside ARTalk."
    fi
}

apply_watermark_patch() {
    log_section "Apply Watermark Monkey Patch (Option 2)"
    
    INFERENCE_PY="${INSTALL_DIR}/external_models/ARTalk/inference.py"
    if [ -f "${INFERENCE_PY}" ]; then
        if grep -q 'GAGAvatar.add_water_mark' "${INFERENCE_PY}"; then
            log_warn "Watermark monkey patch already applied. Skipping."
        else
            log_info "Applying monkey patch to remove watermark in inference.py..."
            sed -i '/engine = ARTAvatarInferEngine(load_gaga=True/a \    # Monkey patch to remove watermark\n    if hasattr(engine, "GAGAvatar"):\n        engine.GAGAvatar.add_water_mark = lambda image: image' "${INFERENCE_PY}"
            log_ok "Watermark monkey patch applied successfully."
        fi
    fi
}

# =============================================================================
# EXECUTE SELECTION
# =============================================================================
case $ENV_CHOICE in
    1)
        install_standard
        ;;
    2)
        install_lightning
        ;;
    *)
        log_error "Invalid choice: '$ENV_CHOICE'. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac
