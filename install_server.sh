#!/usr/bin/env bash
# =============================================================================
#  ARTalk Server — Installation Script
#  Script Version: 2026-03-25-v0.1.1
# =============================================================================
#  Run this script on the MACHINE WITH A CUDA GPU that will host the avatar
#  rendering engine. The client agent (brain) machine uses install_client.sh.
#
#  Two target environments are supported:
#    1) Standard Linux  — Miniconda / Anaconda / uv (local, VPS, cloud VM)
#    2) Lightning.ai    — installs into the restricted 'cloudspace' environment
#
#  Usage:
#    chmod +x install_server.sh
#    ./install_server.sh
#
#  Architecture overview:
#    GPU Machine  →  install_server.sh  →  artalk_server/ (FastAPI on :8000)
#    Brain Agent  →  install_client.sh  →  livekit.plugins.artalk (LiveKit agent)
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
GAGTRACK_REPO="https://github.com/xg-chu/GAGAvatar_track.git"
INSTALL_DIR="${PWD}"
LIGHTNING_MODE=0   # set to 1 inside install_server_lightning()

# =============================================================================
# SHARED PIP WRAPPERS
#   _pip_install   — calls the correct pip install for the active mode
#   _pip_uninstall — calls the correct pip uninstall for the active mode
# =============================================================================
_pip_install() {
    if [ "${LIGHTNING_MODE:-0}" -eq 1 ]; then
        "${ACTIVE_PYTHON}" -m pip install --no-cache-dir "$@"
    else
        uv pip install --python "${ACTIVE_PYTHON}" "$@"
    fi
}

_pip_uninstall() {
    if [ "${LIGHTNING_MODE:-0}" -eq 1 ]; then
        "${ACTIVE_PYTHON}" -m pip uninstall -y "$@"
    else
        uv pip uninstall --python "${ACTIVE_PYTHON}" "$@"
    fi
}


# =============================================================================
# STEP 0 — System Prerequisite: FFmpeg
# =============================================================================
check_ffmpeg() {
    log_section "STEP 0 — System Prerequisites"

    if ! command -v ffmpeg &> /dev/null; then
        log_info "FFmpeg not found. Required for torchaudio / torio."
        if command -v apt-get &> /dev/null; then
            log_info "Installing ffmpeg via apt-get..."
            sudo apt-get update && sudo apt-get install -y ffmpeg \
                || log_warn "Failed to install ffmpeg. Please install manually: sudo apt-get install ffmpeg"
        else
            log_warn "apt-get not found. Install ffmpeg manually for your distribution."
        fi
    else
        log_ok "FFmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
    fi
}

# =============================================================================
# SHARED HELPERS
# =============================================================================

check_gpu_support() {
    log_section "Validating GPU Support"
    log_info "Testing PyTorch CUDA availability..."
    
    local TARGET_PYTHON="${ACTIVE_PYTHON:-python}"
    if [ "${LIGHTNING_MODE}" -eq 0 ] && [ -f "${INSTALL_DIR}/.venv/bin/python" ]; then
        TARGET_PYTHON="${INSTALL_DIR}/.venv/bin/python"
    fi

    local test_cmd="import torch; assert torch.cuda.is_available(), 'CUDA is NOT available. The server requires a GPU build of PyTorch!'"
    
    "${TARGET_PYTHON}" -c "${test_cmd}" || { log_error "PyTorch GPU Validation Failed! The installed PyTorch bundle does not support your GPU."; exit 1; }
    log_ok "PyTorch GPU support validated: $(${TARGET_PYTHON} -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
}


# ── Install pytorch3d ──────────────────────────────────────────────────────────
install_pytorch3d() {
    log_section "Installing pytorch3d"
    
    local TARGET_PYTHON="${ACTIVE_PYTHON:-python}"
    local PYTORCH3D_OK=0
    "${TARGET_PYTHON}" -c "import pytorch3d" 2>/dev/null && PYTORCH3D_OK=1 || true

    if [ "${PYTORCH3D_OK}" -eq 1 ]; then
        local PYTORCH3D_VER=$("${TARGET_PYTHON}" -c "import pytorch3d; print(pytorch3d.__version__)" 2>/dev/null || echo "unknown")
        log_ok "pytorch3d==${PYTORCH3D_VER} is already installed. Skipping."
        return
    fi
    
    log_info "pytorch3d not found. Installing from source (this may take 10-20 minutes)..."
    
    # We must patch cpp_extension.py to bypass CUDA 13.0 strict ENFORCEMENT
    log_info "Directly patching cpp_extension.py to bypass strict CUDA checks..."
    local CPP_EXT=$("${TARGET_PYTHON}" -c "import torch.utils.cpp_extension as c; print(c.__file__)")
    if [ -f "${CPP_EXT}" ]; then
        cp "${CPP_EXT}" "${CPP_EXT}.bak"
        "${TARGET_PYTHON}" -c "
import sys
try:
    with open('${CPP_EXT}', 'r', encoding='utf-8') as f:
        content = f.read()
    if 'raise RuntimeError(CUDA_MISMATCH_MESSAGE' in content:
        content = content.replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'pass # raise RuntimeError(CUDA_MISMATCH_MESSAGE')
        with open('${CPP_EXT}', 'w', encoding='utf-8') as f:
            f.write(content)
        print('    [OK] Successfully patched cpp_extension.py to bypass CUDA mismatch.')
    else:
        print('    [WARN] CUDA_MISMATCH_MESSAGE not found in cpp_extension.py.')
except Exception as e:
    print(f'    [ERROR] Failed to patch cpp_extension.py: {e}')
"
    fi

    # Using --no-build-isolation combined with @stable fb repo
    # Bypass CUDA 13.0 symbol visibility breaking changes which cause "undefined reference to pulsar::Renderer::calc_signature<true>"
    NVCC_FLAGS="-O3 --device-entity-has-hidden-visibility=false -static-global-template-stub=false" \
    "${TARGET_PYTHON}" -m pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    local BUILD_STATUS=$?

    # Restore the original cpp_extension.py
    if [ -f "${CPP_EXT}.bak" ]; then
        log_info "Restoring original cpp_extension.py..."
        mv "${CPP_EXT}.bak" "${CPP_EXT}"
    fi

    if [ "${BUILD_STATUS}" -eq 0 ]; then
        log_ok "pytorch3d installed natively."
    else
        log_error "pytorch3d installation failed."
        exit 1
    fi
}


# ── Install diff-gaussian-rasterization ─────────────────────────────────────
install_diff_gaussian() {
    log_section "Installing diff-gaussian-rasterization"

    local artalk_path="${1}"
    local TARGET_PYTHON="${ACTIVE_PYTHON:-python}"

    # PURIFY ENVIRONMENT: Ensure we use system tools, not Conda's mismatched wrappers
    unset CC CXX LD AR NM RANLIB

    cd "${artalk_path}"
    if [ -d "diff-gaussian-rasterization" ]; then
        log_warn "diff-gaussian-rasterization directory already exists. Attempting to update..."
        cd diff-gaussian-rasterization
        git pull --recurse-submodules || log_warn "Failed to update diff-gaussian-rasterization. Continuing with existing version."
        cd ..
    else
        log_info "Cloning diff-gaussian-rasterization with submodules..."
        git clone --recurse-submodules "${DIFF_GAUSS_REPO}" diff-gaussian-rasterization
        log_ok "Cloned successfully."
    fi
    local DIFF_GAUSS_DIR="${artalk_path}/diff-gaussian-rasterization" # Redefine after potential cd

    # Fix: add #include <cstdint> after #pragma once in rasterizer_impl.h
    local RAST_HEADER="${DIFF_GAUSS_DIR}/cuda_rasterizer/rasterizer_impl.h"
    if [ ! -f "${RAST_HEADER}" ]; then
        log_error "Header not found: ${RAST_HEADER}. Ensure submodule clone succeeded."
        exit 1
    fi

    if grep -q '#include <cstdint>' "${RAST_HEADER}"; then
        log_warn "cstdint fix already applied. Skipping."
    else
        if grep -q '#pragma once' "${RAST_HEADER}"; then
            log_info "Applying cstdint fix after '#pragma once'..."
            sed -i '/#pragma once/a #include <cstdint>' "${RAST_HEADER}"
        else
            log_warn "'#pragma once' not found. Inserting fix at top of file..."
            sed -i '1s/^/#include <cstdint>\n/' "${RAST_HEADER}"
        fi
        log_ok "cstdint fix applied."
    fi

    log_info "Building and installing diff-gaussian-rasterization (this may take several minutes)..."
    cd "${DIFF_GAUSS_DIR}"
    log_info "Ensuring build dependencies (setuptools, wheel) are present..."
    # Always try to install via pip directly just in case uv isn't wrapping pip securely
    pip install setuptools wheel --quiet 2>/dev/null || true

    local TARGET_PYTHON="${ACTIVE_PYTHON:-python}"

    log_info "Building and installing diff-gaussian-rasterization natively (no-isolation)..."
    
    # ── LIGHTNING.AI FIX: Bypass CUDA 13.0 vs 12.1 version enforcement ──────────
    log_info "Directly patching cpp_extension.py to bypass strict CUDA checks..."
    local CPP_EXT
    CPP_EXT=$("${TARGET_PYTHON}" -c "import torch.utils.cpp_extension as c; print(c.__file__)")
    if [ -f "${CPP_EXT}" ]; then
        # Create a backup of the original file
        cp "${CPP_EXT}" "${CPP_EXT}.bak"
        # Comment out the specific RuntimeError using Python for cross-platform robustness
        "${TARGET_PYTHON}" -c "
import sys
try:
    with open('${CPP_EXT}', 'r', encoding='utf-8') as f:
        content = f.read()
    if 'raise RuntimeError(CUDA_MISMATCH_MESSAGE' in content:
        content = content.replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'pass # raise RuntimeError(CUDA_MISMATCH_MESSAGE')
        with open('${CPP_EXT}', 'w', encoding='utf-8') as f:
            f.write(content)
        print('    [OK] Successfully patched cpp_extension.py to bypass CUDA mismatch.')
    else:
        print('    [WARN] CUDA_MISMATCH_MESSAGE not found in cpp_extension.py.')
except Exception as e:
    print(f'    [ERROR] Failed to patch cpp_extension.py: {e}')
"
    fi

    # --no-build-isolation is critical in Lightning due to isolated environments lacking torch metadata
    "${TARGET_PYTHON}" -m pip install . --no-build-isolation --no-cache-dir --force-reinstall
    local BUILD_STATUS=$?
    
    # Restore the original cpp_extension.py
    if [ -f "${CPP_EXT}.bak" ]; then
        log_info "Restoring original cpp_extension.py..."
        mv "${CPP_EXT}.bak" "${CPP_EXT}"
    fi
    
    if [ "${BUILD_STATUS}" -eq 0 ]; then
        log_ok "diff-gaussian-rasterization installed natively."
    else
        log_error "diff-gaussian-rasterization installation failed."
        exit 1
    fi

    cd "${INSTALL_DIR}"
    log_info "Cleaning up diff-gaussian-rasterization build directory..."
    rm -rf "${DIFF_GAUSS_DIR}"
    log_ok "Build directory removed."
}

# ── Fix onnx/DiagnosticOptions conflict ─────────────────────────────────────
# Uses _pip_install / _pip_uninstall (respects LIGHTNING_MODE).
fix_onnx_conflict() {
    log_section "Checking onnx/torch compatibility"

    local TARGET_PYTHON="${ACTIVE_PYTHON:-python}"
    DIAG_ERROR=0
    "${TARGET_PYTHON}" -c "from torch.onnx._internal.exporter import DiagnosticOptions" 2>/dev/null || DIAG_ERROR=1

    if [ "${DIAG_ERROR}" -eq 1 ]; then
        log_warn "DiagnosticOptions conflict detected (onnx2torch incompatible with current PyTorch)."
        log_info "Removing conflicting packages..."
        _pip_uninstall -y torch torchvision torchaudio onnx onnx2torch 2>/dev/null || true
        log_info "Reinstalling PyTorch 2.4.1 (cu121)..."
        _pip_install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
            --index-url https://download.pytorch.org/whl/cu121 \
            --index-strategy unsafe-best-match
        log_ok "PyTorch reinstalled cleanly."
        TORCH_REINSTALLED=1
    else
        log_ok "No DiagnosticOptions conflict. Continuing."
        TORCH_REINSTALLED=0
    fi
}



# ── Download ARTalk model weights ────────────────────────────────────────────
# Inlines all wget downloads directly — does NOT depend on build_resources.sh
# being present in the ARTalk clone. Skips files that already exist.
prepare_model_resources() {
    log_section "Downloading ARTalk Model Weights"

    local artalk_path="${INSTALL_DIR}/external_models/ARTalk"
    local assets_dir="${artalk_path}/assets"

    # If the primary weight is already downloaded, skip the whole step
    if [ -f "${assets_dir}/ARTalk_wav2vec.pt" ] && \
       [ -f "${assets_dir}/FLAME_with_eye.pt" ] && \
       [ -f "${assets_dir}/GAGAvatar/GAGAvatar.pt" ]; then
        log_ok "Model weights already present. Skipping download."
        return
    fi

    log_info "Creating asset directories..."
    mkdir -p \
        "${assets_dir}/GAGAvatar" \
        "${assets_dir}/style_motion"

    # ── GAGAvatar base weights ────────────────────────────────────────────────
    log_info "Downloading FLAME_with_eye.pt..."
    wget -q --show-progress \
        "https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/FLAME_with_eye.pt" \
        -O "${assets_dir}/FLAME_with_eye.pt"

    log_info "Downloading GAGAvatar.pt..."
    wget -q --show-progress \
        "https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/GAGAvatar.pt" \
        -O "${assets_dir}/GAGAvatar/GAGAvatar.pt"

    # ── ARTalk weights ────────────────────────────────────────────────────────
    log_info "Downloading ARTalk_wav2vec.pt..."
    wget -q --show-progress \
        "https://huggingface.co/xg-chu/ARTalk/resolve/main/ARTalk_wav2vec.pt" \
        -O "${assets_dir}/ARTalk_wav2vec.pt"

    log_info "Downloading config.json..."
    wget -q --show-progress \
        "https://huggingface.co/xg-chu/ARTalk/resolve/main/config.json" \
        -O "${assets_dir}/config.json"

    log_info "Downloading GAGAvatar/tracked.pt..."
    wget -q --show-progress \
        "https://huggingface.co/xg-chu/ARTalk/resolve/main/GAGAvatar/tracked.pt" \
        -O "${assets_dir}/GAGAvatar/tracked.pt"

    # ── Style motion files ────────────────────────────────────────────────────
    log_info "Downloading style_motion files..."
    for style in \
        angry_0 curious_0 \
        doubtful_0 doubtful_1 \
        happy_0 happy_1 happy_2 \
        natural_0 natural_1 natural_2 natural_3 natural_4 natural_5 natural_6 natural_7
    do
        wget -q --show-progress \
            "https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/${style}.pt" \
            -O "${assets_dir}/style_motion/${style}.pt"
    done

    log_ok "All model weights downloaded to ${assets_dir}."
}

# ── Watermark monkey-patch ───────────────────────────────────────────────────
apply_watermark_patch() {
    log_section "Removing ARTalk Watermark (Monkey Patch)"

    local INFERENCE_PY="${INSTALL_DIR}/external_models/ARTalk/inference.py"
    if [ ! -f "${INFERENCE_PY}" ]; then
        log_warn "inference.py not found at ${INFERENCE_PY}. Skipping patch."
        return
    fi

    if grep -q 'GAGAvatar.add_water_mark' "${INFERENCE_PY}"; then
        log_warn "Watermark patch already applied. Skipping."
    else
        log_info "Applying watermark removal patch..."
        sed -i '/engine = ARTAvatarInferEngine(load_gaga=True/a \    # Monkey patch to remove watermark\n    if hasattr(engine, "GAGAvatar"):\n        engine.GAGAvatar.add_water_mark = lambda image: image' "${INFERENCE_PY}"
        log_ok "Watermark patch applied."
    fi
}

# ── Generate .env template ───────────────────────────────────────────────────
generate_server_env() {
    local artalk_path="${INSTALL_DIR}/external_models/ARTalk"
    if [ ! -f "${INSTALL_DIR}/.env" ]; then
        log_info "Generating .env template..."
        cat > "${INSTALL_DIR}/.env" <<EOF
# =============================================================
#  ARTalk Server — Environment Variables
#  Fill in ALL values before starting the server.
# =============================================================

# --- ARTalk Model ---
ARTALK_PATH=${artalk_path}
ARTALK_MODEL_STRATEGY=from_scratch
# Options for ARTALK_MODEL_STRATEGY: from_scratch | example_models

# --- ARTalk Server ---
ARTALK_SERVER_HOST=0.0.0.0
ARTALK_SERVER_PORT=8000

# --- LiveKit (server connects as a participant) ---
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
EOF
        log_ok ".env template written."
    else
        log_warn ".env already exists. Skipping generation."
    fi
}

# =============================================================================
# INSTALLATION PATH A — Standard Linux (conda / uv)
# =============================================================================
install_server_standard() {
    log_section "STEP 1 (Standard) — Python Verification"

    PYTHON_CMD=""
    CANDIDATES=("python3.10" "python3.11" "python3" "python")

    if command -v uv &> /dev/null; then
        UV_310=$(uv python find 3.10 2>/dev/null || true)
        if [ -z "${UV_310}" ]; then
            log_info "Python 3.10 not found via uv. Installing..."
            uv python install 3.10
            UV_310=$(uv python find 3.10 2>/dev/null || true)
        fi
        [ -n "${UV_310}" ] && CANDIDATES=("${UV_310}" "${CANDIDATES[@]}")
    fi

    for cmd in "${CANDIDATES[@]}"; do
        if command -v "${cmd}" &> /dev/null; then
            VER_STR=$("${cmd}" --version 2>&1)
            if [[ "${VER_STR}" == *"Python 3.10"* ]] || [[ "${VER_STR}" == *"Python 3.11"* ]]; then
                PYTHON_CMD="${cmd}"
                break
            fi
        fi
    done

    if [ -z "${PYTHON_CMD}" ]; then
        log_error "Python 3.10 or 3.11 is required. Neither was found."
        exit 1
    fi
    log_ok "Using: $(${PYTHON_CMD} --version)"

    # ── uv ───────────────────────────────────────────────────────────────────
    log_section "STEP 2 (Standard) — Install uv"
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1090
        source "${HOME}/.cargo/env"
    fi

    # ── Sync plugin + server extras into .venv ───────────────────────────────
    log_section "STEP 3 (Standard) — Install plugin + server extras"
    log_info "Syncing dependencies into .venv..."
    uv sync --python "${PYTHON_CMD}"
    uv pip install --python .venv -e ".[server]"
    log_ok "Plugin and server extras installed in .venv."

    # ── PyTorch CUDA ─────────────────────────────────────────────────────────
    log_section "STEP 4 (Standard) — PyTorch 2.4.1 + CUDA 12.1"
    log_info "Installing torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 (cu121)..."
    uv pip install --python .venv torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --index-strategy unsafe-best-match
    log_ok "PyTorch tools installed."

    # ── Clone ARTalk ─────────────────────────────────────────────────────────
    log_section "STEP 5 (Standard) — Clone ARTalk"
    mkdir -p "${INSTALL_DIR}/external_models"
    if [ ! -d "${INSTALL_DIR}/external_models/ARTalk" ]; then
        log_info "Cloning ARTalk with submodules..."
        git clone --recurse-submodules "${ARTALK_REPO}" "${INSTALL_DIR}/external_models/ARTalk"
        log_ok "ARTalk cloned."
    else
        log_warn "external_models/ARTalk already exists. Skipping clone."
    fi

    ARTALK_PATH="${INSTALL_DIR}/external_models/ARTalk"

    # ── ARTalk Python deps ───────────────────────────────────────────────────
    log_section "STEP 6 (Standard) — ARTalk Python Dependencies"
    log_info "Installing ARTalk pip requirements..."
    uv pip install --python .venv \
        trimesh \
        gdown \
        scipy \
        tqdm \
        opencv-python \
        soundfile \
        librosa \
        einops
    log_ok "ARTalk Python requirements installed."

    # ── diff-gaussian-rasterization ──────────────────────────────────────────
    log_section "STEP 7 (Standard) — diff-gaussian-rasterization"
    # Activate the venv so pip inside install_diff_gaussian uses the right env
    # shellcheck disable=SC1091
    source "${INSTALL_DIR}/.venv/bin/activate"
    install_pytorch3d
    install_diff_gaussian "${ARTALK_PATH}" "--no-cache-dir"
    deactivate

    # ── Fix onnx conflict ────────────────────────────────────────────────────
    # shellcheck disable=SC1091
    source "${INSTALL_DIR}/.venv/bin/activate"
    fix_onnx_conflict
    deactivate

    prepare_model_resources
    apply_watermark_patch
    generate_server_env

    log_section "STEP 8 (Standard) — Enforcing NumPy Sweet Spot (NumPy 2.0.x)"
    log_info "Locking numpy to 2.0.2. Numba/SciPy require < 2.1, while LiveKit/OpenCV require >= 2.0.1..."
    # shellcheck disable=SC1091
    source "${INSTALL_DIR}/.venv/bin/activate"
    pip install --no-cache-dir "numpy==2.0.2" scipy librosa
    deactivate

    check_gpu_support

    log_section "✅  SERVER INSTALLATION COMPLETE (Standard)"
    echo -e "${GREEN}"
    echo "  ARTalk model  : ${ARTALK_PATH}"
    echo "  Environment   : .venv"
    echo ""
    echo "  Activate environment:"
    echo "    source .venv/bin/activate"
    echo ""
    echo "  Start the ARTalk server:"
    echo "    python examples/start_artalk_server.py"
    echo "    # or: uvicorn artalk_server.main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "  Then run the client plugin on the brain machine using install_client.sh"
    echo -e "${NC}"
}

# =============================================================================
# INSTALLATION PATH B — Lightning.ai Studio
# =============================================================================
install_server_lightning() {
    log_section "STEP 1 (Lightning) — Verify Environment & Enforce Python 3.10"

    # ── uv ───────────────────────────────────────────────────────────────────
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1090
        source "${HOME}/.cargo/env"
    fi
    log_ok "uv ready: $(uv --version)"

    # --- CONDA CLOUDSPACE ENVIRONMENT ---
    # Lightning.ai exposes cloudspace via two path overlays:
    #   /home/zeus/miniconda3/envs/cloudspace/  <-- user writable, packages installed here are visible HERE
    #   /system/conda/miniconda3/envs/cloudspace/ <-- read-only base, only its packages are visible HERE
    # CRITICAL: use whichever python is in PATH (resolves to /home/zeus/...) so it sees
    # ALL installed packages regardless of which overlay they came from.
    # This matches the pattern from the working install_artalk_lightning.sh script.
    ACTIVE_PYTHON="$(which python3)"
    # Prefer the conda python in PATH which already has the right visibility
    if command -v python &>/dev/null; then
        ACTIVE_PYTHON="$(which python)"
    fi
    export ACTIVE_PYTHON

    log_ok "Target Python (PATH): ${ACTIVE_PYTHON}"
    log_info "Python version: $(${ACTIVE_PYTHON} --version)"

    log_info "Installing mandatory build tools (setuptools, wheel) into cloudspace..."
    "${ACTIVE_PYTHON}" -m pip install --no-cache-dir setuptools wheel --upgrade --quiet

    # Use _pip_install/_pip_uninstall wrappers (LIGHTNING_MODE=1)
    LIGHTNING_MODE=1

    # ── Bypass compiler_compat/ld ────────────────────────────────────────────
    # Conda's ld linker often breaks PyTorch C++ extensions. Rename it if it exists.
    for CC_LD in "/home/zeus/miniconda3/envs/cloudspace/compiler_compat/ld" "/system/conda/miniconda3/envs/cloudspace/compiler_compat/ld"; do
        if [ -f "${CC_LD}" ]; then
            log_info "Renaming conda's compiler_compat/ld to prevent linking errors..."
            mv "${CC_LD}" "${CC_LD}.bak" 2>/dev/null || true
        fi
    done

    # ── Clone ARTalk FIRST ────────────────────────────────────────────────────
    # IMPORTANT: clone happens before any pip step that could abort the script
    # so the repo is always available even if a subsequent install fails.
    log_section "STEP 3 (Lightning) — Clone ARTalk repository"
    mkdir -p "${INSTALL_DIR}/external_models"
    if [ ! -d "${INSTALL_DIR}/external_models/ARTalk" ]; then
        log_info "Cloning ARTalk with submodules..."
        git clone --recurse-submodules "${ARTALK_REPO}" "${INSTALL_DIR}/external_models/ARTalk"
        log_ok "ARTalk cloned successfully."
    else
        log_warn "external_models/ARTalk already exists. Skipping clone."
    fi
    ARTALK_PATH="${INSTALL_DIR}/external_models/ARTalk"

    # ── Clone GAGAvatar_track ─────────────────────────────────────────────────
    log_info "Cloning GAGAvatar_track..."
    GAGTRACK_DIR="${INSTALL_DIR}/external_models/GAGAvatar_track"
    if [ ! -d "${GAGTRACK_DIR}" ]; then
        git clone "${GAGTRACK_REPO}" "${GAGTRACK_DIR}"
        log_ok "GAGAvatar_track cloned."
    else
        log_warn "external_models/GAGAvatar_track already exists. Skipping."
    fi

    # ── Dependency Alignment ──────────────────────────────
    log_info "Installing livekit-plugins-artalk base layer..."
    _pip_install -e .
    log_ok "Base plugin packages installed."

    log_info "Installing ARTalk core dependencies..."
    _pip_install \
        trimesh \
        gdown \
        tqdm \
        soundfile \
        librosa \
        opencv-python-headless \
        imageio==2.35.1 \
        imageio-ffmpeg \
        lmdb==1.5.1
    log_ok "ARTalk core packages installed."

    log_info "Installing GAGAvatar_track dependencies (strict pinning)..."
    _pip_install "numpy<2.0.0"
    _pip_install \
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
    log_ok "GAGAvatar packages installed."

    # ── PyTorch Pre-Check: DiagnosticOptions & Corruption ────────────────────
    log_section "STEP 4b (Lightning) — Detect and fix onnx/torch conflict"
    DIAG_ERROR=0
    "${ACTIVE_PYTHON}" -c "from torch.onnx._internal.exporter import DiagnosticOptions" 2>/dev/null || DIAG_ERROR=1
    "${ACTIVE_PYTHON}" -c "import torch.utils.cpp_extension" 2>/dev/null || DIAG_ERROR=1

    if [ "${DIAG_ERROR}" -eq 1 ]; then
        log_warn "Detected Torch corruption or 'DiagnosticOptions' conflict. Forcing complete reinstall..."
        # Safely remove files directly to prevent pip's Errno 2 when uninstalling across symlinks
        for SITE_DIR in "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages" "/system/conda/miniconda3/envs/cloudspace/lib/python3.12/site-packages"; do
            if [ -d "${SITE_DIR}" ]; then 
                rm -rf "${SITE_DIR}/torch" "${SITE_DIR}/torch-"* "${SITE_DIR}/torchvision"* "${SITE_DIR}/torchaudio"* 2>/dev/null || true
                rm -rf "${SITE_DIR}/functorch" "${SITE_DIR}/functorch-"* 2>/dev/null || true
            fi
        done
        log_info "Downloading and reinstalling torch suite cleanly (ignoring existing files)..."
        # We explicitly omit 'pip uninstall' here because it causes [Errno 2] OS Errors 
        # on the Lightning split-path environments. Overwriting in place is safer.
        "${ACTIVE_PYTHON}" -m pip install --ignore-installed --no-cache-dir torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121
        log_ok "torch/torchvision/torchaudio reinstalled successfully."
    else

        log_ok "No DiagnosticOptions conflict detected. Continuing."
    fi


    # ── PyTorch CUDA ─────────────────────────────────────────────────────────
    log_section "STEP 5 (Lightning) — PyTorch 2.4.1 + CUDA 12.1"
    
    # EXACT LOGIC COPY FROM develop_external_mode (install_artalk_lightning.sh)
    # We never blindly `pip install torch` because doing so over a symlinked conda 
    # environment causes Errno 2 on functorch and other SO files.
    TORCH_VER=$("${ACTIVE_PYTHON}" -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
    TORCH_CUDA=$("${ACTIVE_PYTHON}" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")

    if [ -z "${TORCH_VER}" ]; then
        log_warn "PyTorch completely missing in cloudspace. Installing 2.4.1+cu121..."
        "${ACTIVE_PYTHON}" -m pip install --no-cache-dir \
            torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
            --index-url https://download.pytorch.org/whl/cu121
        log_ok "PyTorch 2.4.1+cu121 installed."
    else
        log_info "Detected torch==${TORCH_VER}  CUDA==${TORCH_CUDA}"
        if [ -n "${TORCH_CUDA}" ]; then
            CUDA_TAG="cu$(echo "${TORCH_CUDA}" | tr -d '.')"
        else
            CUDA_TAG="cpu"
        fi
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
        log_info "PyTorch wheel index: ${TORCH_INDEX}"

        TORCHAUDIO_OK=0
        "${ACTIVE_PYTHON}" -c "import torchaudio" 2>/dev/null && TORCHAUDIO_OK=1 || true

        if [ "${TORCHAUDIO_OK}" -eq 0 ]; then
            log_warn "torchaudio not found. Installing..."
            "${ACTIVE_PYTHON}" -m pip install --no-cache-dir "torchaudio==${TORCH_VER}" --index-url "${TORCH_INDEX}" 2>/dev/null \
                || "${ACTIVE_PYTHON}" -m pip install --no-cache-dir torchaudio --index-url "${TORCH_INDEX}" \
                || "${ACTIVE_PYTHON}" -m pip install --no-cache-dir torchaudio \
                || { log_error "All torchaudio installation strategies failed."; exit 1; }
            "${ACTIVE_PYTHON}" -c "import torchaudio; print('torchaudio version:', torchaudio.__version__)" \
                || { log_error "torchaudio installed but cannot be imported."; exit 1; }
        else
            TORCHAUDIO_VER=$("${ACTIVE_PYTHON}" -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "unknown")
            log_ok "torchaudio==${TORCHAUDIO_VER} is already available."
        fi
    fi




    # (Removed redundant hardcoded STEP 6 pip packages because they are 
    # now correctly parsed and inherently installed directly from the
    # environment.yml file earlier in STEP 4, bypassing OS errors.)

    # ── Install ninja (parallel CUDA compilation — dramatically faster) ───────
    log_info "Installing ninja build system..."
    if command -v ninja &>/dev/null; then
        log_ok "ninja already installed: $(ninja --version)"
    else
        sudo apt-get install -y ninja-build 2>/dev/null \
            && log_ok "ninja installed via apt-get." \
            || ( _pip_install ninja && log_ok "ninja installed via pip." ) \
            || log_warn "Failed to install ninja. Build will use slow distutils backend."
    fi

    # (Removed aggressive CUDA_HOME alignment and PyTorch reinstallation
    # because the working install_artalk_lightning.sh script does not do it,
    # and re-installing PyTorch here risks another pip Errno 2 OS crash.)

    # ── diff-gaussian-rasterization ──────────────────────────────────────────
    log_section "STEP 7 (Lightning) — diff-gaussian-rasterization"

    install_pytorch3d
    install_diff_gaussian "${ARTALK_PATH}"


    prepare_model_resources
    apply_watermark_patch
    generate_server_env
    
    log_section "STEP 8 (Lightning) — Enforcing NumPy Sweet Spot (NumPy 2.0.x)"
    log_info "Locking numpy to 2.0.2. Numba/SciPy require < 2.1, while LiveKit/OpenCV require >= 2.0.1..."
    _pip_install "numpy==2.0.2" scipy librosa

    check_gpu_support

    log_section "✅  SERVER INSTALLATION COMPLETE (Lightning.ai)"
    echo -e "${GREEN}"
    echo "  ARTalk model  : ${ARTALK_PATH}"
    echo "  GAGAvatar_track : ${GAGTRACK_DIR}"
    echo "  Environment   : cloudspace (system)"
    echo ""
    echo "  Start the ARTalk server:"
    echo "    python examples/start_artalk_server.py"
    echo "    # or: uvicorn artalk_server.main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "  Expose the server port (e.g. 8000) for the brain agent to connect."
    echo -e "${NC}"
}


# =============================================================================
# MAIN — Environment Selection
# =============================================================================
log_section "ARTalk Server Installer"
echo ""
echo -e "${YELLOW}This script installs the GPU-side ARTalk rendering server.${NC}"
echo -e "${YELLOW}For the LiveKit agent (brain) machine, use install_client.sh instead.${NC}"
echo ""
echo -e "${BOLD}Select your target environment:${NC}"
echo "  1) Standard Linux  (Miniconda / uv / local / VPS / cloud VM)"
echo "  2) Lightning.ai    (installs into 'cloudspace' — no conda env creation)"
echo ""
read -rp "Enter choice [1-2]: " ENV_CHOICE

case "${ENV_CHOICE}" in
    1)
        check_ffmpeg
        install_server_standard
        ;;
    2)
        install_server_lightning
        ;;
    *)
        log_error "Invalid choice '${ENV_CHOICE}'. Run the script again and select 1 or 2."
        exit 1
        ;;
esac
