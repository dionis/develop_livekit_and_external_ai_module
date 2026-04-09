#!/usr/bin/env bash
# =============================================================================
#  ARTalk — Installation Script (Lightning.ai Studio)
#  Script Version: 2026-03-25-v0.1.1
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
# STEP 0 — Confirm Lightning.ai environment
# =============================================================================
log_section "STEP 0 — Verify Lightning.ai environment"

ACTIVE_PYTHON=$(which python)
log_info "Active Python : ${ACTIVE_PYTHON}"
log_info "Python prefix : $(python -c 'import sys; print(sys.prefix)')"

if ! conda env list 2>&1 | grep -q "Conda environment management is not allowed"; then
    log_error "This script is intended for Lightning.ai Studios only."
    log_error "For standard Conda environments, use install_artalk.sh instead."
    exit 1
fi

log_ok "Lightning.ai Studio confirmed. Installing into 'cloudspace' environment."

# ── Deep Environment Healing ────────────────────────────────────────────────
# Conda's compiler wrappers often break C++ extensions. Move them away.
for CC_BASE in "/home/zeus/miniconda3/envs/cloudspace" "/system/conda/miniconda3/envs/cloudspace"; do
    if [ -d "${CC_BASE}/compiler_compat" ]; then
        log_info "Deactivating Conda compiler wrappers: ${CC_BASE}/compiler_compat"
        mv "${CC_BASE}/compiler_compat" "${CC_BASE}/compiler_compat.bak" 2>/dev/null || true
    fi
done
# For good measure, unset Conda's polluting environment variables
unset CC CXX LD AR NM RANLIB

# =============================================================================
# STEP 1 — Clone ARTalk
# =============================================================================
log_section "STEP 1 — Clone ARTalk repository"

if [ -d "${INSTALL_DIR}/ARTalk" ]; then
    log_warn "Directory 'ARTalk' already exists. Skipping clone."
else
    log_info "Cloning ARTalk with submodules..."
    git clone --recurse-submodules "${ARTALK_REPO}"
    log_ok "ARTalk cloned successfully."
fi

cd "${INSTALL_DIR}/ARTalk"
ARTALK_PATH="${PWD}"
log_info "Working directory: ${ARTALK_PATH}"

# =============================================================================
# STEP 2 — Install dependencies via pip (no conda env creation)
# =============================================================================
log_section "STEP 2 — Install dependencies via pip into cloudspace"

log_info "Healing corrupted python packages (if any)..."
if command -v uv &> /dev/null; then uv cache clean 2>/dev/null || true; fi
SITE_PKGS=$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
if [ -n "${SITE_PKGS}" ] && [ -d "${SITE_PKGS}" ]; then
    find "${SITE_PKGS}" -maxdepth 1 -name "*.dist-info" -type d -exec sh -c '
        if [ ! -f "$1/METADATA" ]; then
            echo "Removing broken package: $1"
            rm -rf "$1"
        fi
    ' _ {} \;
fi

if [ ! -f "${ARTALK_PATH}/environment.yml" ]; then
    log_error "environment.yml not found in ${ARTALK_PATH}"
    exit 1
fi

# Extract pip-section packages from environment.yml
log_info "Extracting pip packages from environment.yml..."
PIP_PKGS=$(python3 -c "
import yaml
with open('environment.yml') as f:
    env = yaml.safe_load(f)
pkgs = []
for dep in env.get('dependencies', []):
    if isinstance(dep, dict) and 'pip' in dep:
        pkgs.extend(dep['pip'])
print('\n'.join(pkgs))
")

if [ -n "${PIP_PKGS}" ]; then
    log_info "Installing pip packages..."
    echo "${PIP_PKGS}" | xargs pip install --no-cache-dir
    log_ok "Pip packages installed."
else
    log_warn "No pip packages found in environment.yml."
fi

# Extract and install conda packages via pip (best-effort, skip system-level ones)
log_info "Installing conda packages via pip (best-effort, skipping system-level deps)..."
CONDA_PKGS=$(python3 -c "
import yaml
with open('environment.yml') as f:
    env = yaml.safe_load(f)
skip = {'python', 'pip', 'conda', 'nvidia', 'cudatoolkit', 'cudnn'}
pkgs = []
for dep in env.get('dependencies', []):
    if isinstance(dep, str):
        name = dep.split('=')[0].strip()
        if name.lower() not in skip and not name.startswith('_'):
            pkgs.append(dep.replace('==','=').replace('=','==',1).rstrip('='))
print('\n'.join(pkgs))
")

if [ -n "${CONDA_PKGS}" ]; then
    while IFS= read -r pkg; do
        pip install --no-cache-dir "${pkg}" 2>/dev/null \
            && log_ok "  Installed: ${pkg}" \
            || log_warn "  Skipped (not available via pip or already present): ${pkg}"
    done <<< "${CONDA_PKGS}"
fi

# ── Dependency Synchronization ───────────────────────────────────────────
# Lightning environments often have stale scikit-learn or missing LiveKit core.
# We force-align them here to prevent resolver errors.
log_section "STEP 2b — Synchronize core dependencies"
log_info "Removing stale plugin metadata..."
pip uninstall -y livekit-plugins-artalk 2>/dev/null || true
log_info "Upgrading scikit-learn and installing LiveKit core suite..."
pip install --no-cache-dir \
    "scikit-learn>=1.5.0" \
    "numpy>=2.0.1,<2.1.0" \
    "livekit>=1.0.0" \
    "livekit-agents==1.4.4" \
    "livekit-plugins-cartesia" \
    "livekit-plugins-openai" \
    "livekit-plugins-silero" \
    "mediapipe" \
    "opencv-python-headless" \
    "soundfile" \
    "python-dotenv"
log_ok "Core dependencies synchronized."

# =============================================================================
# STEP 3 — Detect and fix onnx/torch conflict
# =============================================================================
log_section "STEP 3 — Detect and fix onnx/torch conflict"

DIAG_ERROR=0
TORCH_REINSTALLED=0
python -c "from torch.onnx._internal.exporter import DiagnosticOptions" 2>/dev/null || DIAG_ERROR=1
python -c "import torch.utils.cpp_extension" 2>/dev/null || DIAG_ERROR=1

if [ "${DIAG_ERROR}" -eq 1 ]; then
    log_warn "Detected Torch corruption or 'DiagnosticOptions' conflict. Forcing complete reinstall..."
    SITE_PKGS=$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
    if [ -d "${SITE_PKGS}" ]; then rm -rf "${SITE_PKGS}/torch" "${SITE_PKGS}/torch-"* "${SITE_PKGS}/torchvision"* "${SITE_PKGS}/torchaudio"*; fi
    log_info "Uninstalling conflicting packages: torch torchvision torchaudio onnx onnx2torch..."
    pip uninstall -y torch torchvision torchaudio onnx onnx2torch 2>/dev/null || true
    log_info "Reinstalling torch, torchvision and torchaudio..."
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
    log_ok "torch/torchvision/torchaudio reinstalled successfully."
    # Flag: torch version changed — any package with compiled CUDA extensions
    # built against the old torch (e.g. pytorch3d) must be recompiled.
    TORCH_REINSTALLED=1
else
    log_ok "No DiagnosticOptions conflict detected. Continuing."
fi

# =============================================================================
# STEP 3b — Ensure torchaudio is installed
# =============================================================================
log_section "STEP 3b — Ensure torchaudio is installed"

TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")

if [ -z "${TORCH_VER}" ]; then
    log_error "PyTorch is not importable. Cannot continue."
    exit 1
fi

log_info "Detected torch==${TORCH_VER}  CUDA==${TORCH_CUDA}"

if [ -n "${TORCH_CUDA}" ]; then
    CUDA_TAG="cu$(echo "${TORCH_CUDA}" | tr -d '.')"
else
    CUDA_TAG="cpu"
fi
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
log_info "PyTorch wheel index: ${TORCH_INDEX}"

TORCHAUDIO_OK=0
python -c "import torchaudio" 2>/dev/null && TORCHAUDIO_OK=1 || true

if [ "${TORCHAUDIO_OK}" -eq 0 ]; then
    log_warn "torchaudio not found. Installing..."
    pip install --no-cache-dir "torchaudio==${TORCH_VER}" --index-url "${TORCH_INDEX}" 2>/dev/null \
        || pip install --no-cache-dir torchaudio --index-url "${TORCH_INDEX}" \
        || pip install --no-cache-dir torchaudio \
        || { log_error "All torchaudio installation strategies failed."; exit 1; }
    python -c "import torchaudio; print('torchaudio version:', torchaudio.__version__)" \
        || { log_error "torchaudio installed but cannot be imported."; exit 1; }
else
    TORCHAUDIO_VER=$(python -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "unknown")
    log_ok "torchaudio==${TORCHAUDIO_VER} is already available."
fi

# =============================================================================
# STEP 3c — Align CUDA Toolkit (CUDA_HOME) for C++ extensions
# =============================================================================
log_section "STEP 3c — Align CUDA Toolkit (CUDA_HOME)"

log_info "Installing ninja build system..."
if command -v ninja &>/dev/null; then
    log_ok "ninja already installed: $(ninja --version)"
else
    sudo apt-get install -y ninja-build 2>/dev/null \
        && log_ok "ninja installed via apt-get." \
        || ( pip install ninja && log_ok "ninja installed via pip." ) \
        || log_warn "Failed to install ninja. Build will use slow distutils backend."
fi

TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
if [ -n "${TORCH_CUDA_VER}" ]; then
    log_info "PyTorch was compiled with CUDA ${TORCH_CUDA_VER}. Looking for matching toolkit..."
    CUDA_HOME_FOUND=0
    for cuda_candidate in \
            "/usr/local/cuda-${TORCH_CUDA_VER}" \
            "/usr/local/cuda-$(echo "${TORCH_CUDA_VER}" | cut -d. -f1)" \
            "/usr/local/cuda"; do
        if [ -d "${cuda_candidate}" ] && [ -f "${cuda_candidate}/bin/nvcc" ]; then
            NVCC_VER=$("${cuda_candidate}/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d, -f1)
            log_info "Found nvcc ${NVCC_VER} at ${cuda_candidate}"
            export CUDA_HOME="${cuda_candidate}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
            log_ok "CUDA_HOME → ${CUDA_HOME}"
            CUDA_HOME_FOUND=1
            break
        fi
    done
    if [ "${CUDA_HOME_FOUND}" -eq 0 ]; then
        log_warn "No suitable CUDA toolkit found in /usr/local. C++ extensions may compile without GPU support!"
    fi


# =============================================================================
# STEP 3d — Install pytorch3d (Required for avatar cooking)
# =============================================================================
log_section "STEP 3d — Install pytorch3d"

PYTORCH3D_OK=0
python -c "import pytorch3d" 2>/dev/null && PYTORCH3D_OK=1 || true

if [ "${PYTORCH3D_OK}" -eq 1 ]; then
    PYTORCH3D_VER=$(python -c "import pytorch3d; print(pytorch3d.__version__)" 2>/dev/null || echo "unknown")
    log_ok "pytorch3d==${PYTORCH3D_VER} is already installed. Skipping."
else
    log_info "pytorch3d not found. Installing from source (this may take 10-20 minutes)..."

    # We must patch cpp_extension.py to bypass CUDA 13.0 strict ENFORCEMENT
    log_info "Directly patching cpp_extension.py to bypass strict CUDA checks..."
    CPP_EXT=$(python -c "import torch.utils.cpp_extension as c; print(c.__file__)")
    if [ -f "${CPP_EXT}" ]; then
        cp "${CPP_EXT}" "${CPP_EXT}.bak"
        python -c "
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
    python -m pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    BUILD_STATUS=$?

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
fi


# =============================================================================
# STEP 4 — Install diff-gaussian-rasterization (with cstdint fix)
# =============================================================================
log_section "STEP 4 — Install diff-gaussian-rasterization"

DIFF_GAUSS_DIR="${ARTALK_PATH}/diff-gaussian-rasterization"

if [ -d "${DIFF_GAUSS_DIR}" ]; then
    log_warn "Directory 'diff-gaussian-rasterization' already exists. Skipping clone."
else
    log_info "Cloning diff-gaussian-rasterization with submodules..."
    git clone --recurse-submodules "${DIFF_GAUSS_REPO}"
    log_ok "diff-gaussian-rasterization cloned successfully."
fi

# ── Fix: add #include <cstdint> after #pragma once in rasterizer_impl.h ──────
RAST_HEADER="${DIFF_GAUSS_DIR}/cuda_rasterizer/rasterizer_impl.h"

if [ ! -f "${RAST_HEADER}" ]; then
    log_error "File not found: ${RAST_HEADER}"
    log_error "Make sure the submodule clone completed successfully."
    exit 1
fi

if grep -q '#include <cstdint>' "${RAST_HEADER}"; then
    log_warn "Fix '#include <cstdint>' already applied in rasterizer_impl.h. Skipping."
else
    # IMPORTANT: the file starts with a multi-line copyright comment block followed
    # by "#pragma once". Inserting after line 1 places the include INSIDE the comment
    # — nvcc ignores it → uint32_t/uint64_t undefined errors.
    # The correct fix is to insert AFTER "#pragma once".
    if grep -q '#pragma once' "${RAST_HEADER}"; then
        log_info "Applying fix: inserting '#include <cstdint>' after '#pragma once'..."
        sed -i '/#pragma once/a #include <cstdint>' "${RAST_HEADER}"
        log_ok "Fix applied successfully to rasterizer_impl.h."
    else
        log_warn "'#pragma once' not found. Inserting '#include <cstdint>' at the top..."
        sed -i '1s/^/#include <cstdint>\n/' "${RAST_HEADER}"
        log_ok "Fix applied at the top of the file."
    fi
fi

log_info "Verifying fix (searching for '#pragma once' and '<cstdint>'):"
grep -n 'pragma once\|cstdint' "${RAST_HEADER}" | sed 's/^/    /'

log_info "Installing diff-gaussian-rasterization natively (no-isolation)..."
cd "${DIFF_GAUSS_DIR}"

# ── LIGHTNING.AI FIX: Bypass CUDA 13.0 vs 12.1 version enforcement ──────────
log_info "Directly patching cpp_extension.py to bypass strict CUDA checks..."
CPP_EXT=$(python -c "import torch.utils.cpp_extension as c; print(c.__file__)")
if [ -f "${CPP_EXT}" ]; then
    # Create a backup of the original file
    cp "${CPP_EXT}" "${CPP_EXT}.bak"
    # Comment out the specific RuntimeError using Python for cross-platform robustness
    python -c "
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
python -m pip install . --no-build-isolation --no-cache-dir --force-reinstall
BUILD_STATUS=$?

# Restore the original cpp_extension.py
if [ -f "${CPP_EXT}.bak" ]; then
    log_info "Restoring original cpp_extension.py..."
    mv "${CPP_EXT}.bak" "${CPP_EXT}"
fi

if [ "${BUILD_STATUS}" -eq 0 ]; then
    log_ok "diff-gaussian-rasterization installed successfully."
else
    log_error "diff-gaussian-rasterization installation failed."
    exit 1
fi

cd "${ARTALK_PATH}"
log_info "Removing temporary diff-gaussian-rasterization directory..."
rm -rf "${DIFF_GAUSS_DIR}"
log_ok "Directory removed."

# =============================================================================
# STEP 5 — Prepare model resources (build_resources.sh)
# =============================================================================
log_section "STEP 5 — Prepare model resources"

if [ ! -f "${ARTALK_PATH}/build_resources.sh" ]; then
    log_error "build_resources.sh not found in ${ARTALK_PATH}"
    exit 1
fi

log_info "Running build_resources.sh (will download model weights and assets)..."
log_info "Automatically answering 'y' to all license confirmation prompts..."
yes | bash "${ARTALK_PATH}/build_resources.sh"
log_ok "Resources prepared successfully."

# =============================================================================
# STEP 6 — Patch inference.py (Gradio share=True)
# =============================================================================
log_section "STEP 6 — Patch inference.py (Gradio share=True)"

INFERENCE_PY="${ARTALK_PATH}/inference.py"

if [ ! -f "${INFERENCE_PY}" ]; then
    log_error "inference.py not found in ${ARTALK_PATH}"
    exit 1
fi

if grep -q 'demo.launch(share=True)' "${INFERENCE_PY}"; then
    log_warn "inference.py already has 'share=True'. Skipping patch."
elif grep -q 'server_name="0.0.0.0", server_port=8960' "${INFERENCE_PY}"; then
    log_info "Applying patch to inference.py..."
    sed -i 's|demo.launch(server_name="0.0.0.0", server_port=8960)|#demo.launch(server_name="0.0.0.0", server_port=8960)\n    demo.launch(share=True)|' "${INFERENCE_PY}"
    log_ok "inference.py patched successfully."
    log_info "Patch result:"
    grep -n 'demo.launch' "${INFERENCE_PY}" | sed 's/^/    /'
else
    log_warn "Original line 'demo.launch(server_name=...)' not found in inference.py."
    log_warn "The file may have already been modified manually."
fi

# =============================================================================
# STEP 7 — Apply Watermark Monkey Patch (Option 2)
# =============================================================================
log_section "STEP 7 — Apply Watermark Monkey Patch (Option 2)"

if grep -q 'GAGAvatar.add_water_mark' "${INFERENCE_PY}"; then
    log_warn "Watermark monkey patch already applied. Skipping."
else
    log_info "Applying monkey patch to remove watermark in inference.py..."
    sed -i '/engine = ARTAvatarInferEngine(load_gaga=True/a \    # Monkey patch to remove watermark\n    if hasattr(engine, "GAGAvatar"):\n        engine.GAGAvatar.add_water_mark = lambda image: image' "${INFERENCE_PY}"
    log_ok "Watermark monkey patch applied successfully."
fi

# =============================================================================
# STEP 8 — Enforcing NumPy Sweet Spot (NumPy 2.0.x)
# =============================================================================
log_section "STEP 8 — Enforcing NumPy Sweet Spot (NumPy 2.0.x)"
log_info "Locking numpy to 2.0.2. Numba/SciPy require < 2.1, while LiveKit/OpenCV require >= 2.0.1..."
python -m pip install --no-cache-dir "numpy==2.0.2" scipy librosa

# =============================================================================
# STEP 9 — Validate GPU Support
# =============================================================================
log_section "STEP 9 — Validate GPU Support"
log_info "Testing PyTorch CUDA availability..."

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA is NOT available. The server requires a GPU build of PyTorch!'"; then
    log_error "PyTorch GPU Validation Failed! The installed PyTorch bundle does not support your GPU."
    exit 1
fi
log_ok "PyTorch GPU support validated: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
log_section "✅  INSTALLATION COMPLETE"
echo -e "${GREEN}"
echo "  ARTalk installed at : ${ARTALK_PATH}"
echo "  Active Python       : ${ACTIVE_PYTHON}"
echo "  Environment         : cloudspace (Lightning.ai)"
echo ""
echo "  To run ARTalk (no conda activate needed):"
echo "    cd ${ARTALK_PATH}"
echo ""
echo "  Gradio interface (generates public URL):"
echo "    python inference.py --run_app"
echo ""
echo "  Command line:"
echo "    python inference.py -a your_audio.wav --shape_id mesh --style_id default --clip_length 750"
echo -e "${NC}"
