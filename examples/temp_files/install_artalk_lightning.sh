#!/usr/bin/env bash
# =============================================================================
#  ARTalk — Installation Script (Lightning.ai Studio)
#  Target: Lightning.ai Studios — installs into the default 'cloudspace' env
#  Based on: https://github.com/xg-chu/ARTalk
#  Includes known fixes for common build and import errors
#
#  NOTE: Lightning.ai restricts conda to a single environment ('cloudspace').
#  This script installs all packages via pip directly into that environment.
#  Do NOT run this script on standard Linux/Anaconda setups.
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

# =============================================================================
# STEP 3 — Fix DiagnosticOptions (known error with onnx2torch)
# =============================================================================
log_section "STEP 3 — Detect and fix onnx/torch conflict"

DIAG_ERROR=0
TORCH_REINSTALLED=0
python -c "from torch.onnx._internal.exporter import DiagnosticOptions" 2>/dev/null || DIAG_ERROR=1

if [ "${DIAG_ERROR}" -eq 1 ]; then
    log_warn "Detected 'DiagnosticOptions' error — onnx2torch incompatible with current PyTorch."
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
# STEP 3c — Install pytorch3d
# =============================================================================
log_section "STEP 3c — Install pytorch3d"

# pytorch3d is a Facebook Research CUDA extension not listed in environment.yml.
# It must be compiled from source. --no-build-isolation is required on Lightning.ai
# for the same reason as diff-gaussian-rasterization: the isolated build environment
# created by pip has no access to torch, causing ModuleNotFoundError during setup.

PYTORCH3D_OK=0
python -c "import pytorch3d" 2>/dev/null && PYTORCH3D_OK=1 || true

# If torch was reinstalled above, pytorch3d's compiled CUDA extensions are now
# invalid even though the module is still "installed". Force a full rebuild.
if [ "${PYTORCH3D_OK}" -eq 1 ] && [ "${TORCH_REINSTALLED}" -eq 1 ]; then
    log_warn "torch was reinstalled — pytorch3d CUDA extensions are now stale. Forcing rebuild..."
    pip uninstall -y pytorch3d 2>/dev/null || true
    PYTORCH3D_OK=0
fi

if [ "${PYTORCH3D_OK}" -eq 1 ]; then
    PYTORCH3D_VER=$(python -c "import pytorch3d; print(pytorch3d.__version__)" 2>/dev/null || echo "unknown")
    log_ok "pytorch3d==${PYTORCH3D_VER} is already installed. Skipping."
else
    log_info "pytorch3d not found. Installing from source (this may take 10-20 minutes)..."

    # Strategy 1: try a pre-built wheel matching the detected torch/CUDA/Python versions
    # Wheels are hosted at https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/
    PY_TAG="py$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
    TORCH_TAG="pyt$(echo "${TORCH_VER}" | tr -d '.')"
    WHEEL_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PY_TAG}_${CUDA_TAG}_${TORCH_TAG}/download.html"

    log_info "Trying pre-built wheel: ${PY_TAG} + ${CUDA_TAG} + ${TORCH_TAG}..."
    pip install --no-cache-dir --no-build-isolation \
        "pytorch3d" \
        -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PY_TAG}_${CUDA_TAG}_${TORCH_TAG}/download.html" \
        2>/dev/null && log_ok "pytorch3d installed from pre-built wheel." || \

    # Strategy 2: build from source via git (slow but always works)
    ( log_warn "Pre-built wheel not available. Building pytorch3d from source..."
      log_warn "This can take 10-20 minutes. Please be patient..."
      pip install --no-cache-dir --no-build-isolation \
          "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
      log_ok "pytorch3d built and installed from source." ) || \

    { log_error "pytorch3d installation failed. Check CUDA/torch version compatibility."; exit 1; }

    # Final verification
    python -c "import pytorch3d; print('pytorch3d version:', pytorch3d.__version__)" \
        || { log_error "pytorch3d installed but cannot be imported."; exit 1; }
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

log_info "Installing diff-gaussian-rasterization (this may take several minutes)..."
cd "${DIFF_GAUSS_DIR}"
# --no-build-isolation is required on Lightning.ai: pip would otherwise create an
# isolated temp environment for the build that has no access to torch from cloudspace,
# causing "ModuleNotFoundError: No module named 'torch'" during wheel preparation.
pip install --no-cache-dir --no-build-isolation --force-reinstall .
log_ok "diff-gaussian-rasterization installed successfully."

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
