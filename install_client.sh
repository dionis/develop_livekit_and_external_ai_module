#!/usr/bin/env bash
# =============================================================================
#  ARTalk LiveKit Client Plugin — Installation Script
#  Installs the lightweight LiveKit agent plugin on the BRAIN machine.
#  No GPU, no model weights, no CUDA extensions required.
#
#  Two target environments are supported:
#    1) Standard Linux  — uv virtual environment (.venv)
#    2) Lightning.ai    — installs into the restricted 'cloudspace' environment
#
#  Usage:
#    chmod +x install_client.sh
#    ./install_client.sh
#
#  Architecture overview:
#    GPU Machine  →  install_server.sh  →  artalk_server/ (FastAPI on :8000)
#    Brain Agent  →  install_client.sh  →  livekit.plugins.artalk (LiveKit agent)
#
#  After installation:
#    Edit .env (LIVEKIT_URL, API keys, ARTALK_SERVER_URL)
#    Run an example: python examples/example_microservice_agent.py start
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

INSTALL_DIR="${PWD}"

# =============================================================================
# SHARED HELPER — Generate .env template
# =============================================================================
generate_client_env() {
    if [ ! -f "${INSTALL_DIR}/.env" ]; then
        log_info "Generating .env template..."
        cat > "${INSTALL_DIR}/.env" <<EOF
# =============================================================
#  ARTalk Client Plugin — Environment Variables
#  Fill in ALL values before starting the agent.
# =============================================================

# --- LiveKit (brain agent connects here) ---
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# --- ARTalk Microservice Server (GPU machine address) ---
# URL of the machine running install_server.sh / artalk_server
ARTALK_SERVER_URL=http://localhost:8000

# --- AI Provider Keys ---
OPENAI_API_KEY=
CARTESIA_API_KEY=

# --- Avatar (microservice mode) ---
# Set to a custom replica_id created via examples/create_avatar_replica.py
# Leave as "mesh" to use the default neutral head model
ARTALK_REPLICA_ID=mesh
EOF
        log_ok ".env template written. Edit it before starting the agent."
    else
        log_warn ".env already exists. Skipping generation."
    fi
}

# =============================================================================
# INSTALLATION PATH A — Standard Linux (uv / venv)
# =============================================================================
install_client_standard() {
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

    # ── Install uv if missing ─────────────────────────────────────────────────
    log_section "STEP 2 (Standard) — Install uv"
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1090
        source "${HOME}/.cargo/env"
        log_ok "uv installed."
    else
        log_ok "uv already installed: $(uv --version)"
    fi

    # ── Sync plugin dependencies into .venv ──────────────────────────────────
    log_section "STEP 3 (Standard) — Install Plugin into .venv"
    log_info "Creating virtual environment and syncing dependencies..."
    uv sync --python "${PYTHON_CMD}"
    log_ok "Plugin dependencies installed in .venv."

    # NOTE: No torch/torchvision/CUDA is installed here.
    # The client plugin communicates over HTTP with the GPU server.

    generate_client_env

    log_section "✅  CLIENT PLUGIN INSTALLATION COMPLETE (Standard)"
    echo -e "${GREEN}"
    echo "  Plugin        : livekit-plugins-artalk"
    echo "  Environment   : .venv"
    echo ""
    echo "  Activate and run:"
    echo "    source .venv/bin/activate"
    echo "    python examples/example_microservice_agent.py start"
    echo ""
    echo "  Pre-requisite: the ARTalk server must be running."
    echo "    On the GPU machine: python examples/start_artalk_server.py"
    echo ""
    echo "  See examples/README.md for full usage instructions."
    echo -e "${NC}"
}

# =============================================================================
# INSTALLATION PATH B — Lightning.ai Studio
# =============================================================================
install_client_lightning() {
    log_section "STEP 1 (Lightning) — Verify Environment"

    ACTIVE_PYTHON=$(which python)
    log_info "Active Python : ${ACTIVE_PYTHON}"
    log_info "Python prefix : $(python -c 'import sys; print(sys.prefix)')"

    if ! conda env list 2>&1 | grep -q "Conda environment management is not allowed"; then
        log_warn "This path is optimised for Lightning.ai Studios."
        log_warn "For standard conda installs, re-run and choose option 1."
        log_warn "Proceeding with pip --system installation anyway..."
    else
        log_ok "Lightning.ai Studio confirmed."
    fi

    # ── Install uv if missing ─────────────────────────────────────────────────
    log_section "STEP 2 (Lightning) — Install uv"
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1090
        source "${HOME}/.cargo/env"
        log_ok "uv installed."
    else
        log_ok "uv already installed: $(uv --version)"
    fi

    # ── Install plugin into system environment ────────────────────────────────
    log_section "STEP 3 (Lightning) — Install Plugin (system)"
    log_info "Installing livekit-plugins-artalk into system environment..."
    uv pip install --system -e .
    log_ok "Plugin installed."

    generate_client_env

    log_section "✅  CLIENT PLUGIN INSTALLATION COMPLETE (Lightning.ai)"
    echo -e "${GREEN}"
    echo "  Plugin        : livekit-plugins-artalk"
    echo "  Environment   : cloudspace (system)"
    echo ""
    echo "  Run (no activation needed):"
    echo "    python examples/example_microservice_agent.py start"
    echo ""
    echo "  Pre-requisite: the ARTalk server must be running."
    echo "    On the GPU machine: python examples/start_artalk_server.py"
    echo ""
    echo "  See examples/README.md for full usage instructions."
    echo -e "${NC}"
}

# =============================================================================
# MAIN — Environment Selection
# =============================================================================
log_section "ARTalk Client Plugin Installer"
echo ""
echo -e "${YELLOW}This script installs the LiveKit brain-agent plugin (no GPU required).${NC}"
echo -e "${YELLOW}For the ARTalk rendering server (GPU machine), use install_server.sh.${NC}"
echo ""
echo -e "${BOLD}Select your target environment:${NC}"
echo "  1) Standard Linux  (uv virtual environment — local / VPS / cloud VM)"
echo "  2) Lightning.ai    (installs into 'cloudspace' — no conda env creation)"
echo ""
read -rp "Enter choice [1-2]: " ENV_CHOICE

case "${ENV_CHOICE}" in
    1)
        install_client_standard
        ;;
    2)
        install_client_lightning
        ;;
    *)
        log_error "Invalid choice '${ENV_CHOICE}'. Run the script again and select 1 or 2."
        exit 1
        ;;
esac
