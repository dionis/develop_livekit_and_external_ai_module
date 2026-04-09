.PHONY: install install-gpu dev run test lint clean \
       docker-build docker-up docker-down docker-logs \
       ditto-build ditto-up ditto-down \
       helm-install helm-uninstall helm-lint \
       download-checkpoints verify

# ─────────────────────────────────────────────
# Local Development
# ─────────────────────────────────────────────

## Install base dependencies (no GPU, service mode)
install:
	uv sync

## Install with GPU support (embedded/local mode)
install-gpu:
	uv sync --extra gpu
	@echo ""
	@echo "✓ GPU dependencies installed"
	@echo "  To install TensorRT 8.6.1, run: make install-tensorrt"

## Install TensorRT (requires GPU deps first)
install-tensorrt:
	uv pip install pip
	uv pip install tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 \
		--index-url https://pypi.nvidia.com \
		--index-strategy unsafe-best-match
	uv pip install tensorrt==8.6.1 \
		--index-url https://pypi.nvidia.com \
		--index-strategy unsafe-best-match \
		--no-build-isolation
	@echo "✓ TensorRT 8.6.1 installed"

## Install dev dependencies
dev:
	uv sync --extra dev

## Run the agent in dev mode
run:
	uv run python examples/basic_agent.py dev

## Run tests
test:
	uv run pytest tests/ -v

## Run linter
lint:
	uv run ruff check .
	uv run black --check .

## Format code
format:
	uv run ruff check --fix .
	uv run black .

## Clean build artifacts
clean:
	rm -rf .venv uv.lock __pycache__ .pytest_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

## Verify installation
verify:
	uv run python -c "from livekit.plugins.ditto import DittoAvatarSession; print('✓ Plugin imports OK')"
	uv run python scripts/verify_ditto_deps.py

# ─────────────────────────────────────────────
# Ditto TalkingHead (Separate Project)
# ─────────────────────────────────────────────

## Clone and setup ditto-talkinghead
setup-ditto:
	bash ditto-service/scripts/install.sh

## Download model checkpoints from HuggingFace
download-checkpoints:
	bash ditto-service/scripts/download_checkpoints.sh

# ─────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────

## Build all Docker images
docker-build:
	docker compose build

## Build only the agent image
docker-build-agent:
	docker build -t ditto-avatar-agent .

## Build only the ditto-service image
docker-build-ditto:
	docker build -t ditto-service ./ditto-service/

## Start full stack (LiveKit + Ditto + Agent + Redis)
docker-up:
	docker compose up -d

## Stop full stack
docker-down:
	docker compose down

## View logs
docker-logs:
	docker compose logs -f

## Start only ditto-service (standalone)
ditto-up:
	docker compose -f ditto-service/docker-compose.yml up -d

## Stop ditto-service
ditto-down:
	docker compose -f ditto-service/docker-compose.yml down

# ─────────────────────────────────────────────
# Kubernetes / Helm
# ─────────────────────────────────────────────

## Lint Helm chart
helm-lint:
	helm lint helm/ditto-avatar/

## Install Helm chart (dry-run)
helm-dry-run:
	helm install ditto-avatar helm/ditto-avatar/ \
		--values helm/ditto-avatar/values.yaml \
		--dry-run --debug

## Install Helm chart
helm-install:
	helm install ditto-avatar helm/ditto-avatar/ \
		--values helm/ditto-avatar/values.yaml \
		--create-namespace --namespace ditto-avatar

## Upgrade Helm release
helm-upgrade:
	helm upgrade ditto-avatar helm/ditto-avatar/ \
		--values helm/ditto-avatar/values.yaml \
		--namespace ditto-avatar

## Uninstall Helm chart
helm-uninstall:
	helm uninstall ditto-avatar --namespace ditto-avatar

## Template Helm chart (for debugging)
helm-template:
	helm template ditto-avatar helm/ditto-avatar/ \
		--values helm/ditto-avatar/values.yaml

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────

## Show help
help:
	@echo "LiveKit Ditto Avatar - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "LOCAL DEVELOPMENT:"
	@echo "  make install           Install base deps (service mode)"
	@echo "  make install-gpu       Install with GPU support (local mode)"
	@echo "  make install-tensorrt  Install TensorRT 8.6.1"
	@echo "  make dev               Install dev tools"
	@echo "  make run               Run agent in dev mode"
	@echo "  make verify            Verify installation"
	@echo "  make test              Run tests"
	@echo "  make lint              Run linter"
	@echo ""
	@echo "DITTO SETUP:"
	@echo "  make setup-ditto            Clone & setup ditto-talkinghead"
	@echo "  make download-checkpoints   Download model checkpoints"
	@echo ""
	@echo "DOCKER:"
	@echo "  make docker-build      Build all images"
	@echo "  make docker-up         Start full stack"
	@echo "  make docker-down       Stop full stack"
	@echo "  make docker-logs       View logs"
	@echo "  make ditto-up          Start ditto-service only"
	@echo ""
	@echo "KUBERNETES:"
	@echo "  make helm-lint         Lint Helm chart"
	@echo "  make helm-install      Install to K8s"
	@echo "  make helm-upgrade      Upgrade release"
	@echo "  make helm-uninstall    Uninstall from K8s"
	@echo "  make helm-dry-run      Dry-run install"
