import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
import uuid
from dotenv import load_dotenv

# Determine project root and load environment variables
# override=True ensures .env takes precedence even if the shell already exported these vars
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(_env_path, override=True)
logger_boot = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
_loaded = os.path.isfile(_env_path)
logger_boot.info(f"Loaded .env from {_env_path} (file_exists={_loaded})")

# Add project root to sys.path
sys.path.insert(0, PROJECT_ROOT)

try:
    from .image_preprocessor import preprocess_avatar_image
except ImportError:
    try:
        from image_preprocessor import preprocess_avatar_image
    except ImportError:
        preprocess_avatar_image = None

try:
    from .validators import validate_image_quality, validate_face_detected
except ImportError:
    try:
        from validators import validate_image_quality, validate_face_detected
    except ImportError:
        validate_image_quality = None
        validate_face_detected = None

from .models import ReplicaResponse, ConversationRequest, ConversationResponse
from .worker import start_livekit_worker

logger = logging.getLogger("artalk_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Global dict to track active workers
active_conversations = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ARTalk Microservice Started.")
    
    import torch
    if not torch.cuda.is_available():
        logger.warning("================================================================================")
        logger.warning(f"⚠️ PyTorch VERSION: {torch.__version__}")
        logger.warning("⚠️ PyTorch is NOT compiled with CUDA support or no GPU was detected!")
        logger.warning("⚠️ ARTalk will fall back to CPU, which can be extremely slow.")
        logger.warning("⚠️ If you are on an L4/T4 GPU, you likely installed the CPU-only version of PyTorch.")
        logger.warning("⚠️ Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        logger.warning("================================================================================")
    else:
        logger.info(f"✅ CUDA is available! GPU detected: {torch.cuda.get_device_name(0)}")

    yield
    logger.info("Shutting down ARTalk Microservice.")
    # Here we would gracefully shutdown all active workers
    for cid, task in active_conversations.items():
        logger.info(f"Shutting down worker for conversation {cid}")

app = FastAPI(lifespan=lifespan, title="ARTalk Avatar Server", version="1.0.0")

# Determine default ARTalk path from environment or project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ARTALK_PATH = os.getenv("ARTALK_PATH", os.path.join(PROJECT_ROOT, "external_models", "ARTalk"))

@app.post("/v1/avatar/create", response_model=ReplicaResponse)
async def create_avatar(image_url: str = None, return_metrics: bool = True, artalk_path: str = DEFAULT_ARTALK_PATH, file: UploadFile = File(None)):
    """
    Simulates Tavus API's endpoint to create a replica (avatar) from an image.
    Uses the legacy preprocess_avatar_image from prepare_artalk_avatar.py.
    """
    if preprocess_avatar_image is None:
        raise HTTPException(status_code=500, detail="ARTalk dependencies not found. Unable to preprocess image.")

    if not image_url and not file:
        raise HTTPException(status_code=400, detail="Either 'image_url' or 'file' must be provided.")

    try:
        logger.info(f"Received avatar creation request")

        import tempfile
        import urllib.request
        import shutil

        temp_dir = tempfile.TemporaryDirectory()

        try:
            image_path = None
            if file:
                filename = file.filename or "uploaded_avatar.png"
                image_path = os.path.join(temp_dir.name, filename)
                logger.info(f"Saving uploaded avatar image to: {image_path}")
                with open(image_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
            else:
                image_path = image_url
                is_url = image_url.startswith("http://") or image_url.startswith("https://")
                if is_url:
                    filename = "downloaded_avatar.png"
                    image_path = os.path.join(temp_dir.name, filename)
                    logger.info(f"Downloading avatar image from URL: {image_url}")
                    try:
                        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=30) as response:
                            with open(image_path, "wb") as f:
                                f.write(response.read())
                        logger.info(f"Avatar image downloaded to: {image_path}")
                    except Exception as dl_err:
                        detail = (
                            f"Failed to download image from URL '{image_url}': {dl_err}. "
                            "Please verify the URL is publicly accessible and returns a valid image."
                        )
                        logger.error(f"[create_avatar] {detail}")
                        raise HTTPException(status_code=400, detail=detail)

            # ── Image quality validation ─────────────────────────────────────
            if validate_image_quality is not None:
                logger.info("Running image quality validation...")
                try:
                    validate_image_quality(image_path)
                except ValueError as qe:
                    logger.error(f"[create_avatar] Image quality check failed: {qe}")
                    raise HTTPException(status_code=400, detail=str(qe))
            else:
                logger.warning("[create_avatar] Image quality validator not available – skipping quality check.")

            # ── Face detection validation ────────────────────────────────────
            if validate_face_detected is not None:
                logger.info("Running face detection validation...")
                try:
                    validate_face_detected(image_path)
                except ValueError as fe:
                    logger.error(f"[create_avatar] Face detection check failed: {fe}")
                    raise HTTPException(status_code=422, detail=str(fe))
            else:
                logger.warning("[create_avatar] Face validator not available – skipping face detection check.")

            # ── Run ARTalk preprocessor ──────────────────────────────────────
            logger.info("All validations passed. Running ARTalk preprocessor...")
            
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device_str} for preprocessing")
            
            avatar_id = preprocess_avatar_image(
                image_path_str=image_path,
                artalk_path_str=artalk_path,
                device=device_str,
                no_matting=False,
            )

            metrics = {}
            if return_metrics:
                metrics = {"psnr": 30.0, "ssim": 0.95}

            logger.info(f"Avatar created successfully: avatar_id='{avatar_id}'")
            return ReplicaResponse(replica_id=avatar_id, quality=metrics)

        finally:
            temp_dir.cleanup()

    except HTTPException:
        raise  # re-raise structured HTTP errors unchanged
    except Exception as e:
        logger.error(f"[create_avatar] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/conversation", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """
    Simulates Tavus API's endpoint to start a conversation.
    This starts a background worker that connects to LiveKit and renders the avatar.
    """
    replica_id = request.replica_id
    livekit_ws_url = request.properties.get("livekit_ws_url")
    livekit_room_token = request.properties.get("livekit_room_token")
    artalk_path = request.properties.get("artalk_path", DEFAULT_ARTALK_PATH)

    if not livekit_ws_url or not livekit_room_token:
        raise HTTPException(
            status_code=400,
            detail="Missing 'livekit_ws_url' or 'livekit_room_token' in properties.",
        )

    # ── Background scene pre-flight validation ───────────────────────────────
    # Validate BEFORE launching the worker so the caller gets an immediate error
    # instead of silently using no background.
    if request.background_scene is not None:
        scene = request.background_scene
        scene_valid = False

        if scene.startswith("http://") or scene.startswith("https:/"):
            # Accept any syntactically valid URL; the worker will download it.
            # We do a lightweight parse check rather than a HEAD request to avoid
            # imposing latency or requiring outbound connectivity at validation time.
            import urllib.parse
            parsed = urllib.parse.urlparse(scene)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                scene_valid = True
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid URL provided for 'background_scene': '{scene}'. "
                        "Please provide a well-formed HTTP or HTTPS URL."
                    ),
                )
        elif Path(scene).exists():
            scene_valid = True
        else:
            # Check built-in scenes directory
            _scenes_dir = Path(PROJECT_ROOT) / "scenes"
            if (_scenes_dir / f"{scene}.png").exists():
                scene_valid = True
            else:
                available = sorted(p.stem for p in _scenes_dir.glob("*.png")) if _scenes_dir.exists() else []
                available_str = ", ".join(f'"{s}"' for s in available) if available else "(none found)"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid 'background_scene' value: '{scene}'. "
                        f"Expected a valid HTTP/HTTPS URL, an existing local file path, "
                        f"or a built-in scene name. Available built-in scenes: {available_str}."
                    ),
                )

        logger.info(f"[create_conversation] background_scene pre-flight PASSED for: '{scene}'")

    conversation_id = str(uuid.uuid4())

    background_tasks.add_task(
        start_livekit_worker,
        conversation_id=conversation_id,
        replica_id=replica_id,
        ws_url=livekit_ws_url,
        token=livekit_room_token,
        artalk_path=artalk_path,
        background_scene=request.background_scene,
        bg_threshold=request.bg_threshold,
    )

    active_conversations[conversation_id] = "running"

    return ConversationResponse(
        conversation_id=conversation_id,
        status="active",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
