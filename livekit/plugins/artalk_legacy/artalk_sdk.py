"""
ARTalk SDK Wrapper

This module provides a wrapper around the external ARTalk repository,
treating it as an external resource that can be loaded and managed independently 
for generating Gaussian Splatting video frames synced with audio.

Model Load Strategies
---------------------
FROM_SCRATCH    – load each component (BitwiseARModel, GAGAvatar, FLAME) individually
                  from checkpoint files.  Current default; gives fine-grained control.
EXAMPLE_MODELS  – delegate to ARTalk's own ``ARTAvatarInferEngine`` class, which is the
                  same class used by the ARTalk Gradio demo (inference.py).  Use this when
                  you want to rely on ARTalk's upstream load logic without changes.
"""

import enum
import sys
import queue
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Silence Numba and force NVIDIA library path for Cloudspace environments
import os
import glob
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    # Look for NVIDIA libraries inside site-packages (typical for Lightning AI / Cloudspace)
    nvidia_libs = glob.glob(f"{conda_prefix}/lib/python*/site-packages/nvidia/*/lib")
    if nvidia_libs:
        existing_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_path = ":".join(nvidia_libs)
        os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{existing_path}"
        logger.debug(f"DEBUG SDK: Injected LD_LIBRARY_PATH with {len(nvidia_libs)} NVIDIA folders")

logging.getLogger("numba").setLevel(logging.WARNING)


class ModelLoadStrategy(str, enum.Enum):
    """
    Controls how ``ARTalkSDKWrapper`` loads the model pipeline.

    Attributes
    ----------
    FROM_SCRATCH:
        Load each component of the pipeline individually from checkpoint files.
        This is the default and provides the finest-grained control over every
        sub-model.  Requires that ``assets/ARTalk_wav2vec.pt`` (or ``_mimi.pt``)
        exists inside the ARTalk repository.
    EXAMPLE_MODELS:
        Delegate model loading to ARTalk's own ``ARTAvatarInferEngine``, the
        class used by the official Gradio demo (``inference.py`` in the ARTalk
        repository).  Useful when you want to mirror the upstream reference
        implementation exactly without maintaining a custom loading path.
    """
    FROM_SCRATCH = "from_scratch"
    EXAMPLE_MODELS = "example_models"


class ARTalkSDKWrapper:
    """
    Wrapper for the ARTalk generation SDK as an external resource.
    
    This class handles loading the ARTalk module from an external path,
    managing its internal pipeline and gathering generated Gaussian splatting frames.
    """
    
    def __init__(
        self,
        artalk_path: str,
        shape_id: str = "mesh",
        style_id: str = "default",
        max_queue_size: int = 100,
        model_strategy: ModelLoadStrategy = ModelLoadStrategy.FROM_SCRATCH,
        load_gaga: bool = True,
    ):
        """
        Initialize the ARTalk SDK wrapper.
        
        Args:
            artalk_path: Path to the root directory of the ARTalk installation.
            shape_id: Target avatar shape or tracked object (.pt or mesh baseline).
            style_id: Name of standard speaking style or motion baseline.
            max_queue_size: Max number of pre-generated frames to hold in queue.
            model_strategy: Controls how the model pipeline is loaded.
                ``ModelLoadStrategy.FROM_SCRATCH`` (default) loads each component
                individually.  ``ModelLoadStrategy.EXAMPLE_MODELS`` delegates to
                ARTalk's own ``ARTAvatarInferEngine`` (the Gradio demo class).
            load_gaga: When ``model_strategy=EXAMPLE_MODELS``, whether to also
                load the GAGAvatar Gaussian-splatting renderer.  Has no effect for
                the ``FROM_SCRATCH`` strategy (GAGAvatar is always attempted there).
        """
        self.artalk_path = Path(artalk_path).absolute()
        self.shape_id = shape_id
        self.style_id = style_id
        self.model_strategy = ModelLoadStrategy(model_strategy)  # accept str or enum
        self.load_gaga = load_gaga
        self._is_loaded = False
        self._frame_queue = queue.Queue(maxsize=max_queue_size)
        self._audio_queue = queue.Queue(maxsize=max_queue_size)
        
        # Core ARTalk components (populated by whichever load strategy runs)
        self.hubert = None
        self.flame = None
        self.model = None
        self.diff_renderer = None
        self.style_embedding = None
        self._infer_engine = None  # only used by EXAMPLE_MODELS strategy
        
        # Variables for periodic blinking
        self._frame_count = 0
        self._next_blink_frame = 100 # Delayed to avoid start-up spasm
        self._blink_active = False
        self._blink_start_frame = 0
        self._blink_duration = 0
        
        # Stochastic Animation States (for impatience/alive feel)
        self._life_t = 0.0 # Decoupled time for continuous noise
        self._saccade_active = False
        self._saccade_start_frame = 0
        self._saccade_duration = 0
        self._saccade_offsets = np.zeros(6) # RX, RY, RZ for head and eyes
        self._next_saccade_frame = 150 # Delayed to give a neutral intro
        
        # Impatience behaviors
        self._sigh_active = False
        self._sigh_start_frame = 0
        self._double_blink_pending = False
        self._last_reset_id = 0
        
        # Audio buffering for streaming
        self._audio_buffer = np.array([], dtype=np.float32)
        self._audio_bytes_buffer = b""
        
        logger.info(
            f"Initializing ARTalk SDK wrapper | path={self.artalk_path} "
            f"| strategy={self.model_strategy.value}"
        )
        
    def load(self, device: str = "cuda") -> None:
        """
        Load the ARTalk model pipeline.

        The actual load path depends on ``self.model_strategy``:

        * ``ModelLoadStrategy.FROM_SCRATCH``  — loads each sub-model individually
          from checkpoint files (original behaviour).
        * ``ModelLoadStrategy.EXAMPLE_MODELS`` — delegates to ARTalk's own
          ``ARTAvatarInferEngine`` (the class used by the official Gradio demo).

        Args:
            device: PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.
        """
        if self._is_loaded:
            logger.warning("ARTalk SDK already loaded")
            return

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "ARTalk requires a CUDA-capable GPU but none was found. "
                "Use device='cpu' if you must run without a GPU (not recommended)."
            )

        # Add the ARTalk repo root to sys.path so its internal packages are importable
        artalk_dir = str(self.artalk_path)
        if artalk_dir not in sys.path:
            sys.path.insert(0, artalk_dir)
            logger.debug(f"Added {artalk_dir} to sys.path")

        try:
            if self.model_strategy == ModelLoadStrategy.EXAMPLE_MODELS:
                logger.info("Loading via EXAMPLE_MODELS strategy (ARTAvatarInferEngine)...")
                self._load_example_models(device)
            else:
                logger.info("Loading via FROM_SCRATCH strategy (individual components)...")
                self._load_from_scratch(device)

            self._is_loaded = True
            self.device = device
            logger.info("Warming up ARTalk models (Triggering Numba/PyTorch JIT compiling)...")
            
            # WARMUP inference to save GIL starvation later
            dummy_audio = np.zeros(16000, dtype=np.float32)
            dummy_bytes = (dummy_audio * 32768.0).astype(np.int16).tobytes()
            self.process_audio_chunk(dummy_audio, 16000, dummy_bytes, is_silence=True)
            self.reset_buffers()
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            logger.info("Warmup complete. ARTalk SDK loaded successfully.")
            logger.info("!!!! ARTalk SDK REBOOTED - VERSION 3.3.3 - CLEAN STARTUP !!!!")

        except ImportError as e:
            logger.error(f"Failed to import ARTalk modules: {e}")
            raise RuntimeError(
                f"Could not load ARTalk from '{self.artalk_path}'. "
                "Ensure the path points to a cloned https://github.com/xg-chu/ARTalk repo."
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialise ARTalk SDK: {e}")
            raise

    # ------------------------------------------------------------------
    # Private load strategies
    # ------------------------------------------------------------------

    def _load_from_scratch(self, device: str) -> None:
        """
        Load each component of the ARTalk pipeline individually from checkpoint
        files.  This is the original implementation and gives fine-grained control
        over every sub-model.
        """
        orig_cwd = os.getcwd()
        os.chdir(str(self.artalk_path))
        try:
            import json
            from app.models import BitwiseARModel
            from app.GAGAvatar import GAGAvatar
            from app.flame_model import FLAMEModel

            # ── JSON config ───────────────────────────────────────────
            config_path = self.artalk_path / "assets" / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"ARTalk config not found: {config_path}. "
                    "Run build_resources.sh inside the ARTalk repo first."
                )
            with open(config_path, "r") as _f:
                model_cfg = json.load(_f)

            # ── Resolve audio encoder from available checkpoint ───────
            _wav2vec_ckpt = self.artalk_path / "assets" / "ARTalk_wav2vec.pt"
            _mimi_ckpt    = self.artalk_path / "assets" / "ARTalk_mimi.pt"
            if _wav2vec_ckpt.exists():
                ar_ckpt_path = _wav2vec_ckpt
                model_cfg["AR_CONFIG"]["AUDIO_ENCODER"] = "wav2vec"
            elif _mimi_ckpt.exists():
                ar_ckpt_path = _mimi_ckpt
                model_cfg["AR_CONFIG"]["AUDIO_ENCODER"] = "mimi"
            else:
                raise FileNotFoundError(
                    f"No ARTalk checkpoint found in {self.artalk_path / 'assets'}. "
                    "Run build_resources.sh to download model weights."
                )
            logger.info(
                f"Audio encoder: {model_cfg['AR_CONFIG']['AUDIO_ENCODER']} "
                f"(ckpt: {ar_ckpt_path.name})"
            )

            # ── BitwiseARModel ────────────────────────────────────────
            logger.info("Loading BitwiseARModel...")
            self.model = BitwiseARModel(model_cfg=model_cfg).to(device)
            _ar_ckpt = torch.load(str(ar_ckpt_path), map_location=device, weights_only=True)
            self.model.load_state_dict(_ar_ckpt.get("model", _ar_ckpt))
            self.model.eval()
            logger.info("BitwiseARModel loaded.")

            # ── GAGAvatar renderer ────────────────────────────────────
            logger.info("Loading GAGAvatar renderer...")
            self.gagavatar = GAGAvatar().to(device)
            self.gagavatar.eval()
            self.gagavatar.set_avatar_id(self.shape_id)
            logger.info(f"GAGAvatar ready (shape_id='{self.shape_id}').")

            # ── FLAME 3D face model ───────────────────────────────────
            self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0).to(device)
            self.flame_model.eval()
            self.gagavatar_flame = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)
            self.gagavatar_flame.eval()
            
            # Monkey patch watermark for from_scratch strategy
            self.gagavatar.add_water_mark = lambda image: image
            
            logger.info("FLAME models loaded.")

            # ── Style motion (optional) ───────────────────────────────
            _style_dir = self.artalk_path / "assets" / "style_motion"
            _style_file = _style_dir / f"{self.style_id}.pt"
            if not _style_file.exists():
                _candidates = list(_style_dir.glob("*.pt"))
                _style_file = _candidates[0] if _candidates else None
                if _style_file:
                    logger.warning(
                        f"Style '{self.style_id}' not found, using '{_style_file.name}'"
                    )
            if _style_file and _style_file.exists():
                self.style_embedding = torch.load(
                    str(_style_file), map_location=device, weights_only=True
                )
                logger.info(f"Style motion loaded: {_style_file.name}")
            else:
                self.style_embedding = None
                logger.info("No style motion loaded; using null style condition.")

        finally:
            os.chdir(orig_cwd)

    def _load_example_models(self, device: str) -> None:
        """
        Delegate model loading to ARTalk's own ``ARTAvatarInferEngine``, which is the
        class used by the official Gradio demo (``inference.py`` in the ARTalk repo).

        This strategy is controlled by ``model_strategy=ModelLoadStrategy.EXAMPLE_MODELS``
        and mirrors the upstream reference implementation exactly.
        """
        orig_cwd = os.getcwd()
        os.chdir(str(self.artalk_path))
        try:
            # ARTalk's inference.py lives at the repo root and defines ARTAvatarInferEngine.
            # sys.path already has the artalk root, so the import works without further surgery.
            from inference import ARTAvatarInferEngine  # type: ignore[import]

            logger.info(
                f"Instantiating ARTAvatarInferEngine "
                f"(load_gaga={self.load_gaga}, device={device})..."
            )
            engine = ARTAvatarInferEngine(
                load_gaga=self.load_gaga,
                fix_pose=False,
                clip_length=750,
                device=device,
            )

            # ── Remove watermark automatically ────────────────────────
            if self.load_gaga and hasattr(engine, "GAGAvatar"):
                engine.GAGAvatar.add_water_mark = lambda image: image
                logger.info("Watermark monkey-patch applied to GAGAvatar.")

            # ── Apply speaking style ──────────────────────────────────
            if self.style_id and self.style_id != "default":
                _style_path = (
                    self.artalk_path / "assets" / "style_motion" / f"{self.style_id}.pt"
                )
                if _style_path.exists():
                    engine.set_style_motion(self.style_id)
                    logger.info(f"Style motion set to '{self.style_id}'.")
                else:
                    logger.warning(
                        f"Style '{self.style_id}' not found – using default null style."
                    )

            # ── Store engine and bridge references ────────────────────
            self._infer_engine = engine
            # Expose the same attributes the FROM_SCRATCH path would set
            # so that the rest of the class can treat both paths uniformly.
            self.model = engine.ARTalk
            self.flame_model = engine.flame_model
            if self.load_gaga and hasattr(engine, "GAGAvatar"):
                self.gagavatar = engine.GAGAvatar
                # Unified access to the GAGAvatar specific FLAME model
                self.gagavatar_flame = getattr(engine, "GAGAvatar_flame", None)

            logger.info("ARTAvatarInferEngine (EXAMPLE_MODELS) ready.")

        finally:
            os.chdir(orig_cwd)
    
    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        audio_bytes: bytes = b"",
        is_silence: bool = False
    ) -> None:
        """
        Process an audio chunk and queue it for ARTalk video generation.
        
        Takes the raw float PCM audio buffer, aggregates it if necessary, 
        extracts features, and passes to the renderer.
        
        Args:
            audio_chunk: Audio data as numpy array (float32).
            sample_rate: The sample rate of the provided audio array.
            audio_bytes: Corresponding raw PCM bytes to sync perfectly.
            is_silence: Flag to dictate if the frames are pure silence for idle animation.
        """
        if not self._is_loaded:
            raise RuntimeError("ARTalk SDK not loaded. Call load() first.")
        
        try:
            # Aggregate audio buffer for context if the model requires chunks
            # of specific duration (e.g. 50Hz features = 320 samples at 16k)
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
            self._audio_bytes_buffer += audio_bytes
            
            # Capture current reset ID to detect if buffers are cleared while we render
            current_reset_id = self._last_reset_id
            
            # Example windowing: process if we have at least ~0.4s of audio
            min_samples = int(0.4 * sample_rate)
            
            if len(self._audio_buffer) >= min_samples:
                process_buffer = self._audio_buffer
                process_bytes = self._audio_bytes_buffer
                # Keep a 0.2s overlap for continuous feature extraction and smoothing
                overlap = int(0.2 * sample_rate) 
                overlap_bytes = int(0.2 * sample_rate * 2) # 16-bit mono 
                
                # Check if we have overlap from previous chunks to discard
                has_overlap = getattr(self, "_has_overlap", False)
                discard_frames = int((overlap / sample_rate) * 25.0) if has_overlap else 0
                discard_bytes = overlap_bytes if has_overlap else 0
                
                if len(self._audio_buffer) > overlap:
                    self._audio_buffer = self._audio_buffer[-overlap:]
                    self._audio_bytes_buffer = self._audio_bytes_buffer[-overlap_bytes:]
                else:
                    self._audio_buffer = np.array([], dtype=np.float32)
                    self._audio_bytes_buffer = b""
                
                self._generate_frames(process_buffer, process_bytes, sample_rate, discard_past_frames=discard_frames, discard_past_bytes=discard_bytes, is_silence=is_silence, reset_id=current_reset_id)
                self._has_overlap = True
            
        except Exception as e:
            logger.error(f"Failed to process audio chunk through ARTalk: {e}")

    def flush_audio(self, sample_rate: int = 16000) -> None:
        """
        Process any residual audio left in the buffer.
        Called when the TTS stream ends to ensure the end of words are not clipped.
        """
        if not self._is_loaded or len(self._audio_buffer) == 0:
            return
            
        try:
            process_buffer = self._audio_buffer
            process_bytes = self._audio_bytes_buffer
            
            # Check overlap from previous continuous chunks
            has_overlap = getattr(self, "_has_overlap", False)
            overlap = int(0.2 * sample_rate)
            overlap_bytes = int(0.2 * sample_rate * 2)
            
            discard_frames = int((overlap / sample_rate) * 25.0) if has_overlap else 0
            discard_bytes = overlap_bytes if has_overlap else 0
            
            # Capture current reset ID
            current_reset_id = self._last_reset_id
            
            self._generate_frames(process_buffer, process_bytes, sample_rate, discard_past_frames=discard_frames, discard_past_bytes=discard_bytes, is_silence=False, reset_id=current_reset_id)
            
            # Clear buffers after flush
            self._audio_buffer = np.array([], dtype=np.float32)
            self._audio_bytes_buffer = b""
            self._has_overlap = False
            
        except Exception as e:
            logger.error(f"Error flushing ARTalk: {e}")

    def generate_idle_frames(self, num_frames: int = 25, reset_id: int = -1) -> None:
        """
        Isolated method to generate idle 'alive' frames without touching the audio buffer.
        """
        if not self._is_loaded or self.model is None:
            return

        # BUG FIX: If no reset_id provided, use the current active one.
        # Otherwise, 1 > 0 check would abort idle generation after first speech.
        if reset_id == -1:
            reset_id = self._last_reset_id

        # NEW: Ensure correct directory context for GAGAvatar assets during idle rendering
        orig_cwd = os.getcwd()
        os.chdir(str(self.artalk_path))
        
        try:
            # We don't bother with inference for idle. 
            # We create neutral motions and apply behavioral patterns.
            motion_dim = self.model.basic_vae.motion_dim
            pred_motions = torch.zeros((num_frames, motion_dim), device=self.device)
            
            # 1. Strictly zero expressions (preserving LID indices only) and jaw PRIOR to animations
            # ABSOLUTE ZEROING of all potential bleeding indices.
            pred_motions[..., 0:19] = 0.0 
            pred_motions[..., 21:100] = 0.0
            pred_motions[..., 103:] = 0.0
            
            # 2. Apply animations (Blinks, Jitter, etc.)
            self._apply_behavioral_animation(pred_motions)
            
            # Render and queue
            if self.shape_id != "mesh" and self.load_gaga and hasattr(self, "gagavatar"):
                flame_instance = getattr(self, "gagavatar_flame", self.flame_model)
                self.gagavatar.set_avatar_id(self.shape_id)
                
                # IDLE BOOST: Generate 25 frames (1 second) to avoid frequency gaps
                logger.debug(f"DEBUG SDK: Rendering 25 IDLE frames (GAGAvatar)...")
                for i, motion in enumerate(pred_motions):
                    if self._last_reset_id > reset_id:
                        return
                    
                    batch = self.gagavatar.build_forward_batch(
                        motion[None], flame_instance
                    )
                    rgb_tensor = self.gagavatar.forward_expression(batch)
                    frame_chw = rgb_tensor[0].cpu().numpy()
                    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
                    frame_rgb = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                    
                    # Heartbeat
                    frame_rgb[0,0,0] = (self._frame_count + i) % 256
                    
                    if not self._frame_queue.full() and self._last_reset_id <= reset_id:
                        self._frame_queue.put((frame_rgb, b"", True))
                    
                    if i % 5 == 0 or i == num_frames - 1:
                        logger.debug(f"DEBUG SDK: Rendered IDLE frame {i+1}/{num_frames} (Val: {frame_rgb[0,0,0]})")
            else:
                # Mesh fallback
                logger.debug(f"DEBUG SDK: Rendering {num_frames} IDLE frames (Mesh)...")
                shape_code = torch.zeros((1, 300), device=self.device).expand(num_frames, -1)
                verts = self.model.basic_vae.get_flame_verts(
                    self.flame_model, shape_code, pred_motions, with_global=True
                )
                
                if self.diff_renderer is None:
                    from app.flame_model import RenderMesh
                    self.diff_renderer = RenderMesh(
                        n_vertices=self.flame_model.v_template.shape[0],
                        device=self.device,
                        batch_size=1,
                        render_original=True,
                    )

                for v in verts:
                    if self._last_reset_id > reset_id:
                        return
                    rgb_tensor = self.diff_renderer(v[None])[0]
                    frame_chw = (rgb_tensor.cpu() / 255.0).numpy()
                    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
                    frame_rgb = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                    if not self._frame_queue.full() and self._last_reset_id <= reset_id:
                        self._frame_queue.put((frame_rgb, b"", True))
                        
        except Exception as e:
            logger.error(f"Error in generate_idle_frames: {e}", exc_info=True)
        finally:
            os.chdir(orig_cwd)

    def _apply_behavioral_animation(self, pred_motions):
        """
        Injects stochastic 'behavioral' motions for an impatient/alive feel.
        Refined for smoothness and sub-perception.
        """
        for motion in pred_motions:
            # 1. Update Global Clock
            self._frame_count += 1
            t = self._frame_count / 25.0

            # 2. Multi-Frequency 'Organic' Noise (Head)
            # CALIBRATED: Scale 1.5 for visible organic movement in Gaussian.
            scale = 1.5
            p_noise = (np.sin(t * 0.73) * 0.012 + np.sin(t * 1.57) * 0.006) * scale
            y_noise = (np.cos(t * 0.61) * 0.018 + np.sin(t * 1.13) * 0.010) * scale
            r_noise = (np.sin(t * 0.47) * 0.006 + np.cos(t * 2.29) * 0.004) * scale

            # 3. Nervous Jitter (Impatience High-Frequency Tremor)
            jitter_p = np.sin(t * 12.0) * 0.002
            jitter_y = np.cos(t * 11.0) * 0.002

            # 4. Saccadic Jumps + Periodic Sighs - SLOWED DOWN
            if not self._saccade_active and not self._sigh_active and self._frame_count >= self._next_saccade_frame:
                if np.random.rand() < 0.1: # Reduced sigh chance
                    self._sigh_active = True
                    self._sigh_start_frame = self._frame_count
                else:
                    self._saccade_active = True
                    self._saccade_start_frame = self._frame_count
                    self._saccade_duration = np.random.randint(6, 12) 
                    self._saccade_offsets = np.random.uniform(-0.03, 0.03, size=3) # Subtler offsets

            saccade_p, saccade_y, saccade_r = 0.0, 0.0, 0.0
            
            if self._saccade_active:
                progress = self._frame_count - self._saccade_start_frame
                if progress < self._saccade_duration:
                    f = 1.0 - (1.0 - progress / self._saccade_duration)**4 # Ease-out quartic
                    saccade_p, saccade_y, saccade_r = self._saccade_offsets * f
                else:
                    self._saccade_active = False
                    self._next_saccade_frame = self._frame_count + np.random.randint(100, 250) # Longer delay

            # Sigh Logic: Softer vertical movement
            if self._sigh_active:
                progress = self._frame_count - self._sigh_start_frame
                if progress < 20: 
                    f = progress / 20.0
                    sigh_val = np.sin(np.pi * f) * 0.15 # Smoother sine-based sigh
                    saccade_p += sigh_val
                else:
                    self._sigh_active = False
                    self._next_saccade_frame = self._frame_count + 100

            # 5. Apply Results to Head Pos (100-102)
            if torch.is_tensor(motion):
                motion[100] += float(p_noise + jitter_p + saccade_p)
                motion[101] += float(y_noise + jitter_y + saccade_y)
                motion[102] += float(r_noise + saccade_r)
            else:
                motion[100] += float(p_noise + jitter_p + saccade_p)
                motion[101] += float(y_noise + jitter_y + saccade_y)
                motion[102] += float(r_noise + saccade_r)

            # 6. Eye Micro-Motion + Nervous Blinking - SLOWED DOWN
            eye_jitter = np.sin(t * 3.0) * 0.04 # Much slower eye movement (Simulates slow gaze shift)
            if motion.shape[0] > 117:
                if torch.is_tensor(motion):
                    vlat, vlon = float(eye_jitter), float(eye_jitter * 0.6)
                    motion[112] += vlat; motion[115] += vlat
                    motion[113] += vlon; motion[116] += vlon
                else:
                    vlat, vlon = float(eye_jitter), float(eye_jitter * 0.6)
                    motion[112] += vlat; motion[115] += vlat
                    motion[113] += vlon; motion[116] += vlon

            # 7. Blinking logic (Indices 19 & 20)
            if not self._blink_active and self._frame_count >= self._next_blink_frame:
                self._blink_active = True
                self._blink_start_frame = self._frame_count
                self._blink_duration = np.random.randint(6, 12)
                self._double_blink_pending = (np.random.rand() < 0.2)
                logger.debug(f"DEBUG SDK: BLINK START at frame {self._frame_count}")

            if self._blink_active:
                progress = self._frame_count - self._blink_start_frame
                if progress < self._blink_duration:
                    intensity = np.sin(np.pi * (progress + 0.5) / self._blink_duration)
                    # CALIBRATED: 10.0 is ULTRA closure for strictly surgical indices.
                    # Wider Plateau: Multiply by 3.0 and clip to 1.0.
                    peak_factor = np.clip(intensity * 3.0, 0.0, 1.0)
                    v = float(peak_factor * 10.0)
                    
                    if torch.is_tensor(motion):
                        # PURE LID ONLY: Only 19 and 20. NO bleeding possible.
                        for idx in [19, 20]:
                            motion[idx] = max(float(motion[idx]), v)
                    else:
                        for idx in [19, 20]:
                            motion[idx] = max(motion[idx], v)
                else:
                    self._blink_active = False
                    if self._double_blink_pending:
                        self._next_blink_frame = self._frame_count + 3
                        self._double_blink_pending = False
                    else:
                        self._next_blink_frame = self._frame_count + np.random.randint(50, 150)
                    logger.debug(f"DEBUG SDK: BLINK END at frame {self._frame_count}")

        # 8. FINAL SAFETY: Clamp all motions to prevent deformation (Safe Range ±11.0 for 3.3.2)
        if torch.is_tensor(pred_motions):
            pred_motions.clamp_(-11.0, 11.0)

        return pred_motions

    def _generate_frames(self, audio_data: np.ndarray, audio_bytes: bytes, sample_rate: int, discard_past_frames: int = 0, discard_past_bytes: int = 0, is_silence: bool = False, reset_id: int = 0) -> None:
        """
        Internal method to convert audio data to visual frames using ARTalk 
        and push them to the output queue.
        """
        # NEW: Hardened threshold to 0.02 to avoid background noise triggering ghost-mouth
        max_audio_amp = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
        is_chunk_silent = bool(max_audio_amp < 0.02) 
        effective_silence = is_silence or is_chunk_silent

        if not is_silence and not is_chunk_silent:
            logger.debug(f"DEBUG SDK: _generate_frames (SPEECH PATH). amp={max_audio_amp:.4f}, chunk_silent={is_chunk_silent}")
        elif not is_silence:
            # Ghost detection
            logger.debug(f"DEBUG SDK: _generate_frames (SILENT TAIL). amp={max_audio_amp:.4f}, forcing silence logic.")
        
            # Abort check: if a reset happened BEFORE we even started, stop.
            if self._last_reset_id > reset_id:
                logger.debug(f"Generation aborted before start (ID {reset_id} < {self._last_reset_id})")
                return
        
        # Change dir again for relative asset loading that happens during inference
        orig_cwd = os.getcwd()
        os.chdir(str(self.artalk_path))
        
        try:
            with torch.no_grad():
                # Convert numpy float32 audio to torch tensor expected by ARTalk
                audio_tensor = torch.from_numpy(audio_data).float()

                # Safety check against Savitzky-Golay exception in temp_inference.py 
                # (window_length=9 requirement means we need at least 9 * 320 samples) 
                # We enforce minimum length of 0.25s inside the inference loop to skip bad flush chunks.
                min_infer_samples = int(0.25 * sample_rate)
                if len(audio_data) < min_infer_samples:
                    logger.debug(f"Audio chunk too small for ARTalk inference ({len(audio_data)} samples). Skipping to avoid Savitzky-Golay crash.")
                    return

                if self.model_strategy == ModelLoadStrategy.EXAMPLE_MODELS and self._infer_engine:
                    # 1. Predict FLAME motions using the inference engine
                    if effective_silence:
                        # SILENCE OPTIMIZATION: Don't call the heavy inference model.
                        # Create neutral motions directly.
                        # GAGAvatar/ARTalk expects approx 25fps.
                        num_frames = int(len(audio_data) / (sample_rate / 25))
                        if num_frames == 0: num_frames = 1
                        # Use model's native motion dimension (usually 106)
                        motion_dim = self._infer_engine.ARTalk.basic_vae.motion_dim
                        pred_motions = torch.zeros((num_frames, motion_dim), device=self.device)
                    else:
                        # ARTAvatarInferEngine.inference expects [1, T] audio
                        pred_motions = self._infer_engine.inference(audio_tensor)
                        
                        # Add official Savitzky-Golay smoothing here for EXAMPLE_MODELS path
                        from scipy.signal import savgol_filter
                        motion_np = pred_motions.clone().detach().cpu().numpy()
                        if len(motion_np) >= 9:
                            motion_np = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
                            motion_np[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=9, polyorder=3, axis=0)
                        pred_motions = torch.tensor(motion_np).type_as(pred_motions)
                    
                    if discard_past_frames > 0 and len(pred_motions) > discard_past_frames:
                        pred_motions = pred_motions[discard_past_frames:]
                    
                    # Silence Attenuation — Ensure neutral pose during silence.
                    # We do this anyway in case of slight noise.
                    if effective_silence:
                        # NEW: Kill ALL speech-driven motion to ensure clean stop
                        pred_motions.zero_() 

                    # Apply periodic blink and idle injection (adds noise to 100-102 and 19-20)
                    self._apply_behavioral_animation(pred_motions)

                    # 2. Render frames based on whether it's GAGAvatar or Mesh
                    if self.shape_id != "mesh" and self.load_gaga and hasattr(self._infer_engine, "GAGAvatar"):
                        self._infer_engine.GAGAvatar.set_avatar_id(self.shape_id)
                        
                        audio_idx = 0
                        audio_per_frame = int(sample_rate * 2 / 25) # bytes per frame at 25 fps
                        usable_audio = audio_bytes[discard_past_bytes:] if discard_past_bytes > 0 else audio_bytes
                        
                        logger.debug(f"DEBUG SDK: Rendering {len(pred_motions)} frames in GAGAvatar mode...")
                        for i, motion in enumerate(pred_motions):
                            # Abort check: if a reset happened during this batch, stop immediately
                            if self._last_reset_id > reset_id:
                                logger.debug(f"Generation aborted by reset_buffers (ID {reset_id} < {self._last_reset_id})")
                                return

                            # 2. Render frame
                            batch = self._infer_engine.GAGAvatar.build_forward_batch(
                                motion[None], self._infer_engine.GAGAvatar_flame
                            )
                            rgb_tensor = self._infer_engine.GAGAvatar.forward_expression(batch)
                            # Tensor is [1, 3, 512, 512] RGB float in [0, 1] range
                            frame_chw = rgb_tensor[0].cpu().numpy()
                            frame_hwc = np.transpose(frame_chw, (1, 2, 0))
                            frame_rgb = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                            
                            # HEARTBEAT PIXEL: Change top-left pixel to ensure frame uniqueness
                            # This bypasses any potential compression/caching deduplication
                            frame_rgb[0, 0, 0] = (self._frame_count + i) % 256

                            frame_audio = usable_audio[audio_idx : audio_idx + audio_per_frame]
                            audio_idx += audio_per_frame
                            
                            if not self._frame_queue.full():
                                self._frame_queue.put((frame_rgb, frame_audio, is_silence))
                            
                            if i % 10 == 0:
                                logger.debug(f"DEBUG SDK: Rendered frame {i}/{len(pred_motions)}")
                    else:
                        # Mesh rendering fallback from inference engine
                        shape_code = audio_tensor.new_zeros(1, 300).to(self.device).expand(pred_motions.shape[0], -1)
                        verts = self._infer_engine.ARTalk.basic_vae.get_flame_verts(
                            self._infer_engine.flame_model, shape_code, pred_motions, with_global=True
                        )
                        
                        audio_idx = 0
                        audio_per_frame = int(sample_rate * 2 / 25) # bytes per frame at 25 fps
                        usable_audio = audio_bytes[discard_past_bytes:] if discard_past_bytes > 0 else audio_bytes
                        
                        for v in verts:
                            if self._last_reset_id > reset_id:
                                logger.debug(f"Generation aborted (SDK Mesh loop ID {reset_id} < {self._last_reset_id}).")
                                return
                            import time
                            time.sleep(0.001) # Yield GIL
                            rgb_tensor = self._infer_engine.mesh_renderer(v[None])[0]
                            # Tensor is [3, 512, 512] RGB float [0, 255]
                            frame_chw = (rgb_tensor.cpu() / 255.0).numpy()
                            frame_hwc = np.transpose(frame_chw, (1, 2, 0))
                            frame_rgb = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                            
                            frame_audio = usable_audio[audio_idx : audio_idx + audio_per_frame]
                            audio_idx += audio_per_frame
                            
                            if not self._frame_queue.full():
                                self._frame_queue.put((frame_rgb, frame_audio, is_silence))
                            else:
                                logger.warning("Dropping generated ARTalk frame: queue full.")
                else:
                    # 1. Inference logic
                    if effective_silence:
                        num_frames = int(len(audio_data) / (sample_rate / 25))
                        if num_frames == 0: num_frames = 1
                        motion_dim = self.model.basic_vae.motion_dim
                        pred_motions = torch.zeros((num_frames, motion_dim), device=self.device)
                    else:
                        style_motion_t = self.style_embedding[None].to(self.device) if self.style_embedding is not None else None
                        audio_batch = {'audio': audio_tensor[None].to(self.device), 'style_motion': style_motion_t}
                        pred_motions = self.model.inference(audio_batch, with_gtmotion=False)
                        if isinstance(pred_motions, tuple):
                            pred_motions = pred_motions[0]
                        
                        # inference returns a batched tensor of shape (B, Seq, Feat)
                        if pred_motions.dim() == 3:
                            pred_motions = pred_motions[0]
                    
                    # Apply Savitzky-Golay smoothing (only for real speech to avoid noise artifacts)
                    if not effective_silence:
                        from scipy.signal import savgol_filter
                        motion_np = pred_motions.clone().detach().cpu().numpy()
                        if len(motion_np) >= 9:
                            motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
                            motion_np_smoothed[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=9, polyorder=3, axis=0)
                        else:
                            motion_np_smoothed = motion_np
                        pred_motions = torch.tensor(motion_np_smoothed).type_as(pred_motions)
                    
                    if discard_past_frames > 0 and len(pred_motions) > discard_past_frames:
                        pred_motions = pred_motions[discard_past_frames:]
                    
                    # Fix global pose + disable expression blendshapes after 104
                    pred_motions[..., 104:] *= 0.0

                    # Silence Attenuation: ensure neutral pose during silence
                    if effective_silence:
                        pred_motions.zero_()

                    # Apply periodic blink and idle injection
                    self._apply_behavioral_animation(pred_motions)

                    if self.shape_id != "mesh":
                        # GAGAvatar custom image rendering
                        
                        audio_idx = 0
                        audio_per_frame = int(sample_rate * 2 / 25) # bytes per frame at 25 fps
                        usable_audio = audio_bytes[discard_past_bytes:] if discard_past_bytes > 0 else audio_bytes
                        
                        for motion in pred_motions:
                            if self._last_reset_id > reset_id:
                                logger.debug(f"Generation aborted (FROM_SCRATCH ID {reset_id} < {self._last_reset_id}).")
                                return

                            import time
                            time.sleep(0.001) # Yield GIL
                            batch = self.gagavatar.build_forward_batch(
                                motion[None], self.gagavatar_flame
                            )
                            rgb_tensor = self.gagavatar.forward_expression(batch)
                            frame_chw = rgb_tensor[0].cpu().numpy()
                            frame_hwc = np.transpose(frame_chw, (1, 2, 0))
                            frame_rgb = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                            
                            frame_audio = usable_audio[audio_idx : audio_idx + audio_per_frame]
                            audio_idx += audio_per_frame
                            
                            if not self._frame_queue.full():
                                self._frame_queue.put((frame_rgb, frame_audio, is_silence))
                            else:
                                logger.warning("Dropping generated ARTalk frame: queue full.")
                    else:
                        # Fallback for Mesh rendering from_scratch
                        # Instantiate the mesh renderer here if not already available
                        from app.flame_model import RenderMesh
                        if self.diff_renderer is None:
                            self.diff_renderer = RenderMesh(image_size=512, faces=self.flame_model.get_faces(), scale=1.0).to(self.device)
                            
                        shape_code = audio_tensor.new_zeros(1, 300).to(self.device).expand(pred_motions.shape[0], -1)
                        verts = self.model.basic_vae.get_flame_verts(
                            self.flame_model, shape_code, pred_motions, with_global=True
                        )
                        
                        audio_idx = 0
                        audio_per_frame = int(sample_rate * 2 / 25) # bytes per frame at 25 fps
                        usable_audio = audio_bytes[discard_past_bytes:] if discard_past_bytes > 0 else audio_bytes
                        
                        for v_idx, v in enumerate(verts):
                            if self._last_reset_id > reset_id:
                                logger.debug(f"Generation aborted (FROM_SCRATCH MESH ID {reset_id} < {self._last_reset_id}).")
                                return

                            import time
                            time.sleep(0.001) # Yield GIL
                            rgb_tensor = self.diff_renderer(v[None])[0]
                            frame_chw = (rgb_tensor.cpu() / 255.0).numpy()
                            frame_hwc = np.transpose(frame_chw, (1, 2, 0))
                            frame_rgb = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                            
                            frame_audio = usable_audio[audio_idx : audio_idx + audio_per_frame]
                            audio_idx += audio_per_frame
                            
                            if not self._frame_queue.full():
                                self._frame_queue.put((frame_rgb, frame_audio, is_silence))
                            else:
                                logger.warning("Dropping generated ARTalk frame: queue full.")
            
        except Exception as e:
            import traceback
            logger.error(f"Error generating frames in ARTalk: {e}\n{traceback.format_exc()}")
        finally:
            os.chdir(orig_cwd)

    def update_style(self, new_style_id: str) -> None:
        """
        Dynamically attempt to update the ARTalk speaking style.
        """
        if not self._is_loaded:
            return
        
        self.style_id = new_style_id
        
        style_path = os.path.join(str(self.artalk_path), "assets", "style_motion", f"{self.style_id}.pt")
        if os.path.exists(style_path):
            self.style_embedding = torch.load(style_path, map_location=self.device)
            logger.info(f"Updated ARTalk speaking style to {new_style_id}")
        else:
            logger.warning(f"Style file not found: {style_path}")
    
    def get_frame_queue(self) -> queue.Queue:
        """Access the thread-safe queue holding generated CV2 frames."""
        if not self._is_loaded:
            raise RuntimeError("ARTalk SDK not loaded. Call load() first.")
        
        return self._frame_queue

    def get_audio_queue(self) -> queue.Queue:
        """Access the thread-safe queue holding audio bytes chunks."""
        if not self._is_loaded:
            raise RuntimeError("ARTalk SDK not loaded. Call load() first.")
        return self._audio_queue
        
    def reset_buffers(self) -> None:
        """Clear out audio context and frame queues to immediately transition state."""
        self._last_reset_id += 1
        self._frame_count = 0  # NEW: Reset animation clock to zero
        self._next_blink_frame = 10 # NEW: Force blink soon after speech
        self._audio_buffer = np.array([], dtype=np.float32)
        self._audio_bytes_buffer = b""
        self._has_overlap = False
        
        # Drain the frame queue to stop 'ghost' talking after speech ends
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Counter increment above kills any active _generate_frames threads
        
    def queue_audio(self, audio_bytes: bytes) -> None:
        """Push a processed audio chunk into the sync queue."""
        if not self._is_loaded:
            raise RuntimeError("ARTalk SDK not loaded. Call load() first.")
        if not self._audio_queue.full():
            self._audio_queue.put(audio_bytes)
        else:
            logger.warning("Dropping audio chunk: queue full.")


    def close(self) -> None:
        """Close the SDK wrapper and clear resources."""
        if self._is_loaded:
            logger.info("Releasing ARTalk SDK resources")
            self.model = None
            self.diff_renderer = None
            self.style_embedding = None
            self._infer_engine = None  # release EXAMPLE_MODELS engine if present
            self._is_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Drain the frame queue
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the ARTalk tools are loaded into memory."""
        return self._is_loaded
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
