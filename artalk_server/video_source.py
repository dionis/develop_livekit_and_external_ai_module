import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
from livekit import rtc

logger = logging.getLogger(__name__)


class ARTalkVideoSource:
    """
    Manages custom VideoSource proxy for ARTalk avatar frames.
    
    Receives matrices containing ARTalk frames, resizes or color-converts them if required,
    and publishes them natively into the associated LiveKit track format.
    Supports dynamic background replacement using MediaPipe.
    """
    
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: int = 25,
        background_scene: Optional[str] = None,
        bg_threshold: int = 15,
    ):
        """
        Initialize the video source format.
        
        Args:
            width: Standard broadcast video width in pixels.
            height: Standard broadcast video height in pixels.
            fps: Desired broadcast frame rate limit.
            background_scene: Name of a scene (e.g., 'office') or path to a background image.
            bg_threshold: Luminosity cutoff (0-255) to separate avatar from black background.
                          Used only when MediaPipe is unavailable (cv2_fallback mode).

                          Tuning guide:
                            3-8   → Very tight. Keeps maximum dark detail (hair/shadows), but
                                    may leave thin black halos around edges.
                            10-20 → Recommended range. Good balance between edge quality and
                                    avatar detail. Default is 15.
                            25-40 → Aggressive. Cleaner edges but can erode dark zones like
                                    hair or a dark jacket.
                            >40   → Too aggressive. Avatar body parts start disappearing.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self._bg_threshold = max(1, min(255, bg_threshold))
        
        self._source: Optional[rtc.VideoSource] = None
        self._track: Optional[rtc.LocalVideoTrack] = None
        self._is_publishing = False
        
        # Background and Segmentation
        self._bg_image: Optional[np.ndarray] = None
        self._segmenter = None
        
        if background_scene:
            self._load_background(background_scene)
            self._init_segmenter()
        
        logger.info(f"Initialized ARTalkVideoSource: {width}x{height} @ {fps}fps | Background: {background_scene} | Threshold: {self._bg_threshold}")

    def _load_background(self, scene: str) -> None:
        """
        Resolve and load the background image from one of three accepted sources,
        evaluated in this priority order:

          1. HTTP/HTTPS URL  — image is downloaded to a temporary file and loaded.
          2. Local file path — an absolute or relative path that exists on the filesystem.
          3. Scene name      — a short name resolved against the project's ``scenes/``
                               directory (e.g. "office" → scenes/office.png).

        Parameters
        ----------
        scene : str
            URL, absolute/relative file path, or built-in scene name.

        Raises
        ------
        ValueError
            If ``scene`` cannot be resolved by any of the above methods.
            The error message is in English and safe to return to API callers.
        """
        try:
            import cv2

            bg_path = None

            # ── 1. HTTP/HTTPS URL ────────────────────────────────────────────
            if scene.startswith("http://") or scene.startswith("https://"):
                import tempfile
                import urllib.request

                logger.info(f"Downloading background image from URL: {scene}")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp_path = tmp.name
                    req = urllib.request.Request(scene, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        with open(tmp_path, "wb") as f:
                            f.write(resp.read())
                    bg_path = Path(tmp_path)
                    logger.info(f"Background image downloaded to temporary file: {tmp_path}")
                except Exception as download_err:
                    msg = (
                        f"Failed to download background image from URL '{scene}': {download_err}. "
                        "Please verify the URL is publicly accessible and points to a valid image."
                    )
                    logger.error(f"[_load_background] {msg}")
                    raise ValueError(msg)

            # ── 2. Local file path ───────────────────────────────────────────
            elif Path(scene).exists():
                bg_path = Path(scene)
                logger.info(f"Loading background from local file path: {bg_path}")

            # ── 3. Built-in scene name ───────────────────────────────────────
            else:
                project_root = Path(__file__).parent.parent
                scenes_dir = project_root / "scenes"
                candidate = scenes_dir / f"{scene}.png"
                if candidate.exists():
                    bg_path = candidate
                    logger.info(f"Loading built-in scene '{scene}' from {bg_path}")
                else:
                    # List available scenes to help the caller pick a valid one
                    available = sorted(p.stem for p in scenes_dir.glob("*.png")) if scenes_dir.exists() else []
                    available_str = ", ".join(f'"{s}"' for s in available) if available else "(none found)"
                    msg = (
                        f"Background scene '{scene}' could not be resolved. "
                        f"Expected one of the following: a valid HTTP/HTTPS URL, an existing local "
                        f"file path, or a built-in scene name. "
                        f"Available built-in scenes: {available_str}."
                    )
                    logger.error(f"[_load_background] {msg}")
                    raise ValueError(msg)

            # ── Load and resize ──────────────────────────────────────────────
            bg_img = cv2.imread(str(bg_path))
            if bg_img is None:
                msg = (
                    f"Background image at '{bg_path}' could not be read by OpenCV. "
                    "The file may be corrupted or in an unsupported format."
                )
                logger.error(f"[_load_background] {msg}")
                raise ValueError(msg)

            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            self._bg_image = cv2.resize(bg_img, (self.width, self.height))
            logger.info(f"Background image loaded and resized to {self.width}x{self.height}: {bg_path}")

        except ValueError:
            raise  # propagate our structured errors as-is
        except Exception as e:
            msg = f"Unexpected error while loading background '{scene}': {e}"
            logger.error(f"[_load_background] {msg}")
            raise ValueError(msg)


    def _init_segmenter(self):
        """Initializes MediaPipe Selfie Segmentation or falls back to OpenCV."""
        try:
            import mediapipe as mp
            # Force load solutions if the top-level package doesn't expose it
            if not hasattr(mp, 'solutions'):
                import mediapipe.python.solutions as mp_solutions
                mp.solutions = mp_solutions
            
            self._mp_selfie = mp.solutions.selfie_segmentation
            self._segmenter = self._mp_selfie.SelfieSegmentation(model_selection=1)
            logger.info("MediaPipe Selfie Segmentation initialized.")
        except Exception as e:
            logger.warning(f"MediaPipe failed to load '{e}'. Falling back to OpenCV FloodFill Matting.")
            self._segmenter = "cv2_fallback"

    
    def compose_and_publish(self, frame_rgb: np.ndarray) -> None:
        """
        Apply background composition and publish a frame synchronously.
        This is meant to be called from the worker's frame loop (non-async context).
        
        Args:
            frame_rgb: Numpy array in RGB format (HxWx3)
        """
        import cv2

        if self._source is None:
            return

        # Resize if needed
        frame = frame_rgb
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Apply background composition
        if self._segmenter is not None and self._bg_image is not None:
            frame_rgb_input = frame

            mask_3d = None

            if self._segmenter == "cv2_fallback":
                    # Brightness threshold approach: ARTalk renders on pure black (0,0,0).
                    # Any pixel with meaningful brightness is part of the avatar.
                    # This is faster than FloodFill and doesn't leak into dark avatar regions (hair, shadows).
                    gray = cv2.cvtColor(frame_rgb_input, cv2.COLOR_RGB2GRAY)
                    # Threshold: pixels brighter than self._bg_threshold are avatar, rest is background.
                    # See __init__ docstring for tuning guide.
                    _, fg_mask = cv2.threshold(gray, self._bg_threshold, 255, cv2.THRESH_BINARY)

                    # Step 1 — OPEN (erode→dilate): removes small isolated bright blobs
                    # (e.g. the single bright specular pixel visible in the hair area)
                    open_kernel = np.ones((5, 5), np.uint8)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel)

                    # Step 2 — CLOSE (dilate→erode): fills small dark holes inside the avatar
                    # (prevents the background scene from showing through dark zones like hair)
                    close_kernel = np.ones((9, 9), np.uint8)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)

                    # Step 3 — Soften the silhouette boundary for natural blending
                    fg_mask = cv2.GaussianBlur(fg_mask, (9, 9), 0)
                    mask_float = fg_mask.astype(float) / 255.0
                    mask_3d = np.stack([mask_float] * 3, axis=-1)
            else:
                results = self._segmenter.process(frame_rgb_input)
                if results.segmentation_mask is not None:
                    mask_float = results.segmentation_mask
                    mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)
                    mask_3d = np.stack([mask_float] * 3, axis=-1)

            if mask_3d is not None:
                composite = (frame_rgb_input.astype(float) * mask_3d +
                             self._bg_image.astype(float) * (1.0 - mask_3d))
                frame = composite.astype(np.uint8)

        # Convert to BGRA and publish
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        frame_contig = np.ascontiguousarray(frame_bgra)
        frame_bytes = frame_contig.tobytes()
        
        video_frame = rtc.VideoFrame(
            self.width,
            self.height,
            rtc.VideoBufferType.BGRA,
            frame_bytes
        )
        
        # Keep the bytes object alive so Python's Garbage Collector does not free it 
        # before the LiveKit tracking FFI thread finishes encoding the video frame.
        if not hasattr(self, "_video_frame_refs"):
            import collections
            self._video_frame_refs = collections.deque(maxlen=10)
        self._video_frame_refs.append(frame_bytes)
        
        self._source.capture_frame(video_frame)

    
    def create_source(self) -> rtc.VideoSource:
        """Create and return a raw LiveKit video source buffer."""
        if self._source is None:
            self._source = rtc.VideoSource(
                width=self.width,
                height=self.height,
            )
            logger.debug("Created internal LiveKit VideoSource for ARTalk")
        
        return self._source
    
    def create_track(self, track_name: str = "artalk-avatar-video") -> rtc.LocalVideoTrack:
        """
        Create a track encapsulating the underlying source payload.
        
        Args:
            track_name: Label used for client-side subscription.
            
        Returns:
            Publisher's LocalVideoTrack wrapper.
        """
        if self._source is None:
            self.create_source()
        
        self._track = rtc.LocalVideoTrack.create_video_track(
            track_name,
            self._source
        )
        
        logger.info(f"Created video publication track: {track_name}")
        return self._track
    
    async def publish_frame(self, frame: np.ndarray) -> None:
        """
        Push a solitary frame block into the WebRTC stream.
        
        Args:
            frame: Numpy 3D array representing the pixel layout (HxWxC, RGB or RGBA)
        """
        if self._source is None:
            raise RuntimeError("Video source not created. Attempting to publish without a track source.")
        
        try:
            import cv2
            
            # Force target resolution bounds mapping
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Handle background composition if enabled
            if self._segmenter is not None and self._bg_image is not None:
                # Ensure input is RGB for composition (ARTalk output is RGB)
                if frame.shape[2] == 4:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                else:
                    frame_rgb = frame
                
                mask_3d = None
                
                if self._segmenter == "cv2_fallback":
                    # OpenCV FloodFill approach: Assumes background is contiguous black from top corners
                    h, w = frame_rgb.shape[:2]
                    mask = np.zeros((h + 2, w + 2), np.uint8)
                    flags = (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                    
                    process_frame = np.ascontiguousarray(frame_rgb)
                    # Fill from top-left and top-right (tolerance 25 to account for slight rendering noise)
                    cv2.floodFill(process_frame, mask, (0, 0), 255, (25, 25, 25), (25, 25, 25), flags)
                    cv2.floodFill(process_frame, mask, (w - 1, 0), 255, (25, 25, 25), (25, 25, 25), flags)
                    
                    bg_mask = mask[1:h+1, 1:w+1]
                    # Invert: 255 for person, 0 for background
                    fg_mask = cv2.bitwise_not(bg_mask)
                    # Soften the edges to prevent hard aliasing
                    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
                    mask_float = fg_mask.astype(float) / 255.0
                    mask_3d = np.stack([mask_float] * 3, axis=-1)
                else:
                    # MediaPipe approach
                    results = self._segmenter.process(frame_rgb)
                    if results.segmentation_mask is not None:
                        mask_float = results.segmentation_mask
                        mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)
                        mask_3d = np.stack([mask_float] * 3, axis=-1)
                
                if mask_3d is not None:
                    # Composite: Alpha blending -> Result = Avatar * Mask + Background * (1 - Mask)
                    composite = (frame_rgb.astype(float) * mask_3d + 
                                self._bg_image.astype(float) * (1.0 - mask_3d))
                    frame = composite.astype(np.uint8)

            # Map standard RGB to expected LiveKit BGRA format 
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                
            frame_contig = np.ascontiguousarray(frame)
            frame_bytes = frame_contig.tobytes()
            
            # Make the RTC VideoFrame
            video_frame = rtc.VideoFrame(
                self.width,
                self.height,
                rtc.VideoBufferType.BGRA,
                frame_bytes
            )
            
            # Keep the bytes object alive
            if not hasattr(self, "_video_frame_refs"):
                import collections
                self._video_frame_refs = collections.deque(maxlen=10)
            self._video_frame_refs.append(frame_bytes)
            
            # Fire frame data into stream output
            if getattr(self, "_frame_count_debug", 0) % 50 == 0:
                logger.debug(f"DEBUG VIDEO: Publishing frame heartbeat. Background={'ENABLED' if self._bg_image is not None else 'OFF'}")
            
            self._frame_count_debug = getattr(self, "_frame_count_debug", 0) + 1
            self._source.capture_frame(video_frame)
            
        except Exception as e:
            logger.error(f"Error publishing frame format: {e}")
            raise

    
    def start_publishing(self) -> None:
        """Flip status marker to active."""
        self._is_publishing = True
        logger.info("Began stream relay into LiveKit.")
    
    def stop_publishing(self) -> None:
        """Flip status marker to closed."""
        self._is_publishing = False
        logger.info("Halted stream relay.")
    
    @property
    def is_publishing(self) -> bool:
        """Check broadcast status."""
        return self._is_publishing
    
    @property
    def source(self) -> Optional[rtc.VideoSource]:
        """Access underlying buffer payload struct."""
        return self._source
    
    @property
    def track(self) -> Optional[rtc.LocalVideoTrack]:
        """Access high-level publishing track instance."""
        return self._track
