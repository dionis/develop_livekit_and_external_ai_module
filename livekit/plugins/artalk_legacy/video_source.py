"""
ARTalk Video Source

Manages WebRTC video frame generation and publishing to a LiveKit Room.
Converts generated CV2/Numpy buffers from the SDK into valid WebRTC Track data.
"""

import asyncio
import logging
from typing import Optional
import numpy as np
from livekit import rtc

logger = logging.getLogger(__name__)


class ARTalkVideoSource:
    """
    Manages custom VideoSource proxy for ARTalk avatar frames.
    
    Receives matrices containing ARTalk frames, resizes or color-converts them if required,
    and publishes them natively into the associated LiveKit track format.
    """
    
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: int = 25,
    ):
        """
        Initialize the video source format.
        
        Args:
            width: Standard broadcast video width in pixels.
            height: Standard broadcast video height in pixels.
            fps: Desired broadcast frame rate limit.
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self._source: Optional[rtc.VideoSource] = None
        self._track: Optional[rtc.LocalVideoTrack] = None
        self._is_publishing = False
        
        logger.info(f"Initialized ARTalkVideoSource: {width}x{height} @ {fps}fps")
    
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
            
            # Map standard RGB to expected LiveKit BGRA format 
            # WebRTC/LiveKit traditionally prefers BGRA/ARGB memory structures 
            # Raw python memory tobytes() on RGB without padding misaligns the stride matrix (causing stripes)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                
            frame_contig = np.ascontiguousarray(frame)
            
            # Make the RTC VideoFrame
            video_frame = rtc.VideoFrame(
                self.width,
                self.height,
                rtc.VideoBufferType.BGRA,
                frame_contig.tobytes()
            )
            
            # Fire frame data into stream output
            if getattr(self, "_frame_count_debug", 0) % 50 == 0:
                logger.debug(f"DEBUG VIDEO: Publishing frame heartbeat. Format={video_frame.type}, Size={len(video_frame.data)}")
            self._frame_count_debug = getattr(self, "_frame_count_debug", 0) + 1
            self._source.capture_frame(video_frame)
            
        except ImportError as e:
            logger.error("Missing dependency 'cv2' for resizing frames.")
            raise e
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
