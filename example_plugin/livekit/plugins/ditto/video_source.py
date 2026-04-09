"""
Ditto Video Source

Manages video frame generation and publishing to LiveKit.
"""

import asyncio
import logging
from typing import Optional
import numpy as np
from livekit import rtc

logger = logging.getLogger(__name__)


class DittoVideoSource:
    """
    Manages video source for Ditto avatar frames.
    
    This class handles the conversion of Ditto-generated frames to LiveKit
    video frames and manages the video publishing pipeline.
    """
    
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: int = 25,
    ):
        """
        Initialize the video source.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self._source: Optional[rtc.VideoSource] = None
        self._track: Optional[rtc.LocalVideoTrack] = None
        self._is_publishing = False
        
        logger.info(f"Initialized DittoVideoSource: {width}x{height} @ {fps}fps")
    
    def create_source(self) -> rtc.VideoSource:
        """Create and return a LiveKit video source."""
        if self._source is None:
            self._source = rtc.VideoSource(
                width=self.width,
                height=self.height,
            )
            logger.debug("Created LiveKit VideoSource")
        
        return self._source
    
    def create_track(self, track_name: str = "ditto-avatar-video") -> rtc.LocalVideoTrack:
        """
        Create a video track from the source.
        
        Args:
            track_name: Name for the video track
            
        Returns:
            LocalVideoTrack instance
        """
        if self._source is None:
            self.create_source()
        
        self._track = rtc.LocalVideoTrack.create_video_track(
            track_name,
            self._source
        )
        
        logger.info(f"Created video track: {track_name}")
        return self._track
    
    async def publish_frame(self, frame: np.ndarray) -> None:
        """
        Publish a single frame to the video source.
        
        Args:
            frame: Numpy array representing the frame (HxWxC, RGB or RGBA)
        """
        if self._source is None:
            raise RuntimeError("Video source not created. Call create_source() first.")
        
        try:
            # Ensure frame is in the correct format
            if frame.shape[:2] != (self.height, self.width):
                logger.warning(
                    f"Frame size mismatch: expected {self.height}x{self.width}, "
                    f"got {frame.shape[0]}x{frame.shape[1]}"
                )
                # Resize if needed
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert to RGBA if needed
            if frame.shape[2] == 3:  # RGB
                frame = np.dstack([frame, np.full((self.height, self.width), 255, dtype=np.uint8)])
            
            # Create VideoFrame
            video_frame = rtc.VideoFrame(
                width=self.width,
                height=self.height,
                type=rtc.VideoBufferType.RGBA,
                data=frame.tobytes()
            )
            
            # Capture frame to source
            await self._source.capture_frame(video_frame)
            
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
            raise
    
    async def publish_frames(self, frames: list[np.ndarray]) -> None:
        """
        Publish multiple frames sequentially.
        
        Args:
            frames: List of numpy arrays representing frames
        """
        frame_delay = 1.0 / self.fps
        
        for frame in frames:
            await self.publish_frame(frame)
            await asyncio.sleep(frame_delay)
    
    def start_publishing(self) -> None:
        """Mark that publishing has started."""
        self._is_publishing = True
        logger.info("Started video publishing")
    
    def stop_publishing(self) -> None:
        """Mark that publishing has stopped."""
        self._is_publishing = False
        logger.info("Stopped video publishing")
    
    @property
    def is_publishing(self) -> bool:
        """Check if currently publishing."""
        return self._is_publishing
    
    @property
    def source(self) -> Optional[rtc.VideoSource]:
        """Get the video source."""
        return self._source
    
    @property
    def track(self) -> Optional[rtc.LocalVideoTrack]:
        """Get the video track."""
        return self._track
