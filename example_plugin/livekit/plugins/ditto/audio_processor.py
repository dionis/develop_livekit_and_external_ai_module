"""
Ditto Audio Processor

Handles audio stream processing for Ditto avatar generation.
"""

import asyncio
import logging
from typing import Optional, AsyncIterator
import numpy as np
from livekit import rtc

logger = logging.getLogger(__name__)


class DittoAudioProcessor:
    """
    Processes audio streams for Ditto avatar generation.
    
    This class handles audio buffering, resampling, and conversion
    to the format required by Ditto SDK.
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_size_ms: int = 200,
    ):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate: Target sample rate for Ditto (16kHz)
            chunk_size_ms: Size of audio chunks in milliseconds
        """
        self.target_sample_rate = target_sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.chunk_size_samples = int(target_sample_rate * chunk_size_ms / 1000)
        
        self._audio_buffer = []
        self._buffer_lock = asyncio.Lock()
        
        logger.info(
            f"Initialized DittoAudioProcessor: {target_sample_rate}Hz, "
            f"{chunk_size_ms}ms chunks ({self.chunk_size_samples} samples)"
        )
    
    async def process_audio_frame(self, frame: rtc.AudioFrame) -> Optional[np.ndarray]:
        """
        Process a single audio frame from LiveKit.
        
        Args:
            frame: LiveKit AudioFrame
            
        Returns:
            Numpy array of audio data if chunk is complete, None otherwise
        """
        try:
            # Convert frame data to numpy array
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
            
            # Resample if needed
            if frame.sample_rate != self.target_sample_rate:
                audio_data = self._resample(
                    audio_data,
                    from_rate=frame.sample_rate,
                    to_rate=self.target_sample_rate
                )
            
            # Add to buffer
            async with self._buffer_lock:
                self._audio_buffer.extend(audio_data)
                
                # Check if we have enough samples for a chunk
                if len(self._audio_buffer) >= self.chunk_size_samples:
                    # Extract chunk
                    chunk = np.array(
                        self._audio_buffer[:self.chunk_size_samples],
                        dtype=np.float32
                    )
                    
                    # Remove processed samples from buffer
                    self._audio_buffer = self._audio_buffer[self.chunk_size_samples:]
                    
                    # Normalize to [-1, 1]
                    chunk = chunk / 32768.0
                    
                    return chunk
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return None
    
    def _resample(
        self,
        audio: np.ndarray,
        from_rate: int,
        to_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio data
            from_rate: Source sample rate
            to_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if from_rate == to_rate:
            return audio
        
        try:
            import librosa
            audio_float = audio.astype(np.float32) / 32768.0
            resampled = librosa.resample(
                audio_float,
                orig_sr=from_rate,
                target_sr=to_rate
            )
            return (resampled * 32768.0).astype(np.int16)
        except ImportError:
            logger.warning("librosa not available, using simple resampling")
            # Simple linear interpolation as fallback
            ratio = to_rate / from_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)
    
    async def stream_audio_chunks(
        self,
        audio_stream: AsyncIterator[rtc.AudioFrame]
    ) -> AsyncIterator[np.ndarray]:
        """
        Stream audio chunks from LiveKit audio frames.
        
        Args:
            audio_stream: Async iterator of LiveKit AudioFrames
            
        Yields:
            Processed audio chunks as numpy arrays
        """
        async for frame in audio_stream:
            chunk = await self.process_audio_frame(frame)
            if chunk is not None:
                yield chunk
    
    async def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        async with self._buffer_lock:
            self._audio_buffer.clear()
            logger.debug("Audio buffer cleared")
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer size in samples."""
        return len(self._audio_buffer)
