"""
Ditto SDK Wrapper

This module provides a wrapper around the external Ditto TalkingHead SDK,
treating it as an external resource that can be loaded and managed independently.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class DittoSDKWrapper:
    """
    Wrapper for the Ditto TalkingHead SDK as an external resource.
    
    This class handles loading the Ditto SDK from an external path and provides
    a clean interface for video generation.
    """
    
    def __init__(
        self,
        ditto_path: str,
        data_root: str = "./checkpoints/ditto_pytorch",
        cfg_pkl: str = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
    ):
        """
        Initialize the Ditto SDK wrapper.
        
        Args:
            ditto_path: Path to the Ditto TalkingHead installation
            data_root: Path to Ditto model checkpoints
            cfg_pkl: Path to Ditto configuration pickle file
        """
        self.ditto_path = Path(ditto_path)
        self.data_root = data_root
        self.cfg_pkl = cfg_pkl
        self._sdk = None
        self._is_loaded = False
        
        logger.info(f"Initializing Ditto SDK wrapper with path: {ditto_path}")
        
    def load(self) -> None:
        """Load the Ditto SDK from the external path."""
        if self._is_loaded:
            logger.warning("Ditto SDK already loaded")
            return
            
        try:
            # Check for GPU support
            if not torch.cuda.is_available():
                logger.error("No CUDA-capable GPU found")
                raise RuntimeError(
                    "Ditto requires a CUDA-capable GPU to run. "
                    "Please ensure you have a compatible GPU and PyTorch with CUDA support installed."
                )

            # Add Ditto path to sys.path temporarily
            if str(self.ditto_path) not in sys.path:
                sys.path.insert(0, str(self.ditto_path))
                logger.debug(f"Added {self.ditto_path} to sys.path")
            
            # Import the StreamSDK from external Ditto installation
            from stream_pipeline_online import StreamSDK
            
            # Initialize the SDK
            self._sdk = StreamSDK(self.cfg_pkl, self.data_root, start_writer=False)
            self._is_loaded = True
            
            logger.info("Ditto SDK loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Ditto SDK: {e}")
            raise RuntimeError(
                f"Could not load Ditto SDK from {self.ditto_path}. "
                f"Make sure the Ditto installation is at the specified path."
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Ditto SDK: {e}")
            raise
    
    def setup(
        self,
        source_image: str,
        output_path: str,
        **kwargs
    ) -> None:
        """
        Setup Ditto for video generation.
        
        Args:
            source_image: Path to the source avatar image
            output_path: Path for temporary output video
            **kwargs: Additional setup parameters
        """
        if not self._is_loaded:
            raise RuntimeError("Ditto SDK not loaded. Call load() first.")
        
        logger.info(f"Setting up Ditto with source image: {source_image}")
        self._sdk.setup(source_image, output_path, **kwargs)
    
    def setup_frames(
        self,
        num_frames: int,
        fade_in: int = -1,
        fade_out: int = -1,
        ctrl_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Setup number of frames and control information.
        
        Args:
            num_frames: Number of frames to generate
            fade_in: Fade in duration (-1 for no fade)
            fade_out: Fade out duration (-1 for no fade)
            ctrl_info: Control information (emotions, eye movements, etc.)
        """
        if not self._is_loaded:
            raise RuntimeError("Ditto SDK not loaded. Call load() first.")
        
        self._sdk.setup_Nd(
            N_d=num_frames,
            fade_in=fade_in,
            fade_out=fade_out,
            ctrl_info=ctrl_info
        )
    
    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray
    ) -> None:
        """
        Process an audio chunk and queue it for video generation.
        
        Args:
            audio_chunk: Audio data as numpy array (16kHz, mono)
        """
        if not self._is_loaded:
            raise RuntimeError("Ditto SDK not loaded. Call load() first.")
        
        # Convert audio to features
        aud_feat = self._sdk.wav2feat.wav2feat(audio_chunk)
        
        # Queue for processing
        self._sdk.audio2motion_queue.put(aud_feat)

    def update_emotion(self, emo: Union[int, list], intensity: float = 1.0) -> None:
        """
        Update emotional state in real-time.
        
        Args:
            emo: Emotion index (0-7) or list of indices/weights
            intensity: Strength of the emotion (0.0 to 1.0)
        """
        if not self._is_loaded:
            return

        try:
            # We need to import softmax here or at top level if available
            # But since it's a specific scipy function used inside Ditto, 
            # we rely on how Ditto implements it or implement it ourselves if needed.
            # Ditto's condition_handler uses scipy.special.softmax.
            from scipy.special import softmax
            
            # Scale the base weight of 8.0 by the intensity
            weight = 8.0 * max(0.0, min(1.0, float(intensity)))
            
            if isinstance(emo, int):
                emo_vec = np.zeros(8, dtype=np.float32)
                emo_vec[emo] = weight
                emo_arr = softmax(emo_vec).reshape(1, 8)
            elif isinstance(emo, list):
                emo_vec = np.zeros(8, dtype=np.float32)
                for idx in emo:
                    emo_vec[idx] = weight / len(emo)
                emo_arr = softmax(emo_vec).reshape(1, 8)
            else:
                logger.warning(f"Unsupported emotion type for update: {type(emo)}")
                return
            
            # Access the condition handler from the SDK instance
            if hasattr(self._sdk, 'condition_handler'):
                # Update the stored emotion list
                self._sdk.condition_handler.emo_lst = emo_arr
                
                # Update the sequence buffer to apply immediately to next frames
                # The condition handler uses emo_seq for the sequence length
                seq_frames = self._sdk.condition_handler.seq_frames
                self._sdk.condition_handler.emo_seq = np.concatenate(
                    [emo_arr] * seq_frames, 0
                )
                logger.debug(f"Emotion updated in condition_handler: {emo}")
            else:
                logger.warning("Ditto SDK instance missing condition_handler")
                
        except Exception as e:
            logger.error(f"Failed to update emotion: {e}")

    
    def get_frame_queue(self):
        """Get access to the frame output queue."""
        if not self._is_loaded:
            raise RuntimeError("Ditto SDK not loaded. Call load() first.")
        
        # This would need to be implemented in the Ditto SDK
        # For now, return a placeholder
        if hasattr(self._sdk, 'writer_queue'):
            return self._sdk.writer_queue
        return None
    
    def close(self) -> None:
        """Close the Ditto SDK and clean up resources."""
        if self._sdk and self._is_loaded:
            logger.info("Closing Ditto SDK")
            self._sdk.close()
            self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if the SDK is loaded."""
        return self._is_loaded
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
