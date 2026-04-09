"""
Ditto Avatar Session

Main class for managing Ditto avatar sessions in LiveKit.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterable
import numpy as np
from livekit import rtc, agents
from livekit.agents import utils, tts

from .ditto_sdk import DittoSDKWrapper
from .video_source import DittoVideoSource
from .emotion_controller import EmotionController, DittoEmotion

logger = logging.getLogger(__name__)


class TTSWrapper(tts.TTS):
    """
    Wrapper for TTS to intercept audio for Ditto avatar.
    """
    def __init__(self, wrapped_tts: tts.TTS, ditto_sdk: DittoSDKWrapper, emotion_controller: Optional[EmotionController] = None):
        super().__init__(
            capabilities=wrapped_tts.capabilities,
            sample_rate=wrapped_tts.sample_rate,
            num_channels=wrapped_tts.num_channels,
        )
        self._wrapped_tts = wrapped_tts
        self._ditto_sdk = ditto_sdk
        self._emotion_controller = emotion_controller

    def synthesize(self, text: str) -> "ChunkedStream":
        return self._wrapped_tts.synthesize(text)

    def stream(self, **kwargs) -> "SynthesizeStream":
        conn_options = kwargs.get("conn_options")
        logger.info(f"TTSWrapper.stream called with conn_options: {conn_options}")
        wrapped_stream = self._wrapped_tts.stream(**kwargs)
        return TTSWrapperStream(
            wrapped_stream, 
            self._ditto_sdk,
            self._emotion_controller,
            tts=self, 
            conn_options=conn_options
        )


class TTSWrapperStream(tts.SynthesizeStream):
    def __init__(
        self, 
        wrapped_stream: tts.SynthesizeStream, 
        ditto_sdk: DittoSDKWrapper,
        emotion_controller: Optional[EmotionController],
        *,
        tts: tts.TTS,
        conn_options: Any
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        logger.info("TTSWrapperStream initialized")
        self._wrapped_stream = wrapped_stream
        self._ditto_sdk = ditto_sdk
        self._emotion_controller = emotion_controller

    @property
    def validation_error(self) -> Optional[str]:
        return self._wrapped_stream.validation_error

    def push_text(self, token: str | None) -> None:
        self._wrapped_stream.push_text(token)

    def flush(self) -> None:
        if asyncio.iscoroutinefunction(self._wrapped_stream.flush):
            asyncio.create_task(self._wrapped_stream.flush())
        else:
            self._wrapped_stream.flush()

    async def aclose(self) -> None:
        await self._wrapped_stream.aclose()
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Core TTS processing loop. Delegates to the wrapped TTS stream,
        intercepts audio frames for Ditto, and re-emits them downstream.
        
        The LiveKit framework calls this method to drive TTS synthesis.
        Without a working implementation here, no audio is ever generated.
        """
        request_id = utils.shortuuid()
        
        # Initialize the output emitter with the audio format
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )
        
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        
        logger.info(f"TTSWrapperStream._run started (request_id={request_id})")
        
        try:
            # Iterate over the wrapped stream's output
            # The wrapped stream (e.g. Cartesia, ElevenLabs) has its own _run() 
            # that is triggered when we iterate over it via __aiter__
            async for synthesized_audio in self._wrapped_stream:
                frame = synthesized_audio.frame

                # Infer emotion from text if available and controller is present
                if self._emotion_controller and hasattr(synthesized_audio, 'segment') and synthesized_audio.segment:
                     # Check if segment has text, sometimes it's just audio
                     text = getattr(synthesized_audio.segment, 'text', "")
                     if text:
                        self._emotion_controller.set_from_text(text)
                
                # 1. Re-emit audio downstream so LiveKit plays it to the user
                audio_bytes = frame.data.tobytes()
                output_emitter.push(audio_bytes)
                
                # 2. Intercept audio and feed to Ditto for avatar animation
                if self._ditto_sdk.is_loaded:
                    try:
                        # Convert PCM16 audio to float32 for Ditto
                        # TTS typically outputs PCM16 (int16)
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_data.astype(np.float32) / 32768.0
                        
                        # Resample to 16kHz if needed (Ditto expects 16kHz mono)
                        # Most TTS outputs at higher sample rates (22050, 24000, etc)
                        tts_sample_rate = self._tts.sample_rate
                        if tts_sample_rate != 16000:
                            # Simple resampling using numpy interpolation
                            target_len = int(len(audio_float) * 16000 / tts_sample_rate)
                            if target_len > 0:
                                indices = np.linspace(0, len(audio_float) - 1, target_len)
                                audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)
                        
                        self._ditto_sdk.process_audio_chunk(audio_float)
                        logger.debug(f"Fed {len(audio_float)} samples to Ditto")
                        
                    except Exception as e:
                        logger.error(f"Error processing audio for Ditto: {e}")
            
            # Signal end of audio
            output_emitter.flush()
            logger.info("TTSWrapperStream._run completed")
            
        except Exception as e:
            logger.error(f"Error in TTSWrapperStream._run: {e}")
            raise


class DittoAvatarSession:
    """
    Manages a Ditto avatar session within a LiveKit room.
    
    This class coordinates the Ditto SDK, audio processing, and video publishing
    to create a real-time talking head avatar.
    """
    
    def __init__(
        self,
        *,
        ditto_path: str,
        source_image: str,
        avatar_participant_identity: str = "ditto-avatar",
        data_root: str = "./checkpoints/ditto_pytorch",
        cfg_pkl: str = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        video_width: int = 512,
        video_height: int = 512,
        video_fps: int = 25,
        audio_chunk_ms: int = 200,
        emotion_controller: Optional[EmotionController] = None,
    ):
        """
        Initialize the Ditto avatar session.
        
        Args:
            ditto_path: Path to the Ditto TalkingHead installation
            source_image: Path to the avatar source image
            avatar_participant_identity: Identity for the avatar participant
            data_root: Path to Ditto model checkpoints
            cfg_pkl: Path to Ditto configuration file
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frames per second
            audio_chunk_ms: Audio chunk size in milliseconds
        """
        self.ditto_path = ditto_path
        self.source_image = source_image
        self.avatar_identity = avatar_participant_identity
        self.emotion_controller = emotion_controller
        
        # Initialize components
        self.ditto_sdk = DittoSDKWrapper(
            ditto_path=ditto_path,
            data_root=data_root,
            cfg_pkl=cfg_pkl,
        )
        
        self.video_source = DittoVideoSource(
            width=video_width,
            height=video_height,
            fps=video_fps,
        )

        # State
        self._room: Optional[rtc.Room] = None
        self._agent_session = None
        self._tasks: list[asyncio.Task] = []
        self._is_running = False
        
        logger.info(
            f"Initialized DittoAvatarSession: {self.avatar_identity}, "
            f"image={self.source_image}"
        )

    async def _setup_emotion_listener(self, room: rtc.Room) -> None:
        """Listen for emotion control messages from DataChannel."""
        logger.info("Setting up emotion listener")
        
        @room.on("data_received")
        def on_data(data: rtc.DataPacket):
            try:
                import json
                # LiveKit DataPacket data is bytes
                if data.topic: # Optional topic filtering if needed
                    pass
                
                payload_str = data.data.decode()
                payload = json.loads(payload_str)
                
                if payload.get("type") == "emotion_control":
                    emotion = payload.get("emotion")
                    intensity = payload.get("intensity", 1.0)
                    
                    if emotion is not None and self.emotion_controller:
                        logger.info(f"Received emotion command: {emotion}")
                        self.emotion_controller.set_emotion(
                            emotion, 
                            intensity=float(intensity)
                        )
            except Exception as e:
                logger.warning(f"Error processing data message: {e}")
        


    def wrap_tts(self, tts_instance: tts.TTS) -> tts.TTS:
        """
        Wrap a TTS instance to capture generated audio for the avatar.
        """
        return TTSWrapper(tts_instance, self.ditto_sdk, self.emotion_controller)
    
    async def start(
        self,
        agent_session,
        room: rtc.Room,
    ) -> None:
        """
        Start the avatar session.
        
        Args:
            agent_session: The LiveKit AgentSession
            room: The LiveKit Room to publish to
        """
        if self._is_running:
            logger.warning("Avatar session already running")
            return
        
        logger.info("Starting Ditto avatar session")
        
        self._room = room
        self._agent_session = agent_session
        
        try:
            # Load Ditto SDK
            logger.info("Loading Ditto SDK...")
            self.ditto_sdk.load()

            # Attach EmotionController if present
            if self.emotion_controller:
                logger.info("Attaching EmotionController to SDK")
                self.emotion_controller.attach_sdk(self.ditto_sdk)
                await self._setup_emotion_listener(room)
            
            # Setup Ditto with source image
            logger.info("Setting up Ditto with source image...")
            output_path = f"./tmp/{self.avatar_identity}_output.mp4"
            Path("./tmp").mkdir(exist_ok=True)
            self.ditto_sdk.setup(self.source_image, output_path)
            
            # Create and publish video track
            logger.info("Creating video track...")
            video_track = self.video_source.create_track(
                f"{self.avatar_identity}-video"
            )
            
            await room.local_participant.publish_track(
                video_track,
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_CAMERA
                )
            )
            
            logger.info("Video track published successfully")
            
            # Start processing loops
            self._is_running = True
            self.video_source.start_publishing()
            
            # Start video frame processing task
            self._tasks.append(
                asyncio.create_task(self._process_frame_loop())
            )
            
            logger.info("Ditto avatar session started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start avatar session: {e}")
            await self.stop()
            raise
    
    async def _process_frame_loop(self) -> None:
        """
        Loop for retrieving generated video frames from Ditto and publishing them.
        """
        logger.info("Starting frame processing loop")
        
        try:
            frame_queue = self.ditto_sdk.get_frame_queue()
            if frame_queue is None:
                logger.error("Could not get frame queue from Ditto SDK")
                return

            logger.info("Frame queue retrieved, entering loop")
            while self._is_running:
                # Use asyncio.to_thread for blocking queue.get
                try:
                    # Get frame from queue (blocking)
                    # We run this in a thread to verify async loop isn't blocked
                    # logger.debug("Waiting for frame...")
                    frame = await asyncio.to_thread(frame_queue.get, timeout=0.1)
                    
                    if frame is None:
                        continue
                        
                    # logger.debug("Got frame, publishing...")
                    # Publish frame
                    await self.video_source.publish_frame(frame)
                    
                except Exception:
                    # Queue empty or timeout, just continue
                    # Add a small sleep to prevent tight loop if queue.get is not blocking well
                    await asyncio.sleep(0.001) 
                    continue
                
        except Exception as e:
            logger.error(f"Error in frame processing loop: {e}")
        finally:
            logger.info("Frame processing loop ended")
    
    async def stop(self) -> None:
        """Stop the avatar session and clean up resources."""
        if not self._is_running:
            return
        
        logger.info("Stopping Ditto avatar session")
        
        self._is_running = False
        self.video_source.stop_publishing()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        
        # Close Ditto SDK
        self.ditto_sdk.close()
        
        logger.info("Ditto avatar session stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if the session is running."""
        return self._is_running
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
