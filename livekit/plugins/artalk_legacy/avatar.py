"""
ARTalk Avatar Session

Main class for managing ARTalk avatar sessions in LiveKit.
Connects the TTS Wrapper audio hook into the SDK Wrapper 
to render Gaussian Splatting video frames into a LiveKit Room.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterable
import numpy as np
import queue
import threading

from livekit import rtc, agents
from livekit.agents import utils, tts

from .artalk_sdk import ARTalkSDKWrapper, ModelLoadStrategy
from .video_source import ARTalkVideoSource

logger = logging.getLogger(__name__)


class TTSWrapper(tts.TTS):
    """
    Wrapper for TTS to intercept audio for ARTalk avatar.
    """
    def __init__(self, wrapped_tts: tts.TTS, session: "ARTalkAvatarSession"):
        super().__init__(
            capabilities=wrapped_tts.capabilities,
            sample_rate=wrapped_tts.sample_rate,
            num_channels=wrapped_tts.num_channels,
        )
        self._wrapped_tts = wrapped_tts
        self._session = session

    def synthesize(self, text: str) -> "ChunkedStream":
        return self._wrapped_tts.synthesize(text)

    def stream(self, **kwargs) -> "SynthesizeStream":
        conn_options = kwargs.get("conn_options")
        wrapped_stream = self._wrapped_tts.stream(**kwargs)
        return TTSWrapperStream(
            wrapped_stream, 
            self._session,
            tts=self, 
            conn_options=conn_options
        )


class TTSWrapperStream(tts.SynthesizeStream):
    def __init__(
        self, 
        wrapped_stream: tts.SynthesizeStream, 
        session: "ARTalkAvatarSession",
        *,
        tts: tts.TTS,
        conn_options: Any
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._wrapped_stream = wrapped_stream
        self._session = session
        self._emitter: Optional[tts.AudioEmitter] = None

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
        self._emitter = output_emitter
        self._session.active_emitter = output_emitter
        
        try:
            request_id = utils.shortuuid()
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type="audio/pcm",
                stream=True,
            )
            segment_id = utils.shortuuid()
            output_emitter.start_segment(segment_id=segment_id)
            
            # Use session-level flag for worker sync
            logger.debug("DEBUG AVATAR: _is_speaking set to TRUE (TTS start)")
            self._session._is_speaking = True
            self._session._last_audio_push_time = asyncio.get_event_loop().time()
            
            # Immediately clear residual idle frames so first speech word plays instantly
            while not self._session._async_frame_queue.empty():
                try:
                    self._session._async_frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            async for synthesized_audio in self._wrapped_stream:
                frame = synthesized_audio.frame
                audio_bytes = frame.data.tobytes()
                output_emitter.push(audio_bytes)
                
                if self._session.sdk.is_loaded:
                    try:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_data.astype(np.float32) / 32768.0
                        
                        tts_sample_rate = self._tts.sample_rate
                        if tts_sample_rate != 16000:
                            target_len = int(len(audio_float) * 16000 / tts_sample_rate)
                            if target_len > 0:
                                indices = np.linspace(0, len(audio_float) - 1, target_len)
                                audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)
                        
                        audio_bytes_16k = (audio_float * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
                        
                        # Push to session-level inference queue
                        self._session._inference_queue.put_nowait((audio_float, audio_bytes_16k))
                        
                    except Exception as e:
                        logger.error(f"Error intercepting audio to SDK: {e}")
            
            output_emitter.flush()
        except Exception as e:
            logger.error(f"Error inside TTSWrapperStream._run: {e}")
            raise
        finally:
            logger.debug("DEBUG AVATAR: _is_speaking set to FALSE (Stream end)")
            self._session._is_speaking = False
            self._session.active_emitter = None


class ARTalkAvatarSession:
    """
    Coordinates ARTalk inference wrapper, audio interception, and LiveKit Video injection.
    """
    def __init__(
        self,
        *,
        artalk_path: str,
        avatar_participant_identity: str = "artalk-avatar",
        shape_id: str = "mesh",
        style_id: str = "default",
        video_width: int = 512,
        video_height: int = 512,
        video_fps: int = 25,
        device: str = "cuda",
        model_strategy: ModelLoadStrategy = ModelLoadStrategy.FROM_SCRATCH,
        load_gaga: bool = True,
    ):
        self.artalk_path = artalk_path
        self.avatar_identity = avatar_participant_identity
        self.device = device
        self.sdk = ARTalkSDKWrapper(
            artalk_path=artalk_path,
            shape_id=shape_id,
            style_id=style_id,
            model_strategy=model_strategy,
            load_gaga=load_gaga,
        )
        self.video_source = ARTalkVideoSource(
            width=video_width,
            height=video_height,
            fps=video_fps,
        )

        self._room: Optional[rtc.Room] = None
        self._tasks: list[asyncio.Task] = []
        self._is_running = False
        self._is_speaking = False
        self._inference_queue = asyncio.Queue()
        self._async_frame_queue = asyncio.Queue()
        self._inference_lock = asyncio.Lock()
        self._clear_buffer_event = asyncio.Event()
        self._is_buffering = False
        self._last_audio_push_time = 0.0
        self._idle_batch_count = 0
        
        logger.info(f"Initialized ARTalkAvatarSession: {self.avatar_identity}, shape={shape_id}")

    def wrap_tts(self, tts_instance: tts.TTS) -> tts.TTS:
        return TTSWrapper(tts_instance, self)

    async def _inference_worker(self):
        """Unified persistent worker: Speech prioritizing, Idle falling back."""
        logger.debug("DEBUG AVATAR: Inference worker STARTED (v3.3.3)")
        
        # 1. STARTUP STABILIZATION: Allow WebRTC handshake to settle before rushing idle frames.
        await asyncio.sleep(1.0)
        
        # 2. INITIAL PURGE: Ensure no warmup leakage frames exist.
        while not self._async_frame_queue.empty():
            try: self._async_frame_queue.get_nowait()
            except asyncio.QueueEmpty: break

        while self._is_running:
            try:
                now = asyncio.get_event_loop().time()
                has_work = not self._inference_queue.empty()
                
                # SILENCE FALLBACK: Even if technically "speaking", if no audio arrived 
                # for > 300ms, allow idle frames to keep the avatar alive.
                is_stalled = self._is_speaking and (now - self._last_audio_push_time > 0.3)
                
                # 1. Check for SPEECH work first
                if has_work:
                    try:
                        # PURGE IDLE FRAMES: The moment speech starts, throw away 
                        # any residual "stall" frames to ensure zero lip-sync lag.
                        if not self._is_speaking or (now - self._last_audio_push_time > 1.0):
                             logger.debug("DEBUG AVATAR: Speech sequence DETECTED. Purging residual frames.")
                             while not self._async_frame_queue.empty():
                                 try: self._async_frame_queue.get_nowait()
                                 except asyncio.QueueEmpty: break

                        item = await asyncio.wait_for(self._inference_queue.get(), timeout=0.01)
                        audio_float, audio_bytes = item
                        async with self._inference_lock:
                            await asyncio.to_thread(self.sdk.process_audio_chunk, audio_float, 16000, audio_bytes, False)
                        
                        await self._extract_frames()
                        # Record successful speech push
                        self._last_audio_push_time = asyncio.get_event_loop().time()
                        
                        # Explicit cleanup for memory safety
                        del audio_float, audio_bytes, item
                        continue
                    except asyncio.TimeoutError:
                        pass

                # 2. IDLE logic: If NOT speaking OR if STALLED (TTS retry/LLM thinking)
                if (not self._is_speaking or is_stalled) and self.sdk.is_loaded:
                    if self._async_frame_queue.qsize() < 40:
                        if is_stalled:
                            logger.debug(f"DEBUG AVATAR: STALL detected ({now - self._last_audio_push_time:.1f}s). Forcing IDLE frames.")
                        
                        async with self._inference_lock:
                            # Use -1 to bypass the session ID check bug
                            await asyncio.to_thread(self.sdk.generate_idle_frames, num_frames=25, reset_id=-1)
                        
                        await self._extract_frames()
                        
                        # Memory Optimization: Every 50 idle batches, purge GPU cache to stop the 3GB climb.
                        self._idle_batch_count += 1
                        if self._idle_batch_count % 50 == 0:
                            import gc
                            import torch
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.debug("DEBUG AVATAR: Memory garbage collection triggered.")
                    else:
                        await asyncio.sleep(0.1) # Buffer healthy
                else:
                    await asyncio.sleep(0.01) # Yield during speech gaps

            except Exception as e:
                logger.error(f"Inference worker error: {e}")
                await asyncio.sleep(0.1)
        logger.debug("DEBUG AVATAR: Inference worker STOPPED")

    async def _extract_frames(self):
        """Extract all pending frames from SDK sync queue to session async queue."""
        sdk_queue = self.sdk.get_frame_queue()
        count = 0
        while not sdk_queue.empty():
            try:
                frame_item = sdk_queue.get_nowait()
                self._async_frame_queue.put_nowait(frame_item)
                count += 1
            except queue.Empty:
                break
        if count > 0:
            logger.debug(f"DEBUG TRANSFER: Extracted {count} frames. Async size={self._async_frame_queue.qsize()}")

    async def start(self, room: rtc.Room) -> None:
        if self._is_running:
            return
        self._room = room
        try:
            logger.info("Loading ARTalk SDK Engine...")
            self.sdk.load(device=self.device)
            video_track = self.video_source.create_track(f"{self.avatar_identity}-video")
            await room.local_participant.publish_track(video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
            
            self._is_running = True
            
            # 1. PRIME WEBRTC: Publish a single black frame to avoid the initial green screen.
            black_frame = np.zeros((self.video_source.height, self.video_source.width, 3), dtype=np.uint8)
            await self.video_source.publish_frame(black_frame)

            # Start persistent session loops
            self._tasks.append(asyncio.create_task(self._process_frame_loop()))
            self._tasks.append(asyncio.create_task(self._inference_worker()))
            
            self.video_source.start_publishing()
            logger.info("Session started with persistent worker.")
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            await self.stop()
            raise

    async def _process_frame_loop(self) -> None:
        logger.info("Entered video publishing loop.")
        frame_duration = 1.0 / self.video_source.fps
        
        while self._is_running:
            try:
                # Direct block on async queue
                if self._async_frame_queue.empty():
                    # print("DEBUG AVATAR: Queue empty, waiting for frame...", flush=True)
                    pass
                
                queue_item = await self._async_frame_queue.get()
                
                # Check for reset/clear events
                if self._clear_buffer_event.is_set():
                    self._clear_buffer_event.clear()
                    while not self._async_frame_queue.empty():
                        self._async_frame_queue.get_nowait()
                    continue

                if isinstance(queue_item, tuple) and len(queue_item) >= 1:
                    frame = queue_item[0]
                    if not isinstance(frame, str):
                        await self.video_source.publish_frame(frame)
                
                # Enforce FPS pacing
                await asyncio.sleep(frame_duration * 0.9) # Lean sleep for 25fps
                
            except Exception as e:
                logger.warning(f"Error in frame publisher: {e}")
                await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if not self._is_running:
            return
        self._is_running = False
        self.video_source.stop_publishing()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.sdk.close()

    @property
    def is_running(self) -> bool:
        return self._is_running
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
