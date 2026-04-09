import asyncio
import logging
import os
import sys
import numpy as np
from typing import Optional
from dotenv import load_dotenv

from livekit import rtc

# Ensure we can import the ARTalk SDK
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from .artalk_sdk import ARTalkSDKWrapper
    from .video_source import ARTalkVideoSource
except ImportError:
    try:
        from artalk_sdk import ARTalkSDKWrapper
        from video_source import ARTalkVideoSource
    except ImportError as e:
        ARTalkSDKWrapper = None
        ARTalkVideoSource = None
        logging.error(f"Failed to import local ARTalk components: {e}")

logger = logging.getLogger("artalk_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

async def start_livekit_worker(
    conversation_id: str,
    replica_id: str,
    ws_url: str,
    token: str,
    artalk_path: str,
    background_scene: Optional[str] = None,
    bg_threshold: Optional[int] = None,
):
    """
    Background worker that connects to a LiveKit room as an avatar agent (like Tavus),
    receives audio via DataChannels, and streams video back.
    """
    if ARTalkSDKWrapper is None or ARTalkVideoSource is None:
        logger.error("Cannot start worker, ARTalkSDKWrapper not available.")
        return

    logger.info(f"[{conversation_id}] Starting WebRTC Worker for replica {replica_id}")
    
    room = rtc.Room()
    artalk_lock = asyncio.Lock()

    # Initialize ARTalk SDK Wrapper
    logger.info(f"[{conversation_id}] Loading ARTalk SDK...")
    artalk = ARTalkSDKWrapper(
        artalk_path=artalk_path,
        shape_id=replica_id,
        max_queue_size=100
    )
    # We load it synchronously inside this background worker (or in an executor)
    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[{conversation_id}] Loading ARTalk with device: {device_str}")
    artalk.load(device=device_str)

    # Initialize Video Source
    # Reload .env here to guarantee env vars are up-to-date even if the server
    # process had stale values inherited from the shell before modification.
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _env_path = os.path.join(_project_root, ".env")
    load_dotenv(_env_path, override=True)
    logger.info(f"[{conversation_id}] Env loaded from: {_env_path} (exists={os.path.isfile(_env_path)})")

    # Priority: 1. API request param, 2. .env / shell env var, 3. hardcoded default
    scene = background_scene or os.getenv("AVATAR_SCENE") or "office"
    threshold = bg_threshold if bg_threshold is not None else int(os.getenv("AVATAR_BG_THRESHOLD", "15"))
    logger.info(f"[{conversation_id}] Background scene='{scene}' | bg_threshold={threshold}")

    try:
        video_source = ARTalkVideoSource(
            width=512, height=512, fps=25,
            background_scene=scene, bg_threshold=threshold,
        )
    except ValueError as bg_err:
        logger.error(
            f"[{conversation_id}] Failed to load background scene '{scene}': {bg_err}. "
            "Starting conversation WITHOUT a background."
        )
        # Fall back to no background so the conversation can still run
        video_source = ARTalkVideoSource(width=512, height=512, fps=25, background_scene=None, bg_threshold=threshold)

    video_track = rtc.LocalVideoTrack.create_video_track("avatar-video", video_source.create_source())


    # Initialize Audio Source
    audio_source = rtc.AudioSource(16000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("avatar-audio", audio_source)

    # WARM-UP PIPELINE: Generate 1 frame and push it to the internal video source.
    # This forces MediaPipe, OpenCV, and any PyTorch JIT compilation to finish
    # BEFORE we connect to the WebRTC room. This prevents a massive blocking event
    # on the Python asyncio loop right after publishing tracks, which avoids WebRTC timeouts.
    logger.info(f"[{conversation_id}] Warming up VideoSource compositor pipeline...")
    async with artalk_lock:
        await asyncio.to_thread(artalk.generate_idle_frames, 1, -1)
    
    if not artalk._frame_queue.empty():
        try:
            warmup_frame_data = artalk._frame_queue.get_nowait()
            if warmup_frame_data is not None and video_source.source:
                _publish_video_frame(video_source, warmup_frame_data[0])
        except Exception as e:
            logger.warning(f"[{conversation_id}] Failed during pipeline warmup: {e}")

    def on_audio_stream_received(stream: rtc.ByteStreamReader, identity: str):
        async def _read_stream():
            stream_sample_rate = 16000
            if stream.info and stream.info.attributes and "sample_rate" in stream.info.attributes:
                stream_sample_rate = int(stream.info.attributes["sample_rate"])
                
            async for chunk in stream:
                if len(chunk) > 0:
                    try:
                        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Resample to 16000Hz (ARTalk and worker output requirement)
                        if stream_sample_rate != 16000:
                            target_len = int(len(audio_np) * 16000 / stream_sample_rate)
                            if target_len > 0:
                                indices = np.linspace(0, len(audio_np) - 1, target_len)
                                audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.float32)
                                chunk = (audio_np * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
                                
                        async with artalk_lock:
                            await asyncio.to_thread(artalk.process_audio_chunk, audio_np, 16000, chunk, False)
                    except Exception as e:
                        logger.error(f"[{conversation_id}] Error processing audio chunk: {e}")
            
            # Flush ending audio context
            try:
                async with artalk_lock:
                    await asyncio.to_thread(artalk.flush_audio, 16000)
            except Exception as e:
                logger.error(f"[{conversation_id}] Error flushing audio: {e}")
                
        asyncio.create_task(_read_stream())

    room.register_byte_stream_handler("lk.audio_stream", on_audio_stream_received)
            
    # Connect to room
    logger.info(f"[{conversation_id}] Connecting to LiveKit Room...")
    try:
        await room.connect(ws_url, token)
        logger.info(f"[{conversation_id}] Connected to room successfully.")
        
        def on_clear_buffer(data: rtc.RpcInvocationData) -> str:
            # Clear any pending generated idle frames or speech
            while not artalk._frame_queue.empty():
                try:
                    artalk._frame_queue.get_nowait()
                except Exception:
                    pass
            return "ok"
            
        room.local_participant.register_rpc_method("lk.clear_buffer", on_clear_buffer)

        
        # Publish video track
        options_video = rtc.TrackPublishOptions()
        options_video.source = rtc.TrackSource.SOURCE_CAMERA
        await room.local_participant.publish_track(video_track, options_video)
        logger.info(f"[{conversation_id}] Avatar video track published.")

        # Publish audio track
        options_audio = rtc.TrackPublishOptions()
        options_audio.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(audio_track, options_audio)
        logger.info(f"[{conversation_id}] Avatar audio track published.")
        
        # 1. PRIME WEBRTC: Generate 1 static idle frame and hold it to let the video codec stabilize (prevents macroblocking deformation)
        async with artalk_lock:
            await asyncio.to_thread(artalk.generate_idle_frames, 1, -1)
            
        if not artalk._frame_queue.empty():
            try:
                first_frame_data = artalk._frame_queue.get_nowait()
                if first_frame_data is not None:
                    first_frame_rgb = first_frame_data[0]
                    # Publish this identical frame multiple times to force a clean high-quality keyframe baseline
                    for _ in range(15):
                        if video_source.source:
                            _publish_video_frame(video_source, first_frame_rgb)
                        await asyncio.sleep(0.04)
            except Exception as e:
                logger.warning(f"Failed to prime video track: {e}")
        
        # Frame publishing loop
        video_source.start_publishing()
        frame_duration = 1.0 / 25.0
        
        while room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            loop_start = asyncio.get_event_loop().time()
            
            # If idle and no frames pending, generate one idle frame synchronously
            if artalk._frame_queue.empty():
                async with artalk_lock:
                    await asyncio.to_thread(artalk.generate_idle_frames, 1, -1)
                    
            # Pull exactly one frame to maintain 25 FPS pacing
            frame_idx, frame_data = await asyncio.to_thread(_get_frame_from_queue, artalk)
            if frame_data is not None:
                # frame_data is a tuple of (frame_rgb, frame_audio, is_silence)
                frame_rgb = frame_data[0]
                frame_audio = frame_data[1]
                
                if video_source.source:
                    _publish_video_frame(video_source, frame_rgb)

                if isinstance(frame_audio, bytes) and len(frame_audio) > 0:
                    samples_per_channel = len(frame_audio) // 2
                    audio_frame = rtc.AudioFrame(frame_audio, 16000, 1, samples_per_channel)
                    await audio_source.capture_frame(audio_frame)
                    
            loop_end = asyncio.get_event_loop().time()
            sleep_time = max(0.001, frame_duration - (loop_end - loop_start))
            await asyncio.sleep(sleep_time)
                
    except Exception as e:
        logger.error(f"[{conversation_id}] Worker error: {e}")
    finally:
        logger.info(f"[{conversation_id}] Disconnecting from room...")
        await room.disconnect()

def _get_frame_from_queue(artalk):
    try:
        return 0, artalk._frame_queue.get_nowait()
    except Exception:
        return 0, None
        
def _publish_video_frame(video_source, frame_rgb):
    """
    Publish a frame through the video_source, applying background composition if configured.
    Delegates all processing to ARTalkVideoSource.compose_and_publish().
    """
    try:
        video_source.compose_and_publish(frame_rgb)
    except Exception as e:
        logger.error(f"Error publishing frame: {e}")
