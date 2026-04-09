# ARTalk LiveKit Plugin Implementation Plan

This document outlines the current implementation of the ARTalk LiveKit plugin, which integrates the external ARTalk SDK for real-time 3D avatar generation (Gaussian Splatting) with LiveKit's real-time communication infrastructure.

## Core Components

### 1. `avatar.py`: Session Coordination & Audio Interception
- **`ARTalkAvatarSession`**: The main class responsible for coordinating the entire pipeline. It manages the `ARTalkSDKWrapper` and `ARTalkVideoSource`.
- **`TTSWrapper` & `TTSWrapperStream`**: Intercepts generated TTS audio. It pushes the audio downstream to LiveKit so the user hears it, and simultaneously feeds the PCM audio (resampled to 16kHz) to the ARTalk SDK for avatar animation.
- **Asynchronous Frame Processing**: Runs a dedicated asynchronous task (`_process_frame_loop`) that continuously polls the thread-safe frame queue from the SDK and publishes frames to LiveKit.

### 2. `video_source.py`: WebRTC Video Publishing
- **`ARTalkVideoSource`**: Manages the custom `rtc.VideoSource` proxy.
- **Frame Processing (`publish_frame`)**: Takes numpy 3D arrays (frames) from the SDK, resizes them if necessary using OpenCV (`cv2`), converts RGB to RGBA since LiveKit expects RGBA format, and captures the frame into the WebRTC stream via `rtc.VideoFrame`.

### 3. `artalk_sdk.py`: External ARTalk Engine Wrapper
- **`ARTalkSDKWrapper`**: Acts as an abstraction layer over the cloned ARTalk repository.
- **Dynamic Loading**: Modifies `sys.path` and changes the current working directory to load internal ARTalk models (like `Renderer`, FLAME parameters, etc.) correctly without deeply hacking the original repository.
- **Audio Processing**: Aggregates audio chunks into a buffer (`process_audio_chunk`) and processes them to generate implicit neural rendering features using audio feature extraction (e.g., HuBERT).
- **Frame Queueing**: Places generated visual frames into a thread-safe `queue.Queue` to be picked up by the `ARTalkAvatarSession`.
- **Style Management**: Supports dynamic updating of speaking styles (`update_style`) via PyTorch embeddings.
