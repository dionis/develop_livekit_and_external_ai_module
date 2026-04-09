# Microservice Architecture Guide for ARTalk

This guide documents the newly implemented architecture to completely decouple ARTalk from the main flow of the LiveKit Agent, mimicking at a technical level the architecture used by external platforms like Tavus.

---

## 🏗️ 1. Architecture Overview

To maintain optimum performance for Voice Agents ("The Brain") and to easily scale the heavy 3D Avatar rendering payload ("The Body"), the system has been divided into two separate parallel worlds that communicate asynchronously via standard APIs and WebRTC (LiveKit).

1. **The ARTalk Server (The Muscle - GPU):** An independent FastAPI server (`artalk_server/`) meant to run on a dedicated machine with powerful GPUs.
2. **The Client Plugin (The Brain - CPU):** An extremely lightweight plugin in `livekit/plugins/artalk/` that simply dispatches HTTP requests and forwards audio, but never processes a single frame of video itself.

---

## ⚙️ 2. The FastAPI Server (`artalk_server`)

### What does it do?
It is the equivalent of Tavus' private backend servers. It contains three main files:
- `main.py`: Exposes the REST endpoints.
- `models.py`: Pydantic models for I/O validation.
- `worker.py`: The magic happens here. It is a **Headless WebRTC Client**. It connects to the LiveKit room, listens for injected audio, and natively publishes the resulting generated video using the legacy `ARTalkSDKWrapper`.

### How to start it?
You can use the convenience script we created in `examples/start_artalk_server.py`. This script reads the environment variables (`ARTALK_SERVER_HOST`, `ARTALK_SERVER_PORT` and LiveKit credentials) and automatically starts the server.

Ideally on your GPU machine ("Worker node"):

```bash
cd E:\PROJECTS\PROJECT_BRAIN-AIX_VANCOUVER\SOURCE\livekit-plugins-artalk\
pip install fastapi uvicorn pydantic

# Start using the programmatic example script:
python examples/start_artalk_server.py
```

### Endpoints
- **`POST /v1/avatar/create`**: Receives an image URL, uses the legacy processor (`prepare_artalk_avatar.py`), and returns the generated `replica_id` alongside visual quality metrics (PSNR/SSIM).
- **`POST /v1/conversation`**: Receives the `replica_id` and the LiveKit Room credentials. Launches the worker in the background to inject the avatar video into the room.

---

## 🔌 3. The Client Plugin (`artalk`)

This is the new elegant library you import inside your Voice Agent code.

### Where does it live?
`livekit-plugins-artalk/livekit/plugins/artalk/` (To avoid polluting or deleting your old `artalk` plugin).

### Main Files
- `api.py`: Contains `ARTalkAPI`, an HTTP client (`aiohttp`) that talks to your backend on port 8000.
- `avatar.py`: Contains `AvatarSession`. It overrides your Agent's Audio Output so that instead of broadcasting it "on air" (where it would cause echo/interference), it gets sent directly privately to the GPU Worker using DataChannels (`DataStreamAudioOutput`).

### Code Examples (`examples/` folder)

We have created multiple scripts to facilitate usage and testing of the new architecture:

1. **`examples/create_avatar_replica.py`**: This script connects solely to your background GPU server to process an image and generate the Avatar (`preprocess_avatar_image`), returning the Avatar ID and its quality metrics immediately, without joining a LiveKit Room.
2. **`examples/example_microservice_agent.py`**: A complete script that acts as the Voice Agent (the brain), remotely creates the avatar, asks the GPU server to connect to the current LiveKit Room as a virtual participant, and injects your native TTS audio into the avatar using `AvatarSession.start()`.

You can test the full integration by running the latter:

```python
import asyncio
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins.artalk import ARTalkAPI, AvatarSession

async def entrypoint(ctx: JobContext):
    # 1. Start the API Client (point to your GPU FastAPI server HTTP port)
    api = ARTalkAPI(api_url="http://localhost:8000")
    
    print("Creating replica on the ARTalk Backend Cloud...")
    replica = await api.create_replica("https://photo.com/avatar.png")
    print(f"Avatar ID: {replica.replica_id}")

    # 2. Start the Session
    avatar = AvatarSession(replica_id=replica.replica_id, api_url="http://localhost:8000")
    
    # This automatically tricks the GPU Backend into joining your room with a special JWT,
    # and hooks your Audio down through Data Channels, exactly like the Tavus plugin.
    await avatar.start(ctx.agent, ctx.room)
    
    # Done. Your voice agent can continue using OpenAI, Cartesia, etc...
```

> [!IMPORTANT]
> **About passing the Session to the Avatar**: When you call `avatar.start(agent_session=session, ...)`, it is critical that you pass your Pipeline or Brain object (e.g., `AgentSession` or `VoicePipelineAgent`) and **NOT** the `ctx.agent` object (which is just the `LocalParticipant`). The avatar needs to hijack the audio by extracting it directly from your Pipeline's TTS generator before it is broadcast over the network.
>
> Likewise, when starting your final session with `session.start()`, you must pass an instantiated subclass of your agent (e.g., `agent=YourCustomVoiceAgent()`) using keyword arguments, instead of the empty base class `Agent()`, to prevent strict typing overload errors in linters like Pylance.

---

## ✅ Logic Separation Summary

| Module | Before | Now (Microservice Architecture) |
| :--- | :--- | :--- |
| **FLAME / GAGAvatar Generation** | Blocked the Voice Agent thread (GIL). | Runs isolated in `artalk_server/worker.py` on another process/machine. |
| **Video Sending** | Voice Agent published video track by iterating frames locally. | API Backend (`worker.py`) joins room as virtual participant and natively publishes video to LiveKit SFU. |
| **A/V Sync** | Brute forced by matching local frame counts. | Voice Agent pushes audio bytes via *DataChannel* straight to `worker.py`, which queues and async-processes them flawlessly with its video. |
| **`prepare_artalk_avatar.py`** | Manual local CLI script. | Imported and wrapped in an HTTP Endpoint (`POST /v1/avatar/create`). |
