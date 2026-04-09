# ARTalk Examples

This directory contains ready-to-run example scripts for the `livekit-plugins-artalk` package.

## Architecture Overview

```
┌──────────────────────────────┐         ┌─────────────────────────────────┐
│      GPU Machine             │         │       Brain Agent Machine        │
│                              │         │                                  │
│  install_server.sh           │  HTTP   │  install_client.sh               │
│  ┌─────────────────────┐    │◄───────►│  ┌──────────────────────────┐   │
│  │ artalk_server/       │    │         │  │ livekit.plugins.artalk   │   │
│  │ FastAPI :8000        │    │  WebRTC │  │ AvatarSession            │   │
│  │ ARTalk model         │◄──────────►│  │ example_microservice_agent│   │
│  │ diff-gaussian-rast.  │    │         │  └──────────────────────────┘   │
│  └─────────────────────┘    │         │                                  │
└──────────────────────────────┘         └─────────────────────────────────┘
```

**Server** (`install_server.sh`) — runs on the GPU node. Hosts the ARTalk model, renders the video avatar, and joins the LiveKit room as a video participant.

**Client** (`install_client.sh`) — runs on any machine (no GPU needed). Runs the LiveKit Agents "brain" that does STT → LLM → TTS and coordinates with the server via HTTP + DataChannel.

---

## Prerequisites

### 1. Run the Correct Installer First

| Machine | Command |
|---|---|
| GPU server | `chmod +x install_server.sh && ./install_server.sh` |
| Brain agent | `chmod +x install_client.sh && ./install_client.sh` |

Both scripts ask you to choose between:
- **Option 1** — Standard Linux (conda / uv `.venv`)
- **Option 2** — Lightning.ai Studio (`cloudspace`, pip `--system`)

### 2. Edit `.env`

A `.env` template is generated automatically by each installer. Fill in **all** values before running any example.

#### Server `.env` (GPU machine)
```dotenv
ARTALK_PATH=/path/to/external_models/ARTalk
ARTALK_MODEL_STRATEGY=from_scratch   # or: example_models

ARTALK_SERVER_HOST=0.0.0.0
ARTALK_SERVER_PORT=8000

LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
```

#### Client `.env` (brain machine)
```dotenv
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret

ARTALK_SERVER_URL=http://<gpu-machine-ip>:8000
ARTALK_REPLICA_ID=mesh               # or a custom ID from create_avatar_replica.py

OPENAI_API_KEY=your_openai_key
CARTESIA_API_KEY=your_cartesia_key
```

> **TIP** — Both machines can share the same `.env` file when testing locally.

---

## Example Scripts

### 1. `start_artalk_server.py` — Start the GPU Server

**Run on the GPU machine** after `install_server.sh`:

```bash
# Standard (activate venv first):
source .venv/bin/activate
python examples/start_artalk_server.py

# Lightning.ai (no activation needed):
python examples/start_artalk_server.py
```

The server binds to `ARTALK_SERVER_HOST:ARTALK_SERVER_PORT` (default `0.0.0.0:8000`).  
Keep it running; all other examples below connect to it via `ARTALK_SERVER_URL`.

**Environment variables read:**

| Variable | Default | Description |
|---|---|---|
| `ARTALK_SERVER_HOST` | `0.0.0.0` | Bind address |
| `ARTALK_SERVER_PORT` | `8000` | HTTP port |
| `LIVEKIT_URL` | — | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | — | LiveKit API secret |
| `ARTALK_PATH` | `external_models/ARTalk` | Path to ARTalk repo |
| `ARTALK_MODEL_STRATEGY` | `from_scratch` | `from_scratch` or `example_models` |

---

### 2. `create_avatar_replica.py` — Register a Custom Avatar

Creates a personalized 3D avatar from a photo by sending it to the GPU server for preprocessing.  
Run this **once** from any machine that can reach the server.

```bash
# Set the URL to your source photo (local path or HTTPS URL):
export AVATAR_IMAGE_URL="https://example.com/my_face.jpg"
export ARTALK_SERVER_URL="http://<gpu-machine-ip>:8000"

python examples/create_avatar_replica.py
```

The script prints a **replica ID** — save it in `ARTALK_REPLICA_ID` in your `.env`.

**Environment variables read:**

| Variable | Default | Description |
|---|---|---|
| `ARTALK_SERVER_URL` | `http://localhost:8000` | GPU server address |
| `AVATAR_IMAGE_URL` | GitHub default avatar | Source image URL or local path |

**Example output:**
```
✅ Avatar successfully processed and created!
🆔 Avatar ID (replica_id): abc123def456
📊 Quality (PSNR): 30.0
```

---

### 3. `example_microservice_agent.py` — Microservice Architecture Agent ⭐

The **recommended** example for production. Runs the brain agent on any machine (no GPU needed) while the ARTalk server handles avatar rendering on the GPU machine.

```bash
# Standard (venv):
source .venv/bin/activate
python examples/example_microservice_agent.py start

# Lightning.ai:
python examples/example_microservice_agent.py start
```

**Flow:**
1. Agent connects to LiveKit room
2. Agent creates an `AvatarSession` that calls the GPU server via `ARTALK_SERVER_URL`
3. GPU server joins the same room as a video participant and starts streaming the avatar
4. TTS audio is forwarded from the agent to the GPU server via a DataChannel
5. The avatar's lips are synchronized in real-time

**Environment variables read:**

| Variable | Default | Description |
|---|---|---|
| `LIVEKIT_URL` | — | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | — | LiveKit API secret |
| `ARTALK_SERVER_URL` | `http://localhost:8000` | GPU server address |
| `ARTALK_REPLICA_ID` | `mesh` | Avatar replica ID (from step 2 or `"mesh"`) |
| `OPENAI_API_KEY` | — | For STT + LLM |

---

### 4. `basic_agent.py` — Embedded (Single-Machine) Agent

Runs **everything on one machine**: the LiveKit agent and ARTalk inference in the same process. Requires a GPU and a full server installation (`install_server.sh`).

> Use this for quick local experiments. For production, prefer `example_microservice_agent.py`.

```bash
# Standard (venv):
source .venv/bin/activate
python examples/basic_agent.py start

# Lightning.ai:
python examples/basic_agent.py start
```

**Environment variables read:**

| Variable | Default | Description |
|---|---|---|
| `LIVEKIT_URL` | — | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | — | LiveKit API secret |
| `ARTALK_PATH` | `../external_models/ARTalk` | Path to ARTalk repo |
| `ARTALK_MODEL_STRATEGY` | `from_scratch` | `from_scratch` or `example_models` |
| `ARTALK_STYLE_ID` | `natural_0` | Speaking style ID |
| `AVATAR_IMAGE` | `mesh` | `"mesh"` or absolute path to `.png`/`.jpg` |
| `OPENAI_API_KEY` | — | For STT + LLM |
| `CARTESIA_API_KEY` | — | For TTS voice |

---

## Quick-Start Checklist

```
[ ] 1. Run install_server.sh on GPU machine
[ ] 2. Edit .env on the GPU machine (LIVEKIT_*, ARTALK_PATH)
[ ] 3. Start the server:   python examples/start_artalk_server.py
[ ] 4. (Optional) Create a custom avatar:
         python examples/create_avatar_replica.py
[ ] 5. Run install_client.sh on the brain machine
[ ] 6. Edit .env on the brain machine (LIVEKIT_*, ARTALK_SERVER_URL, API keys)
[ ] 7. Start the agent:    python examples/example_microservice_agent.py start
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` on `ARTALK_SERVER_URL` | Ensure the GPU server is running and the port is reachable |
| `torch not found` on client machine | The client installer does NOT install torch — this is expected |
| `DiagnosticOptions` import error | Re-run `install_server.sh` — it auto-detects and fixes the onnx/torch conflict |
| `cstdint` compile error | `install_server.sh` applies the fix automatically to `rasterizer_impl.h` |
| `ARTALK_PATH` missing | Set `ARTALK_PATH` in `.env` to the absolute path of `external_models/ARTalk` |
| Custom avatar not found | Run `create_avatar_replica.py` and store the printed replica ID in `ARTALK_REPLICA_ID` |
