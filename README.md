# livekit-plugins-artalk

LiveKit Agents plugin for ARTalk 3D TalkingHead avatar integration, including a FastAPI microservice GPU server and a lightweight client plugin.

---

## Installation

The project uses a **two-component architecture**. Each component has its own dedicated installer:

| Installer | Machine | Purpose |
|---|---|---|
| `install_server.sh` | **GPU machine** | ARTalk model, CUDA extensions, model weights, FastAPI server |
| `install_client.sh` | **Brain agent machine** (no GPU needed) | LiveKit Agents plugin only |

Both installers support two target environments:

| Option | Description |
|---|---|
| **1 — Standard Linux** | Local machine, VPS, or cloud VM with Miniconda / `uv` |
| **2 — Lightning.ai** | Lightning.ai Studio (`cloudspace`, pip `--system`, no conda) |

### GPU Server (ARTalk model + FastAPI)

```bash
chmod +x install_server.sh
./install_server.sh         # choose option 1 (Standard) or 2 (Lightning.ai)
```

Installs: ARTalk repository, `diff-gaussian-rasterization` CUDA extension, `pytorch3d` (Lightning.ai only), ARTalk + GAGAvatar_track Python dependencies, PyTorch 2.4.1 + CUDA 12.1, model weights, and a `.env` template.

### Brain Agent (LiveKit plugin only)

```bash
chmod +x install_client.sh
./install_client.sh         # choose option 1 (Standard) or 2 (Lightning.ai)
```

Installs: `livekit-plugins-artalk` and all LiveKit agent dependencies into `.venv` (Standard) or system python (Lightning.ai). **No GPU or model weights required.**

> **Legacy single-machine installers** — The older `install_artalk.sh` (Standard) and `install_artalk_lightning.sh` (Lightning.ai) scripts are retained for reference. They install the standalone ARTalk model only (no LiveKit plugin). Use the new scripts above for the full microservice stack.

---

## Quick-Start

```bash
# 1. GPU machine — install and start server
./install_server.sh
# (edit .env: LIVEKIT_URL, keys, ARTALK_PATH)
python examples/start_artalk_server.py

# 2. Brain machine — install and start agent
./install_client.sh
# (edit .env: LIVEKIT_URL, keys, ARTALK_SERVER_URL)
python examples/example_microservice_agent.py start
```

See **[examples/README.md](examples/README.md)** for full usage instructions, environment variable reference, and a troubleshooting guide.

---

---

## What is ARTalk?

**ARTalk** is a 3D head animation model driven by audio. It generates realistic real-time lip sync, blinking, facial expressions, and head poses from a given audio input.

| Resource | Link |
|---|---|
| Official repository | https://github.com/xg-chu/ARTalk |
| CUDA extension repo | https://github.com/xg-chu/diff-gaussian-rasterization |
| Paper (arXiv) | https://arxiv.org/abs/2502.20323 |
| Project page / demos | https://xg-chu.site/project_artalk/ |

**Authors:** Xuangeng Chu, Nabarun Goswami, Ziteng Cui, Hanqin Wang, Tatsuya Harada — University of Tokyo / RIKEN AIP.

---

## What do the scripts install?

| Component | Description |
|---|---|
| **ARTalk** | Main repository with the model, inference pipeline and Gradio app |
| **Python dependencies** | From `environment.yml` — via conda env (standard) or pip (Lightning.ai) |
| **diff-gaussian-rasterization** | CUDA extension for realistic avatar rendering (Gaussian Splatting) |
| **Model weights and assets** | Downloaded via `build_resources.sh`, includes FLAME models |

---

## Prerequisites

Before running either script, make sure you have:

- **Linux** (Ubuntu 20.04+ recommended)
- **NVIDIA GPU** with drivers installed
- **CUDA Toolkit** (`nvcc` available in PATH)
- **Git** with submodule support (`git >= 2.13`)
- Internet connection to download model weights and assets

For `install_artalk.sh` only:
- **Miniconda or Anaconda** installed and available in your shell

---

## How to run

### Standard (local / VPS / cloud VM)

```bash
chmod +x install_artalk.sh
./install_artalk.sh
```

### Lightning.ai Studio

```bash
chmod +x install_artalk_lightning.sh
./install_artalk_lightning.sh
```

Both scripts run **6 steps sequentially** and may take between 10 and 30 minutes depending on internet speed and GPU.

---

## Script steps (common to both)

### Step 0 — Verify environment *(Lightning.ai script only)*
Confirms the script is running inside a Lightning.ai Studio by checking the conda restriction warning. If not on Lightning.ai, it exits with an error and points to the correct script.

---

### Step 1 — Clone ARTalk
Clones the main repository with all its submodules using `--recurse-submodules`.
Uses HTTPS instead of SSH (unlike the official README) so no configured SSH keys are required.

```bash
git clone --recurse-submodules https://github.com/xg-chu/ARTalk.git
```

> If the `ARTalk/` directory already exists, this step is skipped automatically.

---

### Step 2 — Install dependencies

**Standard script:** creates and activates a dedicated conda environment `ARTalk` from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ARTalk
```

**Lightning.ai script:** parses `environment.yml` and installs all packages via `pip` directly into `cloudspace`. Conda packages that have no pip equivalent are skipped gracefully with a warning.

```bash
# pip packages from environment.yml pip section
pip install <pip-packages>
# conda packages installed via pip (best-effort)
pip install <conda-packages>
```

---

### Step 3 — Fix: `onnx` / `onnx2torch` conflict with PyTorch
Proactively detects and fixes the following error:

```
ImportError: cannot import name 'DiagnosticOptions'
from 'torch.onnx._internal.exporter'
```

This error occurs because recent versions of PyTorch removed `DiagnosticOptions` from `torch.onnx._internal.exporter`, but `onnx2torch` still tries to import it. If detected, the script uninstalls the conflicting packages and reinstalls clean versions.

```bash
pip uninstall -y torch torchvision torchaudio onnx onnx2torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 3b — Ensure torchaudio is installed
Independently verifies that `torchaudio` is importable. Detects the installed torch version and CUDA tag to build the correct wheel index URL and install a fully compatible version. Uses 3 fallback strategies before failing.

```bash
# Example for torch 2.4.1 + CUDA 12.1:
pip install torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 4 — Install `diff-gaussian-rasterization` (with CUDA fix)
Clones and installs the CUDA extension for Gaussian Splatting avatar rendering.

**Applied fix: `#include <cstdint>` in `rasterizer_impl.h`**

The file `cuda_rasterizer/rasterizer_impl.h` uses standard C++ integer types (`uint32_t`, `uint64_t`, `std::uintptr_t`) without including the header that defines them. This causes 8 compilation errors with `nvcc`. The file starts with a multi-line copyright comment block followed by `#pragma once` — inserting the include at line 1 places it inside the comment where the compiler ignores it.

**Correct fix:** insert `#include <cstdint>` immediately **after `#pragma once`**:

```bash
sed -i '/#pragma once/a #include <cstdint>' cuda_rasterizer/rasterizer_impl.h
pip install --no-cache-dir --force-reinstall .
```

---

### Step 5 — Prepare model resources
Runs `build_resources.sh` which downloads model weights, FLAME assets and other required files. Uses `yes |` to automatically confirm all license prompts.

```bash
yes | bash build_resources.sh
```

> By running this script you are accepting the FLAME license terms.
> Review them at: https://flame.is.tue.mpg.de

---

### Step 6 — Patch `inference.py` (Gradio `share=True`)
Replaces the fixed-port Gradio launch with a public tunnel URL. Required on all cloud environments where port 8960 is not externally accessible.

```python
# Before:
demo.launch(server_name="0.0.0.0", server_port=8960)

# After:
#demo.launch(server_name="0.0.0.0", server_port=8960)
demo.launch(share=True)
```

Gradio generates a public URL like `https://xxxxxxxx.gradio.live`, valid for 72 hours.

---

## Usage after installation

### Standard script
```bash
conda activate ARTalk
cd ARTalk
python inference.py --run_app
```

### Lightning.ai script
```bash
# No conda activate needed — cloudspace is always active
cd ARTalk
python inference.py --run_app
```

**Command line (both environments):**
```bash
python inference.py \
  -a your_audio.wav \
  --shape_id mesh \
  --style_id default \
  --clip_length 750
```

| Parameter | Description |
|---|---|
| `-a` | Path to the input audio file |
| `--shape_id` | Avatar shape: `mesh` or a tracked avatar stored in `tracked.pt` |
| `--style_id` | Name of a `.pt` file inside `assets/style_motion/` |
| `--clip_length` | Maximum video duration in frames (default: 750) |

---

## Avatar Preprocessing

To use your own custom portraits as avatars (`shape_id`), you must first process them using the `prepare_artalk_avatar.py` script provided in the root of this plugin repository. This script automates the GAGAvatar face tracking and FLAME extraction process.

**Usage:**
```bash
# Using a local image with quality evaluation
python prepare_artalk_avatar.py "path/to/my_image.jpg" --artalk_path "./external_models/ARTalk" --eval

# Or from a URL
python prepare_artalk_avatar.py "https://example.com/photography.png" --artalk_path "./external_models/ARTalk"
```

### Avatar Image Validations

When creating an avatar via the `/v1/avatar/create` endpoint, the system automatically performs several pre-flight validations to ensure the image is suitable for ARTalk processing. This avoids wasting computation on unusable images.

| Validation | Requirement | Failure Code |
|---|---|---|
| **Image Resolution** | Minimum **256×256 pixels** for both width and height. | `400 Bad Request` |
| **Image Sharpness** | Uses Laplacian variance to detect blur. Minimum score of **50.0** required. | `400 Bad Request` |
| **Face Detection** | Must contain at least one detectable human face. | `422 Unprocessable Entity` |

**Face Detection Strategy:** The system uses **MediaPipe FaceDetection** (if available) and falls back to **OpenCV Haar cascades** for maximum robustness across different environments.

If any of these validations fail, the service returns a descriptive **English error message** explaining the requirement.

### Quality Evaluation
The script will output an **Avatar ID** (e.g., `my_image.jpg`). You can then use this ID as the `--shape_id` when running `inference.py`, or directly drop it into your LiveKit Agent's `ARTalkAvatarSession`.

When using the `--eval` flag, the script will automatically calculate and display metrics comparing the original image with the processed 3D representation:
- **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality (aim for >20dB).
- **SSIM (Structural Similarity Index):** Evaluates structural consistency (aim for >0.70).

For more details on these metrics, see the [ARTalk Metrics and Evaluation](documentation/ARTalk_Metrics_and_Evaluation_EN.md) guide.

For more details, see the guides in the `documentation` folder.

---

## Summary of changes from the official README

| # | Change | Applies to |
|---|---|---|
| 1 | HTTPS instead of SSH for `git clone` | Both |
| 2 | No conda env creation — pip install into `cloudspace` | Lightning.ai only |
| 3 | Detection and fix of the `DiagnosticOptions` import error | Both |
| 4 | `#include <cstdint>` inserted **after `#pragma once`** | Both |
| 5 | `pip install --no-cache-dir --force-reinstall` | Both |
| 6 | `yes \|` piped into `build_resources.sh` | Both |
| 7 | `demo.launch(share=True)` in `inference.py` | Both |

---

## Roadmap

- [ ] **[User Emotion Analysis](#user-emotion-analysis)**: Implement real-time recognition of user emotions using video and audio streams to make the avatar more reactive and empathetic.
- [ ] **[Commercial Viability & Alternative Models](#commercial-viability--alternative-models)**: Research and implementation of permissively licensed models for SaaS products.
- [ ] **[Distributed Avatar Cooking Architecture](#distributed-avatar-cooking-architecture)**: Decouple the avatar creation (FLAME feature extraction) from the real-time LiveKit streaming to allow horizontal scaling and reduce GPU VRAM bottlenecks.

---

## User Emotion Analysis

This feature aims to provide the avatar with the ability to understand the user's emotional state in near real-time by processing incoming video and audio tracks.

### How it works
The system subscribes to the user's media tracks, samples the raw frames, and uses specialized AI models (like Computer Vision for facial expressions and Speech Emotion Recognition for vocal tone) to classify emotions.

For a full technical explanation and implementation options, see the dedicated guides:
- 🇬🇧 **[User Emotion Analysis (English)](documentation/Emotion_Analysis_EN.md)**
- 🇪🇸 **[Análisis de Emociones del Usuario (Español)](documentation/Emotion_Analysis_ES.md)**

---

## Commercial Viability & Alternative Models

The current ARTalk implementation relies on components with restrictive licenses (**Inria** for rendering and **FLAME** for face topology) that prohibit direct commercial use (SaaS, paid products, or for-profit derivatives).

### Roadmap for Commercialization

To make ARTalk legally viable for commercial products, the following research and development paths are suggested:

1. **Permissive 3D Rendering:**
   - Replace `diff-gaussian-rasterization` (Inria) with an MIT or Apache 2.0 licensed implementation.
   - **Candidates:** [gsplat](https://github.com/vuer-ai/gsplat) (Apache 2.0) or custom CUDA-based rasterizers that do not derive from the Inria codebase.

2. **Open-Source Face Models:**
   - Search for or train a parametric face model to replace **FLAME**.
   - **Candidates:** Investigate data-driven mesh representations or models with more permissive licenses (avoiding the academic-only restrictions of SMPL-X/FLAME).

3. **Commercial Model Licensing:**
   - For immediate commercial deployment, contact the **Max Planck Institute** for a commercial FLAME license and **Inria** for the rasterizer.

4. **Independent Weights Training:**
   - Re-train the motion and style encoders using datasets and base models that allow for commercial derivation.

For detailed analysis, refer to:
- 🇬🇧 **[ARTalk Licenses Summary (English)](documentation/ARTalk_Licenses_Summary_EN.md)**
- 🇪🇸 **[Resumen de Licencias ARTalk (Español)](documentation/ARTalk_Licenses_Summary_ES.md)**

---

## Distributed Avatar Cooking Architecture

To scale from a monolithic single-node deployment to a robust production environment, the process of "cooking" an avatar (extracting 3D FLAME parameters from a 2D image) must be decoupled from the LiveKit real-time streaming process.

### 1. Why Cooking is Necessary
The ARTalk model requires a volumetric 3D understanding of the face. The cooking process runs a specialized tracking engine (`GAGAvatar_track`) to isolate the person from the background (matting), fit a 3D parametric face model (FLAME) to extract exact identity/texture parameters, and compress this into PyTorch tensors stored locally as `assets/GAGAvatar/tracked.pt`. Without this 3D "mold", the real-time renderer cannot accurately project vocal and facial movements.

### 2. GPU Constraints & Capacity
Both cooking and streaming are heavily dependent on GPU VRAM:
*   **Cooking Process:** Requires ~2-4GB VRAM and loads FLAME models temporally.
*   **Live Streaming:** Each active LiveKit avatar consumes ~3-5GB VRAM sustained.
*   **Capacity Estimate:** A consumer GPU (e.g., RTX 3090/4090) or mid-tier enterprise GPU (A10G/L4) can host 4 to 6 concurrent avatars. An 80GB high-end GPU (A100/H100) can scale up to 15-20+ concurrent avatars.

### 3. Performance Tips
*   **Asynchronous Processing:** Never cook avatars on-demand during room connection. Use background workers to cook images when users upload their profiles.
*   **Disable Matting (`no_matting=True`):** If users upload photos with solid backgrounds, disabling matting reduces processing time massively (~40%) and limits VRAM spikes.
*   **Memory Management:** Always call `torch.cuda.empty_cache()` after cooking to flush residual FLAME models from VRAM.
*   **Crop and Scale:** Standardize input images (e.g., 512x512) prior to processing to avoid wasting cycles on 4K images.

### 4. How to Distribute the Architecture
The monolithic setup where `tracked.pt` is stored locally must be redesigned for horizontal scaling:
1.  **"Cooker" Nodes (Batch Workers):** Dedicated, cheaper GPUs (e.g., T4/L4) exposed via a `/v1/avatar/create` endpoint or a message queue. They only process images into tensors.
2.  **Centralized Storage (Redis/PostgreSQL):** Instead of saving `tracked.pt` to disk via `torch.save()`, serialize the PyTorch tensors and store them in a high-speed centralized database.
3.  **Streaming Nodes (LiveKit Workers):** High-end GPUs (e.g., A10G/A100) dedicated exclusively to rendering. When a session starts for an `avatar_id`, the worker fetches the pre-processed serialized tensors from the centralized Redis database and injects them into the renderer.

For the full detailed documentation in both languages, refer to:
- 🇬🇧 **[ARTalk Avatar Cooking Process (English)](documentation/Avatar_Cooking_EN.md)**
- 🇪🇸 **[Proceso de Cocinado de Avatares en ARTalk (Español)](documentation/Avatar_Cooking_ES.md)**

---

## License

These installation scripts are free to use. ARTalk is distributed under the **MIT license**.
FLAME has its own academic license available at https://flame.is.tue.mpg.de.

---

## Dynamic Backgrounds

The ARTalk microservice supports replacing the default black background of the generated avatar with a custom scene image. Compositing happens entirely server-side before streaming, so no client changes are needed.

### Available Scenes and Sources

The `background_scene` parameter is highly flexible and accepts three types of values (evaluated in order):

1. **Built-in Scene Name:** A short name (without `.png`) for any image located in the `scenes/` directory.
   - Example: `"office"`, `"beach"`, `"popular_street"`.
2. **Local File Path:** An absolute or relative path to a valid JPG/PNG image on the server.
   - Example: `"/home/user/custom_backgrounds/garden.jpg"`.
3. **HTTP/HTTPS URL:** A publicly accessible image URL. The server will download it during session initialization.
   - Example: `"https://example.com/my_background.png"`.

| Scene name | File |
|---|---|
| `office` | `scenes/office.png` |
| `beach` | `scenes/beach.png` |
| `popular_street` | `scenes/popular_street.png` |

### Configuration

**Via environment variable** (applies to all sessions):
```env
AVATAR_SCENE=office
AVATAR_BG_THRESHOLD=15
```

**Via API** (per session, overrides env vars):
```json
{
  "replica_id": "your_avatar.jpg",
  "background_scene": "beach",
  "bg_threshold": 12,
  "properties": { "livekit_ws_url": "...", "livekit_room_token": "..." }
}
```

### Edge Quality — `bg_threshold`

Controls how aggressively the black background is removed. ARTalk renders on pure black, so any pixel brighter than the threshold is kept as avatar.

| Value | Effect |
|---|---|
| **3–8** | Very tight. Preserves dark hair and shadows but may leave thin black halos. |
| **10–20** | ✅ **Recommended.** Good balance. Default: **15**. |
| **25–40** | Cleaner edges, but dark hair or collar may erode. |
| **> 40** | Too aggressive — avatar parts start disappearing. |

**Tuning tips:**
- Black halo visible around the silhouette → **lower** the threshold (try `8`)
- Dark hair or clothing disappears → **lower** the threshold (try `8` or `5`)
- Black specks appear inside the background scene → **raise** the threshold (try `20`)

For full details and the algorithm description see [`documentation/Dynamic_Background_Guide_EN.md`](documentation/Dynamic_Background_Guide_EN.md).
