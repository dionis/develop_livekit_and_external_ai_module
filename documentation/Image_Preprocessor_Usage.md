# ARTalk Avatar Preprocessor (`prepare_artalk_avatar.py`)

This tool allows you to easily process a new avatar image (either a local file or a URL) so that it can be used by the ARTalk model. It runs the `GAGAvatar_track` pipeline to extract FLAME configuration from the image and saves the results directly into ARTalk's `tracked.pt` database.

## Prerequisites

1.  Make sure you have installed the ARTalk plugin or its dependencies.
2.  You must know the absolute or relative path to your ARTalk repository installation (usually `./external_models/ARTalk`).

## Usage Instructions

You can run the script from the root of the `livekit-plugins-artalk` repository.

### 1. Process a Local Image

```bash
python prepare_artalk_avatar.py "path/to/my_image.jpg" --artalk_path "./external_models/ARTalk"
```

### 2. Process an Image from a URL

```bash
python prepare_artalk_avatar.py "https://example.com/photography.png" --artalk_path "./external_models/ARTalk"
```

### Script Arguments

*   `image_input`: (Required) The path to the local image (`.jpg`, `.png`) or URL (`http://` or `https://`).
*   `--artalk_path`: (Required) The path to the original ARTalk repository.
*   `--device`: (Optional) The device to run the tracker on (default: `cuda`).
*   `--no_matting`: (Optional) Skip the background matting (removal) step. It is faster but may include artifacts in the output.

## Automatic Image Validations

When using this script via the ARTalk microservice (`/v1/avatar/create`), several automatic checks are performed **before** the tracking process begins:

1.  **Resolution Check:** The image must be at least **256x256 pixels**.
2.  **Sharpness Check:** Uses Laplacian variance to ensure the image is not too blurry (minimum score: 50.0).
3.  **Face Detection:** Uses MediaPipe (or OpenCV Haar fallback) to confirm a human face is present in the image.

If any of these fail, the service will return a detailed error message in English. Ensure your input images are clear, well-lit, frontal portraits for the best results.

## How to Use the Processed Avatar

Once the script finishes successfully, it will output a generated **Avatar ID** (e.g., `my_image.jpg`).

### In a LiveKit Agent

You can use the generated ID directly in your agent code:

```python
from livekit.plugins.artalk import ARTalkAvatarSession

# Pass the ID printed by the command
avatar = ARTalkAvatarSession(
    shape_id="my_image.jpg", 
    # ... other agent parameters
)
```

### In the Original ARTalk Gradio App (`inference.py`)

If you run the original ARTalk testing interface:

```bash
cd external_models/ARTalk
python inference.py --run_app
```

The ID of your image will automatically appear in the "Choose the appearance of the speaker" dropdown menu under the "Avatar Control" section.

*   **Tip:** If you also want your image to appear as one of the clickable example boxes at the bottom, edit the `inference.py` source code and add your setup to the `examples` list (around line ~178):
    ```python
    ["Audio", "...", None, None, "my_image.jpg", "natural_0"]
    ```
