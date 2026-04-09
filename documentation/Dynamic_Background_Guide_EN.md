# Dynamic Backgrounds for ARTalk Avatar — Configuration Guide

## Overview

The ARTalk server supports replacing the default black background of the avatar with custom scene images. Background compositing happens **server-side before streaming**, so no client changes are required.

When MediaPipe is unavailable (the common case in cloud environments), the system automatically falls back to an **OpenCV Brightness Threshold** algorithm that uses the fact that ARTalk always renders on a pure black `(0, 0, 0)` background to separate avatar pixels from background pixels.

---

### Flexible Background Sources

The `background_scene` parameter (and the `AVATAR_SCENE` env var) is evaluated in the following priority order:

1. **HTTP/HTTPS URL:** If the string starts with `http://` or `https://`, the server initializes a download to a temporary file.
2. **Local File Path:** If the path exists on the server's filesystem, it is loaded directly.
3. **Built-in Scene:** The string is treated as a filename (without extension) inside the `scenes/` directory.

---

## Quick Configuration

### Option 1 — Environment Variables (global for all sessions)

Add these to your `.env` file on the server:

```env
# Can be a scene name, absolute path, or URL
AVATAR_SCENE=office

# Edge threshold (integer 1-255). Default: 15. See tuning guide below.
AVATAR_BG_THRESHOLD=15
```

Available default scenes: `office`, `beach`, `popular_street`.  
To add a new permanent scene, drop a `.png` into `scenes/`.

### Option 2 — Per-Session API Parameter

Send `background_scene` and optionally `bg_threshold` in your `POST /v1/conversation` request body:

```json
{
  "replica_id": "your_avatar.jpg",
  "background_scene": "https://example.com/my_custom_bg.jpg",
  "bg_threshold": 12,
  "properties": {
    "livekit_ws_url": "wss://...",
    "livekit_room_token": "..."
  }
}
```

Per-request values override environment variables.

---

## Edge Quality Tuning — `bg_threshold`

The `bg_threshold` parameter is a luminosity cutoff value between **1 and 255**.  
A pixel is classified as **avatar** if its brightness > threshold, otherwise as **background**.

| Value Range | Effect | When to Use |
|-------------|--------|-------------|
| **3 – 8** | Very tight cut. Dark hair, eyebrows and shadows are fully preserved. May leave a thin black halo around the silhouette. | Avatar with very dark hair or wearing dark clothing. |
| **10 – 20** ✅ | **Recommended range.** Good balance between edge cleanliness and dark-detail preservation. Default is **15**. | Most avatars and scenes. Start here. |
| **25 – 40** | Aggressive cut. Crisp, clean edges but may erode dark hair or a dark jacket collar. | Avatar with light hair, light clothing, or where halo artifacts are prominent. |
| **> 40** | Too aggressive. Avatar body parts (hair, beard, shoulders) start being classified as background and disappear. | Not recommended. |

### Step-by-Step Tuning Workflow

1. Start with the **default (`15`)** and observe the result in LiveKit Playground.
2. If you see **black halos** fringing the avatar's outline → **lower** the threshold (try `8`).
3. If the avatar's **hair or dark clothing disappears** → **lower** the threshold (try `8` or `5`).
4. If there are **residual black specks** visible inside the background scene → **raise** the threshold (try `20`).
5. Iterate in steps of 5 until the result looks natural.

### Example: Tuning via `.env` without restarting

If the server is running in development mode with hot-reload, you can adjust `AVATAR_BG_THRESHOLD` in your `.env` file and the next session will pick it up. Otherwise, restart the server process after changing environment variables.

---

## How the Algorithm Works

```
For each frame:
  1. Convert avatar frame (RGB) to grayscale.
  2. Apply binary threshold at bg_threshold level:
       pixel >= threshold  →  foreground (avatar)
       pixel <  threshold  →  background (scene)
  3. Dilate the mask by 1px to close small holes in dark edges.
  4. Apply Gaussian blur (7x7) to soften the silhouette transition.
  5. Alpha-blend: result = avatar × mask + scene_image × (1 − mask)
  6. Output frame as BGRA to LiveKit track.
```

This runs entirely on CPU in ~1–2 ms per frame, adding negligible overhead to the 25 FPS stream.

---

## Adding Custom Background Scenes

1. Place any `.png` image in the `scenes/` directory at the project root.
2. Reference it by its filename without extension:
   - File: `scenes/living_room.png`
   - Usage: `AVATAR_SCENE=living_room` or `"background_scene": "living_room"` in the API.
3. Images are automatically resized to match the avatar resolution (512×512 by default).

> **Tip:** Use images with a similar color temperature and lighting to the avatar for the most natural result.

---

## Error Handling

The microservice performs **pre-flight validation** on the `background_scene` parameter. If a value is provided that is not a well-formed URL, does not exist on disk, or is not in the `scenes/` directory, the API will return a `400 Bad Request` with a descriptive English error message.

If a valid source is provided but fails during the actual loading process (e.g., a URL times out or a file is corrupted), the worker will log the error and **automatically fall back to no background** (pure black) rather than crashing the session.
