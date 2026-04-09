#!/usr/bin/env python
# Copyright (c) Brain-AIX Vancouver
#
# Preprocess a raw portrait image to create ARTalk-compatible FLAME tracking
# data, which is required by GAGAvatar to render a custom avatar.
#
# ARTalk's GAGAvatar model reads face identities from:
#   assets/GAGAvatar/tracked.pt  (key → FLAME param dict)
#
# GAGAvatar_track is the official standalone tracking repo by xg-chu:
#   https://github.com/xg-chu/GAGAvatar_track
#
# This module clones or expects the tracker at a sibling path next to the
# ARTalk repo root, then:
#   1. Adds GAGAvatar_track to sys.path so its `engines` package is importable
#   2. Runs CoreEngine.track_image() on the raw portrait
#   3. Merges the result into assets/GAGAvatar/tracked.pt so GAGAvatar can
#      call set_avatar_id(image_basename)

import os
import sys
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository URL for the standalone tracker
# ---------------------------------------------------------------------------
_TRACKER_REPO = "https://github.com/xg-chu/GAGAvatar_track.git"


def _ensure_tracker(artalk_path: Path) -> Path:
    """
    Return the path to a usable GAGAvatar_track checkout.

    Search order:
      1. external_models/GAGAvatar_track  (setup.sh standard clone location)
      2. ARTalk/libs/GAGAvatar_track       (git submodule fallback)
      3. Sibling of ARTalk: <parent>/GAGAvatar_track
      4. Auto-clone into external_models/GAGAvatar_track
    """
    # external_models sits next to external_models/ARTalk
    candidates = [
        artalk_path.parent / "GAGAvatar_track",          # external_models/GAGAvatar_track
        artalk_path / "libs" / "GAGAvatar_track",         # ARTalk submodule path
        artalk_path.parent.parent / "GAGAvatar_track",    # project-root sibling
    ]
    for candidate in candidates:
        if (candidate / "engines" / "__init__.py").exists():
            logger.info(f"Found GAGAvatar_track at: {candidate}")
            return candidate

    # Auto-clone into the same external_models directory as ARTalk
    clone_target = artalk_path.parent / "GAGAvatar_track"
    logger.info(f"GAGAvatar_track not found – cloning into {clone_target}…")
    subprocess.run(
        ["git", "clone", _TRACKER_REPO, str(clone_target)],
        check=True,
    )
    return clone_target


def preprocess_avatar_image(
    image_path_str: str,
    artalk_path_str: str,
    device: str = "cuda",
    no_matting: bool = False,
) -> str:
    """
    Track a raw portrait image through GAGAvatar_track and inject the result
    into ARTalk's tracked.pt database.

    Parameters
    ----------
    image_path_str : str
        Absolute path to the portrait image (JPG / PNG).
    artalk_path_str : str
        Absolute path to the cloned ARTalk repository root.
    device : str
        torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    no_matting : bool
        Skip human-matting step (faster, but avatar may have background
        artefacts in the output).

    Returns
    -------
    str
        The ``avatar_id`` (filename basename) that was injected into tracked.pt
        and that should be passed as ``shape_id`` to ``ARTalkAvatarSession``.
    """
    import torch
    import torchvision

    image_path = Path(image_path_str).absolute()
    artalk_path = Path(artalk_path_str).absolute()

    if not image_path.exists():
        raise FileNotFoundError(f"Portrait image not found: {image_path}")
    if not image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    avatar_id = image_path.name  # e.g. "alice.jpg"

    # ── 1. Locate / clone GAGAvatar_track ───────────────────────────────────
    tracker_root = _ensure_tracker(artalk_path)

    # ── 2. Inject tracker into sys.path ─────────────────────────────────────
    tracker_root_str = str(tracker_root)
    if tracker_root_str not in sys.path:
        sys.path.insert(0, tracker_root_str)

    # ── 3. Ensure tracker model assets are present ──────────────────────────
    # GAGAvatar_track needs `assets/flame/FLAME_with_eye.pt` and other files.
    # These are distributed via a tarball at HuggingFace (see build_resources.sh).
    tracker_flame_model = tracker_root / "assets" / "flame" / "FLAME_with_eye.pt"
    if not tracker_flame_model.exists():
        logger.info("Downloading GAGAvatar_track model assets (track_resources.tar)…")
        import tarfile
        import urllib.request
        tar_dest = tracker_root / "track_resources.tar"
        hf_tar_url = (
            "https://huggingface.co/xg-chu/GAGAvatar_track/resolve/main/track_resources.tar"
        )
        (tracker_root / "assets").mkdir(parents=True, exist_ok=True)
        logger.info(f"  Downloading {hf_tar_url} …")
        urllib.request.urlretrieve(hf_tar_url, str(tar_dest))
        logger.info("  Extracting track_resources.tar …")
        with tarfile.open(str(tar_dest)) as tf:
            tf.extractall(str(tracker_root / "assets"))
        tar_dest.unlink(missing_ok=True)
        logger.info("  Assets extracted successfully.")

    from engines import CoreEngine  # noqa: E402  (must be after sys.path injection)

    orig_cwd = os.getcwd()
    try:
        os.chdir(tracker_root)  # CoreEngine resolves asset paths relative to CWD

        logger.info("Initializing GAGAvatar_track CoreEngine…")
        tracker = CoreEngine(focal_length=12.0, device=device)

        # ── 4. Load & track the image ────────────────────────────────────────
        img_tensor = (
            torchvision.io.read_image(
                str(image_path),
                mode=torchvision.io.ImageReadMode.RGB,
            )
            .to(device)
            .float()
        )

        logger.info(f"Tracking portrait {avatar_id} with GAGAvatar_track…")
        track_results = tracker.track_image(
            [img_tensor],
            [avatar_id],
            if_matting=not no_matting,
        )

        if track_results is None or avatar_id not in track_results:
            raise RuntimeError(
                f"GAGAvatar_track could not detect a face in {avatar_id}. "
                "Please use a clear frontal portrait with good lighting."
            )

        result = track_results[avatar_id]

        # ── 5. Convert arrays to tensors ─────────────────────────────────────
        import numpy as np

        for key in list(result.keys()):
            val = result[key]
            if isinstance(val, np.ndarray):
                result[key] = torch.tensor(val).float()
            elif isinstance(val, torch.Tensor):
                result[key] = val.float()

        # Remove non-serialisable debug images before saving
        result.pop("vis_image", None)
        # result.pop("image", None)  # DO NOT REMOVE: REQUIRED BY GAGAvatar

    finally:
        os.chdir(orig_cwd)

    # ── 6. Merge into ARTalk's tracked.pt ────────────────────────────────────
    tracked_pt_path = artalk_path / "assets" / "GAGAvatar" / "tracked.pt"
    tracked_pt_path.parent.mkdir(parents=True, exist_ok=True)

    if tracked_pt_path.exists():
        existing = torch.load(str(tracked_pt_path), map_location="cpu", weights_only=False)
    else:
        existing = {}

    if avatar_id in existing:
        logger.info(f"Avatar '{avatar_id}' already in tracked.pt – overwriting.")

    existing[avatar_id] = result
    torch.save(existing, str(tracked_pt_path))

    logger.info(
        f"Successfully tracked '{avatar_id}' and saved to {tracked_pt_path}. "
        f"Use shape_id='{avatar_id}' in ARTalkAvatarSession."
    )
    return avatar_id
