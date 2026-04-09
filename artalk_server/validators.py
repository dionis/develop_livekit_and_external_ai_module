#!/usr/bin/env python
# Copyright (c) Brain-AIX Vancouver
#
# Avatar image validators for the ARTalk microservice.
# These checks run before the expensive GAGAvatar tracking pipeline
# to catch unusable images early and return clear error messages.

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum acceptable resolution (width or height, in pixels)
MIN_DIMENSION_PX = 256

# Laplacian variance threshold — images below this value are considered blurry.
# A natural portrait in good lighting typically scores 100–500+.
# 50 is a conservative lower bound to catch heavily blurred or out-of-focus images.
MIN_SHARPNESS_SCORE = 50.0


def validate_image_quality(image_path: str) -> None:
    """
    Validate that the image meets minimum quality requirements for ARTalk processing.

    Checks performed:
      1. Minimum resolution: both width and height must be >= MIN_DIMENSION_PX (256 px).
      2. Minimum sharpness: Laplacian variance must be >= MIN_SHARPNESS_SCORE (50.0).
         This catches heavily blurred, out-of-focus, or near-solid-color images.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file on disk.

    Raises
    ------
    ValueError
        If the image fails any quality check. The message is in English and
        is safe to forward directly to the API caller.
    """
    import cv2
    import numpy as np

    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        msg = (
            f"Failed to read image '{path.name}'. "
            "The file may be corrupted or in an unsupported format."
        )
        logger.error(f"[validate_image_quality] {msg}")
        raise ValueError(msg)

    h, w = img.shape[:2]

    # --- Resolution check ---
    if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
        msg = (
            f"Image resolution {w}x{h} is too low. "
            f"A minimum of {MIN_DIMENSION_PX}x{MIN_DIMENSION_PX} pixels is required "
            "for reliable avatar generation. Please provide a higher-resolution portrait."
        )
        logger.error(f"[validate_image_quality] {msg} | file={path.name}")
        raise ValueError(msg)

    # --- Sharpness check (Laplacian variance) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < MIN_SHARPNESS_SCORE:
        msg = (
            f"Image '{path.name}' appears to be too blurry or has insufficient detail "
            f"(sharpness score: {sharpness:.1f}, minimum required: {MIN_SHARPNESS_SCORE}). "
            "Please provide a clear, well-lit, in-focus portrait photograph."
        )
        logger.error(f"[validate_image_quality] {msg}")
        raise ValueError(msg)

    logger.info(
        f"[validate_image_quality] PASSED | file={path.name} | "
        f"resolution={w}x{h} | sharpness={sharpness:.1f}"
    )


def validate_face_detected(image_path: str) -> None:
    """
    Validate that the image contains at least one detectable human face.

    Uses OpenCV's built-in Haar cascade classifier (no extra model download required).
    If MediaPipe is installed, it is preferred as a more robust detector.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file on disk (must already exist).

    Raises
    ------
    ValueError
        If no face is detected. The message is in English and is safe to forward
        directly to the API caller.
    """
    import cv2

    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        # Already caught by validate_image_quality; guard here for standalone use.
        raise ValueError(f"Cannot read image '{path.name}' for face detection.")

    # --- Try MediaPipe first (more accurate, handles partial / angled faces) ---
    mp_face_detected = _detect_face_mediapipe(img, path.name)
    if mp_face_detected is True:
        logger.info(f"[validate_face_detected] Face confirmed via MediaPipe | file={path.name}")
        return
    if mp_face_detected is False:
        # MediaPipe loaded successfully but found no face — definitive failure.
        msg = (
            f"No face detected in '{path.name}' (MediaPipe). "
            "Please provide a clear frontal portrait photograph with a visible face."
        )
        logger.error(f"[validate_face_detected] {msg}")
        raise ValueError(msg)

    # --- Fallback: OpenCV Haar cascade ---
    logger.info("[validate_face_detected] MediaPipe unavailable, falling back to Haar cascade.")
    haar_face_detected = _detect_face_haar(img, path.name)
    if not haar_face_detected:
        msg = (
            f"No face detected in '{path.name}'. "
            "Please provide a clear frontal portrait photograph with a fully visible face "
            "and good lighting. Avoid heavy occlusions (masks, sunglasses, hats)."
        )
        logger.error(f"[validate_face_detected] {msg}")
        raise ValueError(msg)

    logger.info(f"[validate_face_detected] Face confirmed via Haar cascade | file={path.name}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_face_mediapipe(img, filename: str):
    """
    Try to detect a face using MediaPipe FaceDetection.

    Returns:
        True   — face detected
        False  — MediaPipe loaded, but no face found
        None   — MediaPipe not available (caller should fall back)
    """
    try:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            import mediapipe.python.solutions as mp_solutions
            mp.solutions = mp_solutions

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,       # 1 = full-range model (better for portraits)
            min_detection_confidence=0.5,
        ) as detector:
            import cv2
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(img_rgb)
            if results.detections:
                logger.debug(
                    f"[validate_face_detected] MediaPipe found "
                    f"{len(results.detections)} face(s) in '{filename}'"
                )
                return True
            return False

    except ImportError:
        return None  # MediaPipe not installed
    except Exception as e:
        logger.warning(f"[validate_face_detected] MediaPipe error: {e} – falling back to Haar.")
        return None


def _detect_face_haar(img, filename: str) -> bool:
    """
    Detect a face using OpenCV's frontal face Haar cascade.
    Tries multiple scale factors so that both small and large faces are caught.
    """
    import cv2

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        logger.warning(
            "[validate_face_detected] Haar cascade file not found. "
            "Face validation skipped — consider installing a full OpenCV build."
        )
        return True  # Fail-open: don't block the user if the model file is missing

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improve contrast for low-light images

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) > 0:
        logger.debug(f"[validate_face_detected] Haar found {len(faces)} face(s) in '{filename}'")
        return True

    # Retry with a looser scale to catch very large / close-up faces
    faces_retry = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
    )
    return len(faces_retry) > 0
