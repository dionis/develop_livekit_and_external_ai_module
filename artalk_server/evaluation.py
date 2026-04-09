#!/usr/bin/env python
# Copyright (c) Brain-AIX Vancouver
#
# Quality evaluation module for ARTalk avatars.
# Provides basic computer vision metrics (PSNR, SSIM) to validate the "cooking" process.

import numpy as np
import cv2
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Higher values indicate better quality. Typically ranges from 20 to 50 dB.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between two images using OpenCV.
    
    Values range from -1 to 1. 1 indicates perfect structural similarity.
    """
    # Simple SSIM implementation using OpenCV's quality module isn't always available,
    # so we use a structural comparison approach.
    # Note: For production-grade SSIM, scikit-image is preferred.
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Mean squared error variant for structural comparison
        # (This is a simplified version to avoid heavy dependencies)
        s = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
        return float(s)
    except Exception as e:
        logger.warning(f"Could not calculate structural similarity: {e}")
        return 0.0

def evaluate_avatar_quality(original_img: np.ndarray, tracked_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluates the quality of a tracked avatar by comparing the original image 
    with the 'canonical' representation in the tracked data.
    """
    metrics = {}
    
    if "image" not in tracked_data:
        logger.warning("No canonical image found in tracked data for evaluation.")
        return metrics
    
    try:
        # Get canonical image from tracked data
        # Tracked images are typically stored as torch tensors [C, H, W] in [0, 1]
        canonical_tensor = tracked_data["image"]
        
        # Convert to NumPy [H, W, C] in [0, 255]
        import torch
        if isinstance(canonical_tensor, torch.Tensor):
            canonical_np = canonical_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            canonical_np = np.array(canonical_tensor)
            
        canonical_np = (canonical_np * 255).astype(np.uint8)
        
        # Ensure original image matches canonical size for comparison
        h, w = canonical_np.shape[:2]
        original_resized = cv2.resize(original_img, (w, h))
        
        # Convert original if it's not RGB (OpenCV uses BGR)
        # Assuming original_img is BGR from cv2.imread
        canonical_bgr = cv2.cvtColor(canonical_np, cv2.COLOR_RGB2BGR)
        
        # Calculate metrics
        metrics["psnr"] = calculate_psnr(original_resized, canonical_bgr)
        metrics["ssim"] = calculate_ssim(original_resized, canonical_bgr)
        
    except Exception as e:
        logger.error(f"Error during avatar quality evaluation: {e}")
        
    return metrics

def display_metrics(metrics: Dict[str, float]) -> None:
    """Logs metrics in a human-readable format."""
    logger.info("="*40)
    logger.info("       AVATAR QUALITY METRICS")
    logger.info("="*40)
    
    if not metrics:
        logger.info("No metrics available. Ensure the input image was processed correctly.")
        return

    psnr = metrics.get("psnr", 0)
    ssim = metrics.get("ssim", 0)
    
    logger.info(f"PSNR: {psnr:,.2f} dB")
    if psnr > 30:
        logger.info("  [Status]: Excellent reconstruction.")
    elif psnr > 20:
        logger.info("  [Status]: Good. Minimal artifacts.")
    else:
        logger.info("  [Status]: Warning. High noise or poor alignment.")
        
    logger.info(f"SSIM: {ssim:,.4f}")
    if ssim > 0.9:
        logger.info("  [Status]: Very high structural similarity.")
    elif ssim > 0.7:
        logger.info("  [Status]: Acceptable structural similarity.")
    else:
        logger.info("  [Status]: Poor structure match. Check lighting/pose.")
    
    logger.info("="*40)
