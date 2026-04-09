# Analysis of Inference Metrics in ARTalk

This document details the metrics used to evaluate the ARTalk model and analyzes the feasibility of integrating them into the LiveKit plugin.

## 1. Evaluation Metrics in ARTalk

Based on the source code research and the ARTalk technical paper (2025), the metrics are divided into two main categories: motion accuracy and image quality.

### A. Lip Sync and Facial Motion
These metrics evaluate how well the generated movements match the input audio.

*   **LMD (Landmark Distance):** Measures the Euclidean distance between the generated facial landmarks (especially the mouth) and the targets. A low LMD indicates high geometric accuracy.
*   **F-LMD (Frame-wise LMD):** A variant of LMD that evaluates motion stability frame-by-frame to prevent sudden jumps (*jitter*).
*   **SyncNet Score (Confidence/Offset):** Uses a pre-trained model (SyncNet) to detect if audio and video are in phase. It provides a confidence score and a temporal offset in milliseconds.

### B. Visual Quality and Realism (GAGAvatar)
Since ARTalk uses **GAGAvatar** for Gaussian Splatting-based rendering, standard computer vision metrics are used:

*   **PSNR (Peak Signal-to-Noise Ratio):** Measures the image reconstruction quality. Higher values indicate less noise.
*   **SSIM (Structural Similarity Index):** Evaluates the similarity of structures, textures, and contrast between the generated frame and the original image.
*   **LPIPS (Learned Perceptual Image Patch Similarity):** Measures "perceptual" similarity using deep neural networks. It is considered closer to human judgment than PSNR/SSIM.
*   **FID (Fréchet Inception Distance):** Evaluates overall realism by comparing the distribution of generated images with a set of real images.

---

## 2. Integration Feasibility in this Project

### Technical Feasibility
Integration is **feasible** as the current environment uses PyTorch and has most of the base dependencies.

### Implementation Proposal

> [!IMPORTANT]  
> It is not recommended to run these metrics in real-time during a LiveKit session, as calculating LPIPS or SyncNet would add significant latency that would degrade the user experience.

| Use Case | Recommendation |
| :--- | :--- |
| **Quality Control (Offline)** | **Highly recommended.** Create a validation script that processes a test video and reports metrics before "approving" a new avatar. |
| **Real-Time Monitoring** | **Not recommended.** The computational cost is too high for the immediate benefit. |
| **Style Tuning (Fine-tuning)** | **Useful.** Can help decide which `style_id` works best for a specific voice. |

### Required Dependencies
To integrate these metrics, we should add:
*   `lpips` (for perceptual similarity).
*   `scikit-image` (for simplified SSIM/PSNR).
*   `syncnet-python` (optional, for rigorous lip-sync validation).

---

## 3. Suggested Next Steps

1.  **Implement an Evaluation Module:** Create a `livekit/plugins/artalk/evaluation.py` file to calculate these metrics asynchronously.
2.  **Avatar Benchmarking:** Use this module to generate a quality report every time a new image is processed with `prepare_artalk_avatar.py`.
