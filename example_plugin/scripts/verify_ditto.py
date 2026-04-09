import sys
import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyDitto")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from sota_benchmarker.models.ditto_wrapper import DittoWrapper
except ImportError as e:
    logger.error(f"Failed to import DittoWrapper: {e}")
    sys.exit(1)

def main():
    logger.info("Initializing DittoWrapper...")
    wrapper = DittoWrapper()
    
    logger.info("Checking generation mode...")
    mode = wrapper.get_generation_mode()
    logger.info(f"Generation Mode: {mode}")

    logger.info("Attempting to load model (dry run)...")
    try:
        # Note: This might fail if the external_models/ditto-talkinghead repo is not fully set up
        # We catch the error to allow verification of the wrapper structure
        wrapper.load(precision="FP16", device="cuda", denoising_steps=10)
        logger.info("Load successful!")
        
        if wrapper.is_placeholder:
            logger.warning("Wrapper loaded as PLACEHOLDER. Check model paths.")
        else:
            logger.info("Wrapper loaded REAL model.")
            
        if hasattr(wrapper, 'last_metrics'):
            logger.info("✓ Wrapper has 'last_metrics' attribute for telemetry.")
        else:
            logger.error("❌ Wrapper MISSING 'last_metrics' attribute!")
            
    except Exception as e:
        logger.error(f"Load failed with error: {e}")
        logger.info("This is expected if the physical model checkpoints are missing.")
        
    logger.info("Verification script finished.")

if __name__ == "__main__":
    main()
