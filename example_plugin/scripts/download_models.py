import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_turn_detector():
    try:
        logger.info("Downloading Turn Detector models...")
        from livekit.plugins.turn_detector.multilingual import MultilingualModel
        # This triggers the download of ONNX and languages.json if not present
        # We pass load_languages=True explicitly or just initialize it
        # MultilingualModel has a download_files method via its Plugin wrapper
        from livekit.plugins.turn_detector.base import EOUPlugin
        from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual
        
        runner = _EUORunnerMultilingual
        runner._download_files()
        logger.info("✓ Turn Detector models downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading Turn Detector models: {e}")
        # Don't exit here, attempt others

def download_silero_vad():
    try:
        logger.info("Downloading Silero VAD models...")
        from livekit.plugins import silero
        # silero.VAD.load() triggers download
        silero.VAD.load()
        logger.info("✓ Silero VAD models downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading Silero VAD models: {e}")

def main():
    logger.info("Starting standalone model download...")
    
    download_turn_detector()
    download_silero_vad()
    
    logger.info("Model download process completed.")

if __name__ == "__main__":
    main()
