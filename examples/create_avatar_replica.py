import asyncio
import aiohttp
import logging
import os
import sys

from dotenv import load_dotenv

# .env lookup strategy (cascade — first match wins):
#   1. examples/.env  — local dev/test credentials (create this file to override)
#   2. project root .env — production / shared credentials
#
# This way you can put testing values in examples/.env without touching the
# root .env, and both `python examples/example_microservice_agent.py start`
# and `python -m examples.example_microservice_agent start` work correctly.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_candidates = [
    os.path.join(_script_dir, ".env"),          # examples/.env  (highest priority)
    os.path.join(_script_dir, "..", ".env"),    # project root .env (fallback)
]
for _env_path in _env_candidates:
    if os.path.isfile(_env_path):
        load_dotenv(dotenv_path=_env_path, override=True)
        break
# Add root directory to path to allow importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from livekit.plugins.artalk.api import ARTalkAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_avatar_example")

async def main():
    """
    Example script to request the creation (cooking) of an avatar image 
    using the ARTalk microservice abstracted by the API.
    Ensure the FastAPI server is running at the given URL.
    """
    
    # Read from environment variables or use defaults
    api_url = os.getenv("ARTALK_SERVER_URL", "http://localhost:8000")
    image_url = os.getenv("AVATAR_IMAGE_URL", "https://raw.githubusercontent.com/livekit/agents/main/examples/assets/default_avatar.png")
    
    async with aiohttp.ClientSession() as session:
        api = ARTalkAPI(api_url=api_url, http_session=session)
        
        logger.info(f"Requesting avatar creation/processing on {api_url}")
        logger.info(f"Source image: {image_url}")
        
        try:
            # This calls the backend via HTTP. The GPU downloads the image, 
            # processes it via FLAME, and saves the .pt tracking file
            replica = await api.create_replica(image_url, return_metrics=True)
            
            print("\n" + "="*50)
            print("✅ Avatar successfully processed and created!")
            print(f"🆔 Avatar ID (replica_id): {replica.replica_id}")
            if replica.metrics:
                print(f"📊 Quality (PSNR): {replica.metrics.psnr}")
                print(f"📊 Quality (SSIM): {replica.metrics.ssim}")
            print("="*50 + "\n")
            
            print(f"You can now use ID '{replica.replica_id}' in your AvatarSession(replica_id='...').")
            
        except Exception as e:
            logger.error(f"Error connecting to the ARTalk server: {e}")
            logger.error("Make sure to start the server first (e.g. `python examples/start_artalk_server.py`)")

if __name__ == "__main__":
    asyncio.run(main())
