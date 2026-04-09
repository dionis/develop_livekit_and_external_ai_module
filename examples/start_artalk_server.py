import os
import sys
import uvicorn
import logging

# Add the root directory to the path so uvicorn can find artalk_server
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("artalk_server_launcher")
    
    # Background configuration from environment variables
    host = os.getenv("ARTALK_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("ARTALK_SERVER_PORT", "8000"))
    
    # Other critical variables that your backend assumes will exist:
    # LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    
    print("\n" + "="*60)
    print(f"🚀 Starting the ARTalk Microservice Server on {host}:{port}")
    print("This server must be executed on the node hosting the GPU (Cuda).")
    print("="*60 + "\n")
    
    # Programmatically bind Uvicorn to our FastAPI app
    uvicorn.run(
        "artalk_server.main:app", 
        host=host, 
        port=port, 
        reload=False  # Use a uvicorn proxy in production if multithreading is required
    )
