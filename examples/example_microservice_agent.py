import asyncio
import logging
import os

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

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import openai, silero # Dummy imports for illustration

# Import the official client plugin
from livekit.plugins.artalk import AvatarSession

logger = logging.getLogger("example_microservice_agent")
logging.basicConfig(level=logging.INFO)

class ExampleVoiceAgent(Agent):
    """
    Simple Voice Agent subclass for testing the architecture.
    Replace this with your actual agent logic (or VoicePipelineAgent in newer SDKs).
    """
    def __init__(self):
        super().__init__(
            instructions="You are a helpful 3D avatar assistant. Be extremely concise."
        )

async def entrypoint(ctx: JobContext):
    logger.info("Starting Example Agent with Microservice Architecture...")
    
    # 1. Provide or read LiveKit environment variables explicitly
    # You can pass these directly without relying on os.getenv if building a dynamic app
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([livekit_url, livekit_api_key, livekit_api_secret]):
        logger.error("LiveKit credentials must be set in your environment variables.")
        return

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 2. Define the Replica ID that was created previously 
    # Use 'examples/create_avatar_replica.py' to generate a valid replica ID beforehand
    # For testing without a previous replica, you can fall back to the default "mesh"
    replica_id = os.getenv("ARTALK_REPLICA_ID", "mesh")
    
    # URL to the ARTalk Backend (FastAPI Server) running on the GPU
    backend_url = os.getenv("ARTALK_SERVER_URL", "http://localhost:8000")
    
    try:
        # 3. Initialize your Voice Agent (The Brain) BEFORE starting the avatar
        # You need an agent session to pass to the avatar so it can intercept the audio
        session = AgentSession(
            vad=silero.VAD.load(),       # Configure your own plugins
            stt=openai.STT(),
            llm=openai.LLM(),
            tts=openai.TTS(),
        )
        
        # 4. Initialize the Avatar session
        logger.info(f"Joining ARTalk Avatar (Replica: {replica_id}) into the LiveKit room...")
        avatar = AvatarSession(replica_id=replica_id, api_url=backend_url)
        
        # start() requests the GPU worker to connect to this same WebRTC room,
        # and forwards `agent_session`'s local native TTS audio output to the GPU server via a Data Channel.
        # Here we demonstrate passing the LiveKit credentials explicitly as parameters.
        #
        # CRITICAL CONCEPT: We pass `session` (the AgentSession representing the Brain), 
        # NOT `ctx.agent` (the LocalParticipant representing the network peer).
        # The avatar plugin needs access to your actual Pipeline so it can intercept 
        # the raw TTS audio bytes BEFORE they are broadcasted to the room.
        await avatar.start(
            agent_session=session, # <-- Send the newly created AgentSession
            room=ctx.room,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret
        )
        
        logger.info("✅ Avatar is active and broadcasting video in the room!")
        
        # 5. Start your Voice Agent in the room
        # IMPORTANT: We use keyword arguments (room=, agent=) and pass an instance 
        # of our custom subclass (`ExampleVoiceAgent()`). Passing the empty base class 
        # `Agent()` or using positional arguments will cause strict type overload errors in Pylance.
        await session.start(room=ctx.room, agent=ExampleVoiceAgent())

        await asyncio.sleep(0.5)
        await session.generate_reply()

    except Exception as e:
        logger.error(f"Critical failure starting the avatar: {e}")


if __name__ == "__main__":
    # Ensure to configure your environment variables:
    # LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
