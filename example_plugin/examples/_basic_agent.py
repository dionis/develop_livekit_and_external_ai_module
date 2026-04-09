"""
Example: Basic Ditto Avatar Agent

This example demonstrates how to create a simple voice AI agent
with a Ditto talking head avatar.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai, silero, deepgram, elevenlabs, cartesia, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import the Ditto plugin
import sys
sys.path.insert(0, "../")
from livekit.plugins.ditto import DittoAvatarSession

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DittoVoiceAgent(Agent):
    """Simple voice agent with Ditto avatar."""
    
    def __init__(self):
        super().__init__(
            instructions="""
            You are a friendly AI assistant with a visual avatar.
            Keep your responses brief and conversational, ideally under 2 sentences.
            Be helpful and engaging.
            """
        )


async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the agent.
    
    This function is called when a new participant joins the room.
    """
    logger.info(f"Starting agent for room: {ctx.room.name}")
    await ctx.connect()
    
    try:
        # Create Ditto avatar session
        # NOTE: Update these paths to match your environment
        avatar_session = DittoAvatarSession(
            ditto_path="../",  # Path to Ditto installation
            source_image="../example/image.png",  # Avatar source image
            avatar_participant_identity="ditto-avatar",
            data_root="../checkpoints/ditto_pytorch",
            cfg_pkl="../checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        )
        
        # session = AgentSession(
        #     stt=deepgram.STT(model="nova-2"),
        #     llm=openai.LLM(model="gpt-4o"),
        #     tts=avatar_session.wrap_tts(elevenlabs.TTS(voice="Rachel")),
        #     vad=silero.VAD.load(),
        # )

        # Example configuration for Cartesia TTS and other models:
        # Note: Ensure CARTESIA_API_KEY is set in your .env file
        # cartesia_tts = cartesia.TTS(
        #     model="sonic-3", 
        #     voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        #     api_key=os.getenv("CARTESIA_API_KEY")
        # )
        # session = AgentSession(
        #     stt=deepgram.STT(model="nova-3", language="multi"),
        #     llm=openai.LLM(model="gpt-4o-mini"),
        #     tts=avatar_session.wrap_tts(cartesia_tts),
        #     vad=silero.VAD.load(),
        #     turn_detection=MultilingualModel(),
        # )

        # Example configuration for Gemini LLM:
        # Note: Ensure GOOGLE_API_KEY is set in your .env file
        cartesia_tts = cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", api_key=os.getenv("CARTESIA_API_KEY"))
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi", api_key=os.getenv("DEEPGRAM_API_KEY")),
            llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY") ),
            tts=avatar_session.wrap_tts(cartesia_tts),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        # Example configuration for OpenAI LLM:
        # Note: Ensure OPENAI_API_KEY is set in your .env file
        # session = AgentSession(
        #     stt=deepgram.STT(model="nova-3", language="multi", api_key=os.getenv("DEEPGRAM_API_KEY")),
        #     llm=openai.LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
        #     tts=avatar_session.wrap_tts(cartesia_tts),
        #     vad=silero.VAD.load(),
        #     turn_detection=MultilingualModel(),
        # )
        
        # Start avatar session
        logger.info("Starting avatar session...")
        await avatar_session.start(session, room=ctx.room)
        
        # Start agent session
        logger.info("Starting agent session...")
        await session.start(
            room=ctx.room,
            agent=DittoVoiceAgent()
        )
        
        # Generate initial greeting
        await session.generate_reply()
        
        logger.info("Agent started successfully")
        
    except Exception as e:
        logger.error(f"Error starting agent: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the agent
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # You can add more worker options here
        )
    )
