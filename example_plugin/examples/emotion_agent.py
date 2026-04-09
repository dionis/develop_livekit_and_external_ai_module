"""
Example: Emotional Ditto Avatar Agent

This example demonstrates how to create a voice AI agent with a Ditto talking head
that can express different emotions based on the conversation context or commands.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import llm
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai, silero, deepgram, elevenlabs, cartesia, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import the Ditto plugin and emotion components
import sys
sys.path.insert(0, "../")
from livekit.plugins.ditto import DittoAvatarSession, EmotionController, DittoEmotion

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt to instruct the LLM to use emotion tags if desired, 
# although our EmotionController also supports keyword inference.
SYSTEM_PROMPT = """
You are a helpful and empathetic AI assistant with a visual avatar.
Your responses should be conversational and brief (under 2 sentences).

You can express emotions. When you feel a specific emotion is appropriate, 
you can use keywords that will trigger my facial expressions.
- For happiness: use words like "happy", "great", "wonderful".
- For sadness: use words like "sad", "sorry", "unfortunate".
- For surprise: use words like "wow", "amazing".
- For anger: use words like "angry", "frustrated".

You can also be explicit by starting your sentence with a tag like [HAPPY], [SAD], etc.
"""

class EmotionalDittoAgent(Agent):
    """Voice agent with emotional Ditto avatar."""
    
    def __init__(self):
        super().__init__(
            instructions=SYSTEM_PROMPT
        )

async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the agent."""
    logger.info(f"Starting emotional agent for room: {ctx.room.name}")
    await ctx.connect()
    
    try:
        # 1. Create Emotion Controller
        emotion_ctrl = EmotionController(
            default_emotion=DittoEmotion.NEUTRAL
        )
        
        # 2. Create Ditto avatar session with emotion controller
        # NOTE: Update these paths to match your environment
        avatar_session = DittoAvatarSession(
            ditto_path="../",  # Path to Ditto installation
            source_image="../example/image.png",  # Avatar source image
            avatar_participant_identity="ditto-avatar",
            data_root="../checkpoints/ditto_pytorch",
            cfg_pkl="../checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
            emotion_controller=emotion_ctrl, # Pass the controller
            video_width=512,
            video_height=512,
        )
        
        # 3. Configure Agent Session
        # Replace APIs with your preferred providers
        cartesia_tts = cartesia.TTS(
            model="sonic-3", 
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            api_key=os.getenv("CARTESIA_API_KEY")
        )
        
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi", api_key=os.getenv("DEEPGRAM_API_KEY")),
            llm=google.LLM(
                model="gemini-2.0-flash-exp", 
                api_key=os.getenv("GOOGLE_API_KEY")
            ),
            tts=avatar_session.wrap_tts(cartesia_tts), # Wrap TTS to capture audio/text for Ditto
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        # 4. Start avatar session
        logger.info("Starting avatar session...")
        await avatar_session.start(session, room=ctx.room)
        
        # 5. Start agent session
        logger.info("Starting agent session...")
        await session.start(
            room=ctx.room,
            agent=EmotionalDittoAgent()
        )
        
        # 6. Generate initial greeting
        await session.generate_reply()
        
        logger.info("Agent started successfully. Try sending identifying emotion commands via DataChannel!")
        logger.info("Example JSON: { \"type\": \"emotion_control\", \"emotion\": \"happy\" }")
        
    except Exception as e:
        logger.error(f"Error starting agent: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
