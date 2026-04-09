from livekit import agents
import os
import asyncio
import logging
from dotenv import load_dotenv
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import openai, silero, deepgram, elevenlabs, cartesia, google
from livekit.plugins.ditto import DittoAvatarSession, EmotionController, DittoEmotion
from livekit.plugins.turn_detector.multilingual import MultilingualModel
#HF import os
#  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


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

    # 1. Setup Emotion Controller
    emotion_ctrl = EmotionController(default_emotion=DittoEmotion.NEUTRAL)
    
    # 2. Initialize Avatar Session    
    avatar_session = DittoAvatarSession(
            ditto_path= os.environ.get('DITTO_PATH'),  # Path to Ditto installation
            source_image= os.environ.get('DITTO_SOURCE_IMAGE'),  # Avatar source image
            avatar_participant_identity="ditto-avatar",
            data_root=  os.environ.get('DITTO_DATA_ROOT'),
            cfg_pkl=  os.environ.get('DITTO_CFG_PKL'),
            emotion_controller=emotion_ctrl,
        )

    # 3. Initialize Agent Session
    cartesia_tts = cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", api_key=os.getenv("CARTESIA_API_KEY"))
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi", api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=openai.LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
        tts=avatar_session.wrap_tts(cartesia_tts),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
        
    #Change the emotion
    emotion_ctrl.set_emotion(DittoEmotion.DISGUST, intensity=0.85)
    
    # 4. Start
    # Start avatar session
    logger.info("Starting avatar session...")
    await avatar_session.start(session, room=ctx.room)
    
    logger.info("Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=DittoVoiceAgent()
    )

    #await session.start(room=ctx.room, agent=Agent(instructions="Be helpful."))
   
    await session.generate_reply()
    logger.info("Agent started successfully")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))