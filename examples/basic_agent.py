import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import cartesia, openai, silero
from livekit.plugins.artalk import ARTalkAvatarSession, ModelLoadStrategy
from livekit.plugins.artalk.image_preprocessor import preprocess_avatar_image

load_dotenv()  # Searches up the tree automatically — finds .env in the project root
logger = logging.getLogger("artalk-echo-agent")

# Suppress harmless internal torchaudio debug exception traces when probing for FFmpeg
logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)



class ARTalkVoiceAgent(Agent):
    """Simple voice agent with ARTalk 3D avatar."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful live-stream assistant speaking through a 3D avatar. "
                "Keep responses extremely short and conversational."
            )
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Entrypoint for the basic ARTalk agent example.
    Starts an ARTalk 3D visual avatar synchronized with the TTS voice output.
    """
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()

    # ── ARTalk Path ─────────────────────────────────────────────────────────
    artalk_clone_dir = os.environ.get("ARTALK_PATH", "../external_models/ARTalk")

    # ── Model Load Strategy ──────────────────────────────────────────────────
    # Controls whether models are loaded individually (from_scratch) or via
    # ARTalk's official Gradio engine class (example_models).
    # See .env.example.txt for all valid values and trade-offs.
    _strategy_raw = os.environ.get("ARTALK_MODEL_STRATEGY", "from_scratch")
    try:
        model_strategy = ModelLoadStrategy(_strategy_raw)
    except ValueError:
        logger.warning(
            f"Unknown ARTALK_MODEL_STRATEGY='{_strategy_raw}', "
            f"falling back to 'from_scratch'. Valid: {[s.value for s in ModelLoadStrategy]}"
        )
        model_strategy = ModelLoadStrategy.FROM_SCRATCH

    # ── Speaking Style ───────────────────────────────────────────────────────
    # See .env.example.txt for the full list of bundled style IDs.
    style_id = os.environ.get("ARTALK_STYLE_ID", "natural_0")

    # ── Shape ID / Avatar Image ──────────────────────────────────────────────
    # Shape ID: use "mesh" for the default neutral head, or a raw image path
    avatar_env_val = os.environ.get("AVATAR_IMAGE", "mesh")
    # If AVATAR_IMAGE is an absolute path to an image file, run the GAGAvatar preprocessing pipeline.
    # For the default "mesh" shape or any named shape_id, skip this step.
    shape_id = avatar_env_val
    if (
        avatar_env_val.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.isabs(avatar_env_val)
        and os.path.isfile(avatar_env_val)
    ):
        logger.info("Initializing pre-flight image tracking pipeline...")
        shape_id = preprocess_avatar_image(avatar_env_val, artalk_path_str=artalk_clone_dir)

    logger.info(
        f"ARTalk config: strategy={model_strategy.value}, "
        f"shape_id={shape_id!r}, style_id={style_id!r}"
    )

    # ── Avatar Session ───────────────────────────────────────────────────────
    avatar = ARTalkAvatarSession(
        artalk_path=artalk_clone_dir,
        avatar_participant_identity="artalk-visual-bot",
        shape_id=shape_id,
        style_id=style_id,
        model_strategy=model_strategy,
    )

    # ── TTS ──────────────────────────────────────────────────────────────────
    orig_tts = cartesia.TTS(voice="87748186-23bb-4158-a1eb-332911b0b708")

    # ── Agent Session (livekit-agents 1.x) ──────────────────────────────────
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=avatar.wrap_tts(orig_tts),  # Intercept TTS → ARTalk
    )

    # Start the avatar video track first so WebRTC negotiation is ready
    await avatar.start(room=ctx.room)

    logger.info("Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=ARTalkVoiceAgent(),
    )

    await asyncio.sleep(0.5)
    await session.generate_reply()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
