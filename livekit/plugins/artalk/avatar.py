from __future__ import annotations

import os
import logging

from livekit import api, rtc
from livekit.agents import AgentSession, NOT_GIVEN, NotGivenOr, get_job_context
from livekit.agents.voice.avatar import DataStreamAudioOutput

from .api import ARTalkAPI

logger = logging.getLogger("livekit.plugins.artalk.avatar")
SAMPLE_RATE = 16000

_AVATAR_AGENT_IDENTITY = "artalk-avatar-server"
_AVATAR_AGENT_NAME = "artalk-avatar-server"

class AvatarSession:
    """An ARTalk avatar session powered by the external ARTalk Microservice"""

    def __init__(self, replica_id: str, api_url: str = "http://localhost:8000"):
        self.replica_id = replica_id
        self._api = ARTalkAPI(api_url=api_url)
        self.conversation_id: str | None = None
        
    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
    ) -> None:
        """
        Connects the ARTalk Backend Server to the current room and routes audio to it.
        Credentials can be explicitly passed; otherwise, they are read from environment variables.
        """
        livekit_url = livekit_url or os.getenv("LIVEKIT_URL")
        livekit_api_key = livekit_api_key or os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = livekit_api_secret or os.getenv("LIVEKIT_API_SECRET")
        
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise ValueError("LiveKit credentials must be provided or set in env vars.")
        
        local_participant_identity = room.local_participant.identity

        
        # Create a token for the GPU Server so it can join and publish video
        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(_AVATAR_AGENT_IDENTITY)
            .with_name(_AVATAR_AGENT_NAME)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # Custom attribute to hint the frontend that this video belongs to the current agent
            .with_attributes({"lk.publish_on_behalf": local_participant_identity})
            .to_jwt()
        )
        
        logger.info(f"Starting AvatarSession with Backend at {self._api.api_url}")
        
        # Ask backend to start the WebRTC worker
        self.conversation_id = await self._api.create_conversation(
            replica_id=self.replica_id,
            properties={
                "livekit_ws_url": livekit_url, 
                "livekit_room_token": livekit_token,
                # Additional config like artalk_path could be passed here if dynamic
            }
        )
        
        # The key magic: Override the local agent's audio output
        # Instead of sending TTS to normal audio track, it sends bytes
        # via DataChannels directly to the virtual avatar participant
        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=_AVATAR_AGENT_IDENTITY,
            sample_rate=SAMPLE_RATE,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO
        )
        
        logger.info("Avatar Session configured. Waiting for Backend Video Track...")
