"""
LiveKit Agents Plugin for Ditto TalkingHead Avatar Integration

This plugin integrates the Ditto TalkingHead model as a virtual avatar provider
for LiveKit Agents, enabling real-time talking head video generation.
"""

from .avatar import DittoAvatarSession
from .emotion_controller import EmotionController, DittoEmotion
from .version import __version__

__all__ = [
    "DittoAvatarSession",
    "EmotionController",
    "DittoEmotion",
    "__version__",
]
