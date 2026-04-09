import logging
from enum import IntEnum
from typing import Union, Optional
import asyncio

logger = logging.getLogger(__name__)

class DittoEmotion(IntEnum):
    """
    Emotions supported by Ditto TalkingHead.
    These map directly to the indices used by Ditto's condition_handler.
    """
    ANGRY    = 0
    DISGUST  = 1
    FEAR     = 2
    HAPPY    = 3
    NEUTRAL  = 4
    SAD      = 5
    SURPRISE = 6
    CONTEMPT = 7

# Mapping of emotions to keywords for simple inference
EMOTION_KEYWORDS = {
    DittoEmotion.HAPPY:    ["happy", "joy", "excited", "great", "wonderful", "alegre", "feliz", "contento"],
    DittoEmotion.SAD:      ["sad", "sorry", "unfortunate", "triste", "lamentable", "pena"],
    DittoEmotion.ANGRY:    ["angry", "frustrated", "annoyed", "enojado", "frustrado", "molesto"],
    DittoEmotion.SURPRISE: ["wow", "amazing", "incredible", "surprised", "sorprendente", "increíble"],
    DittoEmotion.FEAR:     ["worried", "concerned", "afraid", "preocupado", "miedo"],
    DittoEmotion.DISGUST:  ["disgusting", "terrible", "awful", "horrible", "asco"],
    DittoEmotion.CONTEMPT: ["whatever", "indifferent", "meh", "indiferente"],
    DittoEmotion.NEUTRAL:  ["neutral", "okay", "fine", "normal", "bien"],
}

class EmotionController:
    """
    Controller for Ditto avatar emotional states.
    
    Manages the current emotion and provides methods to change it
    from multiple sources (LLM text, DataChannel, API).
    """
    
    def __init__(
        self,
        default_emotion: DittoEmotion = DittoEmotion.NEUTRAL,
    ):
        self._current_emotion = default_emotion
        self._ditto_sdk = None
        
    def attach_sdk(self, ditto_sdk) -> None:
        """Attach to the DittoSDKWrapper instance."""
        self._ditto_sdk = ditto_sdk
        # Sync the initial state now that SDK is attached
        if self._ditto_sdk.is_loaded:
            self._ditto_sdk.update_emotion(self._current_emotion)
    
    def set_emotion(
        self, 
        emotion: Union[DittoEmotion, int, str],
        intensity: float = 1.0,
        blend_with_neutral: bool = False,
    ) -> None:
        """
        Change the avatar's emotion.
        
        Args:
            emotion: Emotion as enum, int (0-7) or string ("happy")
            intensity: Intensity 0.0-1.0 (currently used for blending logic)
            blend_with_neutral: If True, blends with Neutral based on intensity
        """
        try:
            emo_int = self._resolve_emotion(emotion)
            
            # For now, we support direct emotion switching.
            # Blending logic can be expanded here if needed by passing a list to the SDK.
            # Simple blending implementation:
            if blend_with_neutral and intensity < 1.0:
                # This would need support in SDK for weighted lists
                 # For now we stick to the primary emotion but log intent
                 pass

            self._current_emotion = emo_int
            
            # Apply to SDK immediately if loaded
            if self._ditto_sdk and self._ditto_sdk.is_loaded:
                self._ditto_sdk.update_emotion(emo_int, intensity=intensity)
            
            logger.info(f"Emotion changed to: {DittoEmotion(emo_int).name} (intensity={intensity})")
            
        except ValueError as e:
            logger.error(f"Failed to set emotion: {e}")
    
    def infer_from_text(self, text: str) -> DittoEmotion:
        """
        Infer emotion from text using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Inferred emotion (defaults to NEUTRAL if no match)
        """
        text_lower = text.lower()
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return DittoEmotion.NEUTRAL
    
    def set_from_text(self, text: str) -> DittoEmotion:
        """Infer and set emotion from text."""
        emotion = self.infer_from_text(text)
        if emotion != self._current_emotion:
            self.set_emotion(emotion)
        return emotion
    
    @property
    def current_emotion(self) -> DittoEmotion:
        return DittoEmotion(self._current_emotion)
    
    def _resolve_emotion(self, emotion: Union[DittoEmotion, int, str]) -> int:
        """Resolve various input types to a valid emotion index."""
        if isinstance(emotion, int):
            if 0 <= emotion <= 7:
                return emotion
            raise ValueError(f"Emotion index out of range (0-7): {emotion}")
            
        if isinstance(emotion, str):
            emotion_lower = emotion.lower()
            for emo in DittoEmotion:
                if emo.name.lower() == emotion_lower:
                    return int(emo)
            # Try matching against keywords if direct name match fails
            for emo, keywords in EMOTION_KEYWORDS.items():
                 if emotion_lower in keywords:
                     return int(emo)
            raise ValueError(f"Unknown emotion string: {emotion}")
            
        if isinstance(emotion, DittoEmotion):
            return int(emotion)
            
        raise ValueError(f"Unsupported emotion type: {type(emotion)}")
