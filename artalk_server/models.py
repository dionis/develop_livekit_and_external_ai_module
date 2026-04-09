from pydantic import BaseModel
from typing import Dict, Any, Optional

class ReplicaResponse(BaseModel):
    replica_id: str
    quality: Optional[Dict[str, float]] = None

class ConversationRequest(BaseModel):
    replica_id: str
    background_scene: Optional[str] = None
    """
    background_scene: Specifies the background image to composite behind the avatar.
    Accepted formats (evaluated in this order):
      1. Default scene name — one of the built-in scenes in the `scenes/` directory,
         e.g. "office", "beach", "popular_street" (without the .png extension).
      2. Local file path   — an absolute or relative path to a PNG/JPG image on the
         server filesystem, e.g. "/home/user/custom_bg.png".
      3. HTTP/HTTPS URL    — a publicly accessible image URL that will be downloaded
         at conversation startup, e.g. "https://example.com/my_background.jpg".
    If the value does not match any of the above, the service returns HTTP 400
    with a descriptive English error message.
    """
    bg_threshold: Optional[int] = None
    """
    bg_threshold: Luminosity cutoff (1-255) for avatar/background separation.
      3-8   -> Keeps dark hair/shadows, may leave thin black halo.
      10-20 -> Recommended range. Default is 15.
      25-40 -> Cleaner edges but may erode dark hair or jacket.
      >40   -> Too aggressive, avatar parts start disappearing.
    """
    properties: Dict[str, str]

class ConversationResponse(BaseModel):
    conversation_id: str
    status: str
