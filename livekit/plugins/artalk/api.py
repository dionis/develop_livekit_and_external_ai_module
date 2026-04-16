import os
import aiohttp
from typing import Optional, Dict, Any
from .models import CreateReplicaResponse, AvatarQualityMetrics
from livekit.agents.utils import http_context

class ARTalkAPI:
    def __init__(self, api_url: str = "http://localhost:8000", http_session: Optional[Any] = None):
        """
        Lightweight HTTP client to the ARTalk Backend Service.
        """
        self.api_url = api_url.rstrip("/")
        self._http_session = http_session
        
    def _ensure_session(self):
        if self._http_session is None:
            self._http_session = http_context.http_session()
        return self._http_session

    async def create_replica(self, image_url: str, return_metrics: bool = True) -> CreateReplicaResponse:
        """
        Calls the backend to preprocess the image and create an avatar ID.
        """
        session = self._ensure_session()
        url = f"{self.api_url}/v1/avatar/create"
        
        is_url = image_url.startswith("http://") or image_url.startswith("https://")
        params = {"return_metrics": str(return_metrics).lower()}
        
        # Cooking an avatar can take several minutes, disable the default 5-minute timeout
        timeout = aiohttp.ClientTimeout(total=None)
        
        if is_url:
            params["image_url"] = image_url
            async with session.post(url, params=params, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
        else:
            data_form = aiohttp.FormData()
            data_form.add_field('file', open(image_url, 'rb'), filename=os.path.basename(image_url))
            async with session.post(url, params=params, data=data_form, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
                
        metrics = data.get("quality", {})
        return CreateReplicaResponse(
            replica_id=data["replica_id"],
            metrics=AvatarQualityMetrics(
                psnr=metrics.get("psnr"), 
                ssim=metrics.get("ssim")
            )
        )
            
    async def create_conversation(
        self, 
        replica_id: str, 
        properties: Dict[str, Any],
        background_scene: Optional[str] = None,
        bg_threshold: Optional[int] = None,
    ) -> str:
        """
        Commands the backend to join the LiveKit room.
        """
        session = self._ensure_session()
        url = f"{self.api_url}/v1/conversation"
        payload = {
            "replica_id": replica_id, 
            "properties": properties,
            "background_scene": background_scene,
            "bg_threshold": bg_threshold,
        }
        
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["conversation_id"]
