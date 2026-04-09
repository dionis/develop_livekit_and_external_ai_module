from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class AvatarQualityMetrics:
    psnr: Optional[float] = None
    ssim: Optional[float] = None

@dataclass
class CreateReplicaResponse:
    replica_id: str
    metrics: AvatarQualityMetrics
