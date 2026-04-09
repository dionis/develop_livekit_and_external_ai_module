import torch
import os

path = r"e:\PROJECTS\PROJECT_BRAIN-AIX_VANCOUVER\SOURCE\livekit-plugins-artalk\external_models\ARTalk\assets\style_motion\natural_0.pt"
if os.path.exists(path):
    data = torch.load(path, map_location='cpu')
    print(f"Data shape: {data.shape}")
    # data is [Frames, 106]
    # Check variance of each index to find moving parts
    var = torch.var(data, dim=0)
    top_v, top_i = torch.topk(var, 20)
    print("Top 20 moving indices (possible blinks/head):")
    for i, v in zip(top_i, top_v):
        print(f"Index {i.item()}: Var {v.item():.6f}")
else:
    print(f"File not found: {path}")
