import os
import cv2
import torch
import numpy as np
from typing import List, Optional, Union

import torch.nn.functional as F
from torchvision.transforms import v2


def createTransform() -> v2.Compose:
    return v2.Compose([
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def _ceil_to_multiple(n: int, m: int) -> int:
    return (n + m - 1) // m * m


@torch.no_grad()
def pad_bchw_to_patch_multiple(
    x: torch.Tensor,
    patch_size: int = 16,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad (B, C, H, W) on bottom/right so H, W are divisible by patch_size."""
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")

    _, _, h, w = x.shape
    h2 = _ceil_to_multiple(h, patch_size)
    w2 = _ceil_to_multiple(w, patch_size)
    pad_h = h2 - h
    pad_w = w2 - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)


@torch.no_grad()
def preprocessImages(
    image_tensor: torch.Tensor,
    transform: v2.Compose,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    patch_size: int = 16,
) -> tuple:
    """Preprocess image tensor for Pixio (no resize; pad to patch multiple only).

    Args:
        image_tensor: [B, H, W, 3], float32, range [0, 1]
        transform: normalization transform (ImageNet mean/std)
        device: target device
        dtype: compute dtype (e.g. bfloat16 on GPU)
        patch_size: ViT patch size (default 16)

    Returns:
        (processed_tensor, input_dtype, input_device)
    """
    input_dtype = image_tensor.dtype
    input_device = image_tensor.device

    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor.to(device, dtype=torch.float32)
    image_tensor = pad_bchw_to_patch_multiple(image_tensor, patch_size=patch_size, pad_value=0.0)
    image_tensor = image_tensor.to(dtype=dtype)
    image_tensor = transform(image_tensor)

    return image_tensor, input_dtype, input_device


@torch.no_grad()
def postprocessPixioFeatures(
    features: List[dict],
    input_device: torch.device,
    input_dtype: torch.dtype,
) -> List[dict]:
    out: List[dict] = []
    for block in features:
        out.append(
            {k: v.to(input_device, dtype=input_dtype) for k, v in block.items()}
        )
    return out


@torch.no_grad()
def detectFile(
    detector,
    image_file_path: str,
    block_ids: Optional[List[int]] = None,
) -> Union[List[dict], None]:
    if not os.path.exists(image_file_path):
        print("[ERROR][detectFile]")
        print("\t image file not exist!")
        print("\t image_file_path:", image_file_path)
        return None

    image_bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print("[ERROR][detectFile]")
        print("\t failed to read image!")
        print("\t image_file_path:", image_file_path)
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).unsqueeze(0)

    return detector.detect(image_tensor, block_ids=block_ids)
