import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.io import read_image

from pixio_detect.Model import (
    pixio_vit1b16,
    pixio_vit5b16,
    pixio_vitb16,
    pixio_vith16,
    pixio_vitl16,
)


def _imagenet_normalize() -> T.Normalize:
    return T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _ceil_to_multiple(n: int, m: int) -> int:
    return (n + m - 1) // m * m


@torch.no_grad()
def _pad_bchw_to_patch_multiple(
    x: torch.Tensor,
    patch_size: int = 16,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    h2 = _ceil_to_multiple(h, patch_size)
    w2 = _ceil_to_multiple(w, patch_size)
    pad_h, pad_w = h2 - h, w2 - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)


@torch.no_grad()
def _preprocess(
    image_tensor: torch.Tensor,
    norm: T.Normalize,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.dtype, torch.device]:
    """[B,H,W,3] in [0,1] -> BCHW on device, padded to patch multiple, normalized."""
    input_dtype = image_tensor.dtype
    input_device = image_tensor.device
    x = image_tensor.permute(0, 3, 1, 2).to(device, dtype=torch.float32)
    x = _pad_bchw_to_patch_multiple(x, patch_size=patch_size, pad_value=0.0)
    x = x.to(dtype=dtype)
    x = norm(x)
    return x, input_dtype, input_device


@torch.no_grad()
def _postprocess_features(
    features: List[dict],
    input_device: torch.device,
    input_dtype: torch.dtype,
) -> List[dict]:
    return [
        {k: v.to(input_device, dtype=input_dtype) for k, v in block.items()}
        for block in features
    ]


class Detector(object):
    def __init__(
        self,
        model_type: str,
        model_file_path: Union[str, None] = None,
        dtype="auto",
        device: str = "cpu",
        patch_size: int = 16,
        is_offload_cpu: bool = False,
    ) -> None:
        self.device = device
        self.patch_size = patch_size
        # offload 模式：模型加载并常驻 CPU，``detect()`` 推理窗口内才搬到
        # ``self.device``；默认模式保持原始按 ``device`` 常驻的行为。
        # 与 ``dino_detect.Module.detector.Detector`` 的同名开关语义一致。
        self.is_offload_cpu = bool(is_offload_cpu)
        if dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif torch.cuda.is_available():
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        elif isinstance(dtype, str):
            td = getattr(torch, dtype, None)
            if td is None or not isinstance(td, torch.dtype):
                raise ValueError(
                    f"Unknown dtype string '{dtype}'. "
                    "Use a torch name like 'float32', 'float16', 'bfloat16'."
                )
            self.dtype = td
        else:
            self.dtype = dtype

        model_configs = {
            "vitb16": pixio_vitb16,
            "vitl16": pixio_vitl16,
            "vith16": pixio_vith16,
            "vit1b16": pixio_vit1b16,
            "vit5b16": pixio_vit5b16,
        }

        if model_type not in model_configs:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {list(model_configs.keys())}"
            )

        self.model = model_configs[model_type]()
        # offload 模式：模型常驻 CPU；默认模式：按 ``self.device`` 常驻。
        load_device = "cpu" if self.is_offload_cpu else self.device
        self.model = self.model.to(load_device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self._norm = _imagenet_normalize()

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            self.is_valid = False
            return False

        state = torch.load(model_file_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=True)

        print("[INFO][Detector::loadModel]")
        print("\t model loaded from:", model_file_path)
        self.is_valid = True
        return True

    def _moveModelToDevice(self) -> None:
        '''offload 模式下推理前把模型从 CPU 搬到 ``self.device``。'''
        if not self.is_offload_cpu:
            return
        if self.model is not None:
            self.model = self.model.to(self.device, dtype=self.dtype)

    def _offloadModelToCPU(self) -> None:
        '''offload 模式下推理结束后把模型卸载回 CPU 并清显存。'''
        if not self.is_offload_cpu:
            return
        if self.model is not None:
            self.model = self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(
        self,
        image_tensor: torch.Tensor,
        block_ids: Optional[List[int]] = None,
    ) -> List[dict]:
        '''
        float, [0 - 1], RGB order
        '''
        self._moveModelToDevice()
        try:
            x, input_dtype, input_device = _preprocess(
                image_tensor,
                self._norm,
                self.device,
                self.dtype,
                self.patch_size,
            )

            device_type = self.device if isinstance(self.device, str) else self.device.type
            device_type = device_type.split(":")[0]
            use_amp = device_type == "cuda" and self.dtype in (
                torch.float16,
                torch.bfloat16,
            )
            if use_amp:
                with torch.autocast(device_type, dtype=self.dtype):
                    features = self.model.forward(x, block_ids=block_ids)
            else:
                features = self.model.forward(x, block_ids=block_ids)

            assert isinstance(features, list)
            return _postprocess_features(features, input_device, input_dtype)
        finally:
            self._offloadModelToCPU()

    @torch.no_grad()
    def detectFile(
        self,
        image_file_path: str,
        block_ids: Optional[List[int]] = None,
    ) -> Union[List[dict], None]:
        if not os.path.exists(image_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t image file not exist!")
            print("\t image_file_path:", image_file_path)
            return None

        try:
            img = read_image(image_file_path)
        except Exception:
            print("[ERROR][Detector::detectFile]")
            print("\t failed to read image!")
            print("\t image_file_path:", image_file_path)
            return None

        if img.shape[0] != 3:
            print("[ERROR][Detector::detectFile]")
            print("\t expected 3-channel RGB image")
            return None

        t = (img.float() / 255.0).permute(1, 2, 0).unsqueeze(0)
        return self.detect(t, block_ids=block_ids)
