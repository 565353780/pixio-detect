import os
import torch

from typing import List, Optional, Union

from pixio_detect.Method.detect import (
    createTransform,
    preprocessImages,
    postprocessPixioFeatures,
    detectFile,
)

import pixio as _pixio  # noqa: E402


class Detector(object):
    def __init__(
        self,
        model_type: str,
        model_file_path: Union[str, None] = None,
        dtype="auto",
        device: str = "cpu",
        patch_size: int = 16,
    ) -> None:
        self.device = device
        self.patch_size = patch_size
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
            "vitb16": _pixio.pixio_vitb16,
            "vitl16": _pixio.pixio_vitl16,
            "vith16": _pixio.pixio_vith16,
            "vit1b16": _pixio.pixio_vit1b16,
            "vit5b16": _pixio.pixio_vit5b16,
        }

        if model_type not in model_configs:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {list(model_configs.keys())}"
            )

        factory = model_configs[model_type]
        self.model = factory(pretrained=None)

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.transform = createTransform()

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][PixioDetector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            self.is_valid = False
            return False

        model_state_dict = torch.load(
            model_file_path, map_location="cpu", weights_only=False
        )
        self.model.load_state_dict(model_state_dict, strict=True)

        print("[INFO][PixioDetector::loadModel]")
        print("\t model loaded from:", model_file_path)
        self.is_valid = True
        return True

    @torch.no_grad()
    def detect(
        self,
        image_tensor: torch.Tensor,
        block_ids: Optional[List[int]] = None,
    ) -> List[dict]:
        """
        Args:
            image_tensor: [B, H, W, 3], float32, range [0, 1]. No resize; H/W are
                padded (bottom/right with zeros) to multiples of ``patch_size``.
            block_ids: Optional block indices to return (same semantics as Pixio ``forward``).
        Returns:
            List of per-block dicts with keys such as ``patch_tokens_norm``,
            ``cls_tokens_norm``, ``patch_tokens``, ``cls_tokens`` (see PixioViT.forward).
            Tensors are cast back to the input image tensor's device/dtype.
        """
        image_tensor, input_dtype, input_device = preprocessImages(
            image_tensor,
            self.transform,
            self.device,
            self.dtype,
            patch_size=self.patch_size,
        )

        device_type = self.device if isinstance(self.device, str) else self.device.type
        device_type = device_type.split(":")[0]
        use_amp = device_type == "cuda" and self.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        if use_amp:
            with torch.autocast(device_type, dtype=self.dtype):
                features = self.model.forward(image_tensor, block_ids=block_ids)
        else:
            features = self.model.forward(image_tensor, block_ids=block_ids)

        assert isinstance(features, list)
        return postprocessPixioFeatures(features, input_device, input_dtype)

    @torch.no_grad()
    def detectFile(
        self,
        image_file_path: str,
        block_ids: Optional[List[int]] = None,
    ) -> Union[List[dict], None]:
        return detectFile(self, image_file_path, block_ids=block_ids)
