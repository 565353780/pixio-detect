import os
import cv2
import torch
import numpy as np

from tqdm import trange

from pixio_detect.Module.detector import Detector


def demo():
    home = os.environ["HOME"]

    model_type = "vith16"
    model_file_path = f"{home}/path/to/your_pixio_vith16_checkpoint.pth"
    dtype = "auto"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_file_path = f"{home}/tmp/test_pixio.png"
    if not os.path.exists(image_file_path):
        # Non-square size to show aspect ratio is preserved (only padded to /16).
        h, w = 480, 640
        noise_img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
        cv2.imwrite(image_file_path, noise_img)

    if not os.path.exists(model_file_path):
        print("[WARN] Checkpoint not found at:", model_file_path)
        print("\tSet model_file_path to a valid Pixio .pth or load weights later via loadModel().")
        model_file_path = None

    detector = Detector(model_type, model_file_path, dtype, device)

    for _ in trange(10):
        pixio_features = detector.detect(
            torch.rand([2, 480, 640, 3], dtype=torch.float32, device="cpu")
        )

    print("pixio_features: num blocks =", len(pixio_features))
    if pixio_features:
        first = pixio_features[0]
        for k, v in first.items():
            print(f"  {k}: shape {tuple(v.shape)}, dtype {v.dtype}")

    for _ in trange(10):
        pixio_features = detector.detectFile(image_file_path)

    print("from file, num blocks =", len(pixio_features) if pixio_features else None)
    return True


if __name__ == "__main__":
    demo()
