import torch
from PIL import Image


def device_dtype():
    """cuda + bf16 on a GPU, cpu + fp32 otherwise."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        import io, urllib.request
        with urllib.request.urlopen(path_or_url) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(path_or_url).convert("RGB")
