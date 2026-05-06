import io
import os

import cairosvg
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Redirect HuggingFace downloads to a project-local cache so the global
# ~/.cache/huggingface directory is not required.
_DEFAULT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", ".model_cache"
)
os.environ.setdefault("HF_HOME", os.path.abspath(_DEFAULT_CACHE))


def _open_image(path: str) -> Image.Image:
    """Open an image file as a PIL RGB image, rasterizing SVGs via cairosvg.

    RGBA images (including SVGs with transparent backgrounds) are composited
    onto a white background so transparent pixels become white, not black.
    """
    if path.lower().endswith(".svg"):
        png_bytes = cairosvg.svg2png(url=path)
        img = Image.open(io.BytesIO(png_bytes))
    else:
        img = Image.open(path)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        """Returns (N, D) L2-normalized image embeddings.

        Raises FileNotFoundError if any path does not exist.
        """
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")

        images = [_open_image(p) for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
            features = self._model.visual_projection(vision_out.pooler_output)
        emb = features.numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    def encode_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Returns (N, D) L2-normalized text embeddings, processed in batches."""
        for i, t in enumerate(texts):
            if t is None or (not isinstance(t, str) and not isinstance(t, np.str_)):
                raise ValueError(f"encode_texts: expected str at index {i}, got {type(t).__name__}: {t!r}")
        texts = [str(t) for t in texts]
        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                text_out = self._model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                features = self._model.text_projection(text_out.pooler_output)
            emb = features.numpy().astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            all_embs.append(emb / norms)
        return np.concatenate(all_embs, axis=0)
