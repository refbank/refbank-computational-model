import os

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

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            vision_out = self._model.vision_model(pixel_values=inputs["pixel_values"])
            features = self._model.visual_projection(vision_out.pooler_output)
        emb = features.numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Returns (N, D) L2-normalized text embeddings."""
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            text_out = self._model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            features = self._model.text_projection(text_out.pooler_output)
        emb = features.numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms
