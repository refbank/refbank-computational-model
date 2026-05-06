import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image


def _make_rgba_png(fg_pixel=(0, 0, 0, 255), bg_alpha=0, size=(4, 4)) -> str:
    """Write a PNG with one opaque corner pixel and the rest transparent; return path."""
    img = Image.new("RGBA", size, (0, 0, 0, bg_alpha))
    img.putpixel((0, 0), fg_pixel)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def test_open_image_composites_transparent_onto_white():
    """RGBA images with transparent backgrounds must composite onto white, not black."""
    from code.embeddings.clip_encoder import _open_image

    # Black foreground pixel at (0,0); everything else transparent.
    path = _make_rgba_png(fg_pixel=(0, 0, 0, 255), bg_alpha=0)
    try:
        result = _open_image(path)
        arr = np.array(result)

        assert result.mode == "RGB"
        # Background (transparent) should become white after compositing.
        bg_pixel = arr[1, 1]
        assert np.all(bg_pixel == 255), (
            f"Expected white background (255,255,255), got {bg_pixel}. "
            "Transparent pixels must be composited onto white, not left as black."
        )
        # Foreground (black ink) stays black.
        fg_pixel = arr[0, 0]
        assert np.all(fg_pixel == 0), f"Expected black foreground (0,0,0), got {fg_pixel}"
    finally:
        os.unlink(path)


def test_open_image_different_rgba_images_differ():
    """Two RGBA images with different content produce different pixel arrays."""
    from code.embeddings.clip_encoder import _open_image

    path_red = _make_rgba_png(fg_pixel=(255, 0, 0, 255))
    path_blue = _make_rgba_png(fg_pixel=(0, 0, 255, 255))
    try:
        arr_red = np.array(_open_image(path_red))
        arr_blue = np.array(_open_image(path_blue))
        assert not np.allclose(arr_red, arr_blue), "Different images must produce different pixel arrays"
    finally:
        os.unlink(path_red)
        os.unlink(path_blue)
