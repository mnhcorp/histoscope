"""Shared utility helpers for image encoding."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Tuple

from PIL import Image


def _encode_image(img: Image.Image, format_: str = "JPEG", quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format=format_, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_thumb(path: str, max_side: int = 256) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return _encode_image(img, quality=85)


def encode_large_image(path: str, max_side: int = 800) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return _encode_image(img, quality=90)


def encode_extra_large_image(path: str, max_side: int = 1200) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return _encode_image(img, quality=95)


def encode_original_size_image(path: str) -> str:
    img = Image.open(path).convert("RGB")
    return _encode_image(img, quality=95)


def get_image_dimensions(path: str) -> Tuple[int, int]:
    img = Image.open(path)
    return img.size
