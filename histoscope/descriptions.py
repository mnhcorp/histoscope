"""Utilities for generating and caching image descriptions."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from typing import Optional

from PIL import Image

import torch

from . import data

HISTOSCOPE_PROMPT = """You are describing a small H&E histology patch for non-experts.

First, provide a brief clinical interpretation (1-2 sentences) using appropriate pathological terminology that a pathologist would use to describe this tissue.

Then, report only what is VISIBLE (colors, shapes, textures, arrangement, relative sizes) in layperson terms. Include the clinical/medical term in parentheses after the layperson description where appropriate. No diagnoses, disease names, grades/stages, or certainty words in the descriptive section.

Use this exact format:
Clinical: <1-2 sentences using pathological terminology>

Gist: <1 sentence, plain English description of what's visible>
Elements:
- <label in 2-4 words> — <1 short sentence of what it looks like + arrangement> (clinical term if applicable)
- <label> — <description> (clinical term if applicable)
- <label> — <description> (clinical term if applicable)

Motifs:
1. <Pattern name> — <Clear description of the visual pattern, arrangement, or structure visible>
2. <Pattern name> — <Description of spatial organization or recurring visual elements>
3. <Pattern name> — <Description of textural or architectural features>

Helpful color anchors for the Elements section (use only if they match what you see): dark-purple/blue = cell centers (nuclei); mid-pink = cell bodies; pale-pink = supporting fibers; clear/white = spaces or mucus-like pools.
Keep total under ~250 words and stick to what the eye can see in the Elements and Motifs sections.
"""

medgemma_model = None
medgemma_processor = None
_class_label: Optional[str] = None
_model_name: Optional[str] = None
_medgemma_enabled: bool = False


def set_medgemma_enabled(enabled: bool) -> None:
    global _medgemma_enabled, medgemma_model, medgemma_processor
    _medgemma_enabled = enabled
    if not enabled:
        medgemma_model = None
        medgemma_processor = None


def set_description_context(model_name: Optional[str], class_label: Optional[str]) -> None:
    global _class_label, _model_name
    _class_label = class_label
    _model_name = model_name


def is_description_cached(model_name: str, image_path: str, class_context: Optional[str] = None) -> bool:
    cache = load_description_cache(model_name)
    cache_key = get_image_cache_key(image_path)
    if cache_key in cache:
        entry = cache[cache_key]
        cached_context = entry.get("class_context")
        if cached_context == class_context:
            return True
    return False


def get_description_cache_path(model_name: str) -> str:
    mdir = data.get_model_dir(model_name)
    cache_dir = os.path.join(mdir, "analysis")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "image_descriptions.json")


def get_image_cache_key(image_path: str) -> str:
    return hashlib.md5(image_path.encode()).hexdigest()


def load_description_cache(model_name: str) -> dict:
    cache_path = get_description_cache_path(model_name)
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Error loading description cache: {exc}")
        return {}


def save_description_cache(model_name: str, cache_data: dict) -> None:
    cache_path = get_description_cache_path(model_name)
    try:
        if len(cache_data) > 1000:
            items = list(cache_data.items())
            items.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
            cache_data = dict(items[:1000])

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as exc:
        print(f"Error saving description cache: {exc}")


def get_cached_description(model_name: str, image_path: str, class_context: Optional[str] = None) -> Optional[str]:
    cache = load_description_cache(model_name)
    cache_key = get_image_cache_key(image_path)
    if cache_key in cache:
        entry = cache[cache_key]
        cached_context = entry.get("class_context")
        if cached_context == class_context:
            return entry.get("description")
    return None


def cache_description(model_name: str, image_path: str, description: str,
                     model_used: str = "medgemma", class_context: Optional[str] = None) -> None:
    cache = load_description_cache(model_name)
    cache_key = get_image_cache_key(image_path)

    cache[cache_key] = {
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "model_used": model_used,
        "class_context": class_context,
    }

    def save_async():
        save_description_cache(model_name, cache)

    threading.Thread(target=save_async, daemon=True).start()


def _init_medgemma() -> None:
    global medgemma_model, medgemma_processor
    if not _medgemma_enabled:
        return
    if medgemma_model is None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

            model_id = "google/medgemma-4b-it"
            medgemma_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            medgemma_processor = AutoProcessor.from_pretrained(model_id)
            print("MedGemma model loaded successfully")
        except Exception as exc:
            print(f"Failed to load MedGemma model: {exc}")
            medgemma_model = "failed"
            medgemma_processor = "failed"


def _medgemma_generate_description(image_path: str) -> str:
    global medgemma_model, medgemma_processor
    if _model_name is None:
        return "Model context missing"

    if not _medgemma_enabled:
        return "MedGemma disabled"

    cached = get_cached_description(_model_name, image_path, _class_label)
    if cached:
        return cached

    if medgemma_model == "failed" or medgemma_processor == "failed":
        return "MedGemma model failed to load"

    if medgemma_model is None or medgemma_processor is None:
        _init_medgemma()

    if medgemma_model in (None, "failed") or medgemma_processor in (None, "failed"):
        return "MedGemma model unavailable"

    try:
        image = Image.open(image_path).convert("RGB")

        if _class_label:
            prompt = (
                f"CLASS CONTEXT: This patch is labeled as '{_class_label}'. "
                f"Use this ONLY to refine the visual description; do not restate the label as a diagnosis.\n\n"
                f"{HISTOSCOPE_PROMPT}"
            )
        else:
            prompt = HISTOSCOPE_PROMPT

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert pathologist."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = medgemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(medgemma_model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = medgemma_model.generate(**inputs, max_new_tokens=250, do_sample=False)
            generation = generation[0][input_len:]

        decoded = medgemma_processor.decode(generation, skip_special_tokens=True)
        description = decoded.strip() if decoded.strip() else "No description returned."

        cache_description(_model_name, image_path, description, model_used="medgemma", class_context=_class_label)
        return description

    except Exception as exc:
        return f"MedGemma generation failed: {exc}"


def _gemini_generate_description(image_path: str) -> str:
    if _model_name is None:
        return "Model context missing"

    cached = get_cached_description(_model_name, image_path, _class_label)
    if cached:
        return cached

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client()  # Uses GEMINI_API_KEY from environment
    except Exception as exc:
        return f"Gemini client init failed: {exc}"

    import mimetypes

    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    try:
        with open(image_path, "rb") as f:
            data_bytes = f.read()
        img_part = types.Part.from_bytes(data=data_bytes, mime_type=mime)
    except Exception as exc:
        return f"Failed reading image: {exc}"

    prompt = HISTOSCOPE_PROMPT
    if _class_label:
        prompt = (
            f"CLASS CONTEXT: This patch is labeled as '{_class_label}'. "
            f"Use this ONLY to refine the visual description; do not restate the label as a diagnosis.\n\n"
            f"{HISTOSCOPE_PROMPT}"
        )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img_part, prompt],
        )
        text = getattr(resp, "text", "").strip()
        description = text if text else "No description returned."
        cache_description(_model_name, image_path, description, model_used="gemini", class_context=_class_label)
        return description
    except Exception as exc:
        return f"Gemini generation failed: {exc}"


def describe_patch(image_path: str) -> str:
    """Try Gemini first, then fall back to MedGemma."""
    description = _gemini_generate_description(image_path)
    if _medgemma_enabled and description.startswith("Gemini") and ("failed" in description or "error" in description.lower()):
        description = _medgemma_generate_description(image_path)
    return description


def ensure_models_loaded() -> None:
    _init_medgemma()
