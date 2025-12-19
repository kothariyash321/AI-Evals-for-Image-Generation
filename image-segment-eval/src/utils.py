"""Utility functions for image generation and evaluation."""
import json
import os
from pathlib import Path
from typing import Dict, Any


def load_segments(segments_path: str) -> Dict[str, Any]:
    """Load segment definitions from JSON file."""
    with open(segments_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_base_prompt(base_prompt_path: str) -> str:
    """Load base prompt from text file."""
    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_segment_prompt(base_prompt: str, style_modifiers: list) -> str:
    """Build full prompt by combining base prompt with style modifiers."""
    modifiers_str = ", ".join(style_modifiers)
    return f"{base_prompt}, {modifiers_str}"


def build_segment_descriptor(segment_name: str, segment_data: Dict[str, Any]) -> str:
    """Build text descriptor for segment for CLIP similarity."""
    keywords_str = ", ".join(segment_data.get("keywords", []))
    description = segment_data.get("description", "")
    return f"An ad image that appeals to {segment_name}. Keywords: {keywords_str}. Description: {description}"


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device(force_cpu: bool = False) -> str:
    """Get device (cuda or cpu) for model inference."""
    if force_cpu:
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

