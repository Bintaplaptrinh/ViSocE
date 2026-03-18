"""Text normalization utilities for Vietnamese social text."""

from __future__ import annotations

import re
from pathlib import Path

from .utils import read_json


VIETNAMESE_PATTERN = re.compile(
    r"[^a-zA-Z0-9\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩị"
    r"óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+"
)


def load_slang_dict(path: Path) -> dict[str, str]:
    data = read_json(path)
    return {str(k).lower().strip(): str(v).lower().strip() for k, v in data.items()}


def normalize_text(text: str, slang_dict: dict[str, str]) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Replace longer slang phrases first to avoid partial replacement side effects.
    for slang, meaning in sorted(slang_dict.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(slang) + r"\b", meaning, text)

    text = VIETNAMESE_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
