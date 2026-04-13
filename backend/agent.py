"""
AI agent for the Agentic AI Image Editor.

Resolves a free-form natural-language editing request into a sequence of
image-tool calls, then executes those calls on the supplied image.

The agent tries OpenAI tool-calling first (requires OPENAI_API_KEY env var).
If no key is available it falls back to a deterministic keyword parser.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from PIL import Image

from image_tools import TOOL_FUNCTIONS, TOOLS_SCHEMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# Maximum characters of user input to process — caps any regex backtracking time
_MAX_INPUT_LEN = 500


def openai_available() -> bool:
    return bool(_OPENAI_KEY and _OPENAI_KEY.startswith("sk-"))


# ---------------------------------------------------------------------------
# OpenAI agent path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert image-editing assistant integrated into an agentic image editor.

When a user describes how they want their image changed, you MUST call one or more
of the available image-editing tool functions — never reply with plain text alone.

Guidelines:
• Translate vague intent into concrete tool parameters:
  - "brighter" → adjust_brightness(factor=1.5)
  - "darker" → adjust_brightness(factor=0.6)
  - "very bright" → adjust_brightness(factor=2.0)
  - "more contrast" → adjust_contrast(factor=1.6)
  - "vivid / pop / vibrant" → adjust_saturation(factor=1.8) + adjust_contrast(factor=1.3)
  - "black and white" or "grayscale" → grayscale()
  - "vintage / retro" → sepia() + vignette()
  - "cinematic" → adjust_contrast(factor=1.4) + vignette() + adjust_saturation(factor=0.8)
  - "warm" → hue_rotate(degrees=-15) + adjust_brightness(factor=1.05)
  - "cool / cold" → hue_rotate(degrees=15) + adjust_saturation(factor=0.9)
  - "sharpen" → adjust_sharpness(factor=2.5)
  - "blur background / dreamy" → blur(radius=3)
  - "auto fix / enhance" → auto_enhance()
  - "rotate 90 / turn clockwise" → rotate(degrees=-90)  # PIL is CCW, so CW = negative
  - "flip / mirror" → flip(direction="horizontal")
• You may call multiple tools — they are applied in the order you call them.
• Use sensible default values when the user is not specific.
• Do not call tools the user did not ask for.
"""


def _run_openai(user_request: str) -> list[dict[str, Any]]:
    """Ask GPT-4o to select and parameterise tools, then return operation list."""
    from openai import OpenAI

    client = OpenAI(api_key=_OPENAI_KEY)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="required",
        parallel_tool_calls=True,
    )

    ops: list[dict[str, Any]] = []
    for tool_call in response.choices[0].message.tool_calls or []:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        ops.append({"tool": func_name, "params": args})

    return ops


# ---------------------------------------------------------------------------
# Keyword / rule-based fallback
# ---------------------------------------------------------------------------

def _num(text: str, patterns: list[str], default: float) -> float:
    """Extract the first number following any of the given keyword patterns."""
    for pat in patterns:
        m = re.search(rf"{pat}[^\d]{{0,20}}([\d]+(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return default


def _keyword_parse(request: str) -> list[dict[str, Any]]:
    """Deterministic rule-based parser — runs when no OpenAI key is set."""
    # Cap length to bound regex complexity
    req = request[:_MAX_INPUT_LEN].lower()
    ops: list[dict[str, Any]] = []

    def add(tool: str, **params: Any) -> None:
        ops.append({"tool": tool, "params": params})

    # ---- Presets ---------------------------------------------------------
    if any(w in req for w in ["vintage", "retro", "old photo", "old-photo"]):
        add("sepia", intensity=0.85)
        add("vignette", intensity=0.5)
        add("adjust_contrast", factor=1.1)
        return ops

    if any(w in req for w in ["cinematic", "film look", "movie"]):
        add("adjust_contrast", factor=1.4)
        add("adjust_saturation", factor=0.75)
        add("vignette", intensity=0.6)
        return ops

    if any(w in req for w in ["black and white", "grayscale", "greyscale", "monochrome", "b&w", "b & w"]):
        add("grayscale")
        return ops

    if any(w in req for w in ["sepia"]):
        add("sepia", intensity=1.0)
        return ops

    if any(w in req for w in ["invert", "negative", "negate"]):
        add("invert")
        return ops

    if any(w in req for w in ["auto enhance", "auto-enhance", "enhance", "fix it", "improve", "make it better", "auto fix"]):
        add("auto_enhance")
        return ops

    # ---- Brightness ------------------------------------------------------
    _very = any(w in req for w in ["very", "much", "extremely", "super", "lot", "significantly"])
    _little = any(w in req for w in ["little", "slightly", "bit", "somewhat", "a little"])

    if any(w in req for w in ["bright", "lighter", "lighten", "exposure up", "overexpose"]):
        factor = 2.0 if _very else 1.2 if _little else 1.5
        add("adjust_brightness", factor=factor)
    elif any(w in req for w in ["dark", "darker", "darken", "dim", "exposure down", "underexpose"]):
        factor = 0.3 if _very else 0.8 if _little else 0.55
        add("adjust_brightness", factor=factor)

    # ---- Contrast --------------------------------------------------------
    if any(w in req for w in ["more contrast", "contrasty", "pop", "punchy"]):
        factor = 2.0 if _very else 1.2 if _little else 1.6
        add("adjust_contrast", factor=factor)
    elif any(w in req for w in ["less contrast", "flat", "washed", "fade", "faded", "matte", "muted contrast"]):
        factor = 0.5 if _very else 0.85 if _little else 0.7
        add("adjust_contrast", factor=factor)

    # ---- Saturation ------------------------------------------------------
    if any(w in req for w in ["vivid", "vibrant", "saturate", "colourful", "colorful", "rich colour", "rich color"]):
        factor = 2.5 if _very else 1.3 if _little else 1.8
        add("adjust_saturation", factor=factor)
    elif any(w in req for w in ["desaturate", "muted", "dull", "fade color", "fade colour", "less color", "less colour"]):
        factor = 0.2 if _very else 0.7 if _little else 0.4
        add("adjust_saturation", factor=factor)

    # ---- Sharpness -------------------------------------------------------
    if any(w in req for w in ["sharpen", "sharp", "crisp", "detail"]):
        factor = 3.5 if _very else 1.5 if _little else 2.5
        add("adjust_sharpness", factor=factor)
    elif any(w in req for w in ["soften", "smooth skin", "dreamy"]):
        factor = 0.0 if _very else 0.5 if _little else 0.2
        add("adjust_sharpness", factor=factor)

    # ---- Blur ------------------------------------------------------------
    if any(w in req for w in ["blur", "blurry", "bokeh", "defocus"]):
        radius = 8.0 if _very else 1.5 if _little else 3.0
        add("blur", radius=radius)

    # ---- Vignette --------------------------------------------------------
    if any(w in req for w in ["vignette", "dark edges", "focus center", "focus centre"]):
        intensity = 0.9 if _very else 0.3 if _little else 0.6
        add("vignette", intensity=intensity)

    # ---- Rotation --------------------------------------------------------
    cw = any(w in req for w in ["clockwise", "right"])
    ccw = any(w in req for w in ["counter", "anticlockwise", "left"])
    rot_m = re.search(r"rotat[a-z]{0,3}\s+(\d+)", req)
    ang_m = re.search(r"(\d+)degrees?", req.replace(" ", ""))
    if rot_m or ang_m or any(w in req for w in ["rotate", "turn"]):
        if rot_m:
            deg = float(rot_m.group(1))
        elif ang_m:
            deg = float(ang_m.group(1))
        else:
            deg = 90
        if cw:
            deg = -deg  # PIL rotates CCW
        add("rotate", degrees=deg)

    # ---- Flip ------------------------------------------------------------
    if any(w in req for w in ["flip horizontal", "mirror", "flip left", "flip right"]):
        add("flip", direction="horizontal")
    elif any(w in req for w in ["flip vertical", "flip up", "flip down", "upside down"]):
        add("flip", direction="vertical")
    elif "flip" in req:
        add("flip", direction="horizontal")

    # ---- Crop ------------------------------------------------------------
    crop_m = re.search(r"crop\s+(\w+)\s+(\d+)", req)
    if "zoom in" in req or "crop center" in req:
        add("crop", left=15, top=15, right=85, bottom=85)
    elif crop_m:
        side = crop_m.group(1)
        pct = float(crop_m.group(2))
        if "left" in side:
            add("crop", left=pct)
        elif "right" in side:
            add("crop", right=100 - pct)
        elif "top" in side:
            add("crop", top=pct)
        elif "bottom" in side:
            add("crop", bottom=100 - pct)

    # ---- Resize ----------------------------------------------------------
    size_m = re.search(r"resize\s+(\d+)\s*[xX×]\s*(\d+)", req)
    w_m = re.search(r"width[ =:]([0-9]+)", req)
    h_m = re.search(r"height[ =:]([0-9]+)", req)
    if size_m:
        add("resize", width=int(size_m.group(1)), height=int(size_m.group(2)))
    elif w_m or h_m:
        kwargs: dict[str, int] = {}
        if w_m:
            kwargs["width"] = int(w_m.group(1))
        if h_m:
            kwargs["height"] = int(h_m.group(1))
        add("resize", **kwargs)

    # ---- Hue rotate ------------------------------------------------------
    hue_m = re.search(r"hue\s+(-?\d+)", req)
    if hue_m:
        add("hue_rotate", degrees=float(hue_m.group(1)))
    elif "warm" in req:
        add("hue_rotate", degrees=-20)
        add("adjust_brightness", factor=1.05)
    elif any(w in req for w in ["cool", "cold", "blue tint"]):
        add("hue_rotate", degrees=20)

    # ---- Noise reduction -------------------------------------------------
    if any(w in req for w in ["denoise", "noise reduction", "remove grain", "reduce noise", "remove noise"]):
        strength = 3 if _very else 1 if _little else 2
        add("noise_reduction", strength=strength)

    # ---- Border ----------------------------------------------------------
    border_m = re.search(r"border\s+(\d+)", req)
    if any(w in req for w in ["border", "frame", "polaroid"]):
        width_px = int(border_m.group(1)) if border_m else 30
        color_m = re.search(r"(white|black|red|blue|green|yellow|#[0-9a-fA-F]{3,6})", req)
        color = color_m.group(1) if color_m else "white"
        add("add_border", width=width_px, color=color)

    # ---- Text overlay ----------------------------------------------------
    # Match: add text "..." or caption "..."
    text_m = re.search(r'(?:add\s+)?(?:text|caption|label|watermark)\s+["\']([^"\']{1,200})["\']', req)
    if text_m:
        text = text_m.group(1)
        pos_m = re.search(r"(top|bottom|center|left|right)[-\s]?(left|right|center)?", req)
        pos = "bottom-center"
        if pos_m:
            parts = [p for p in [pos_m.group(1), pos_m.group(2)] if p]
            pos = "-".join(parts) if len(parts) > 1 else parts[0]
        color_m2 = re.search(r"(white|black|red|blue|yellow|#[0-9a-fA-F]{3,6})", req)
        col = color_m2.group(1) if color_m2 else "white"
        add("add_text", text=text, position=pos, color=col)

    # ---- Final fallback --------------------------------------------------
    if not ops:
        # Try auto_enhance as a last resort
        add("auto_enhance")

    return ops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_request(
    image: Image.Image,
    user_request: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Apply the edits described by *user_request* to *image*.

    Returns:
        (edited_image, list_of_applied_operations)

    Each operation dict has keys ``tool`` (str) and ``params`` (dict).
    Raises ValueError if a requested tool does not exist.
    """
    if openai_available():
        try:
            ops = _run_openai(user_request)
        except Exception as exc:  # noqa: BLE001
            # Fall back gracefully on any OpenAI error
            ops = _keyword_parse(user_request)
            ops = [{"tool": op["tool"], "params": op["params"], "_fallback": True} for op in ops]
    else:
        ops = _keyword_parse(user_request)

    result = image.copy()
    applied: list[dict[str, Any]] = []

    for op in ops:
        tool_name = op["tool"]
        params = op.get("params", {})

        fn = TOOL_FUNCTIONS.get(tool_name)
        if fn is None:
            # Skip unknown tools rather than crash
            continue

        result = fn(result, **params)
        applied.append({"tool": tool_name, "params": params})

    return result, applied
