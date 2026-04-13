"""
Image editing tools for the Agentic AI Image Editor.

Each function accepts a PIL Image as its first argument and returns a PIL Image.
All parameters have sensible defaults so they work well with AI-generated calls.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def resize(
    image: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True,
) -> Image.Image:
    """Resize image to the specified dimensions.

    If only one dimension is given and keep_aspect_ratio is True, the other
    dimension is calculated to preserve the original aspect ratio.
    """
    orig_w, orig_h = image.size

    if width is None and height is None:
        return image

    if keep_aspect_ratio:
        if width and not height:
            ratio = width / orig_w
            height = max(1, int(orig_h * ratio))
        elif height and not width:
            ratio = height / orig_h
            width = max(1, int(orig_w * ratio))
        else:
            ratio = min(width / orig_w, height / orig_h)
            width = max(1, int(orig_w * ratio))
            height = max(1, int(orig_h * ratio))
    else:
        width = width or orig_w
        height = height or orig_h

    return image.resize((width, height), Image.LANCZOS)


def crop(
    image: Image.Image,
    left: float = 0,
    top: float = 0,
    right: float = 100,
    bottom: float = 100,
) -> Image.Image:
    """Crop image using percentage coordinates (0–100).

    left/top/right/bottom are percentages of the image dimensions, so
    crop(left=10, top=10, right=90, bottom=90) removes a 10 % border.
    """
    w, h = image.size
    l = int(w * left / 100)
    t = int(h * top / 100)
    r = int(w * right / 100)
    b = int(h * bottom / 100)
    # Guard against degenerate boxes
    r = max(l + 1, r)
    b = max(t + 1, b)
    return image.crop((l, t, r, b))


def rotate(
    image: Image.Image,
    degrees: float = 90,
    expand: bool = True,
) -> Image.Image:
    """Rotate image counter-clockwise by *degrees*.

    When expand=True the output canvas is enlarged so no content is clipped.
    """
    return image.rotate(degrees, expand=expand, resample=Image.BICUBIC)


def flip(image: Image.Image, direction: str = "horizontal") -> Image.Image:
    """Flip image horizontally or vertically.

    direction: "horizontal" (mirror left-right) or "vertical" (flip top-bottom).
    """
    direction = direction.lower()
    if direction in {"horizontal", "h", "lr", "left-right", "mirror"}:
        return ImageOps.mirror(image)
    return ImageOps.flip(image)


# ---------------------------------------------------------------------------
# Tone / colour adjustments
# ---------------------------------------------------------------------------


def adjust_brightness(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Adjust brightness.

    factor: 1.0 = original, < 1.0 darker, > 1.0 brighter. Typical range 0.0–3.0.
    """
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Adjust contrast.

    factor: 1.0 = original, < 1.0 flatter, > 1.0 more contrast.
    """
    return ImageEnhance.Contrast(image).enhance(factor)


def adjust_saturation(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Adjust colour saturation.

    factor: 1.0 = original, 0.0 = grayscale, 2.0 = very vivid.
    """
    return ImageEnhance.Color(image).enhance(factor)


def adjust_sharpness(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """Adjust sharpness / detail.

    factor: 1.0 = original, 0.0 = blurred, > 1.0 = sharper.
    """
    return ImageEnhance.Sharpness(image).enhance(factor)


def hue_rotate(image: Image.Image, degrees: float = 30) -> Image.Image:
    """Rotate all hue values by *degrees* (-180 to 180).

    Uses vectorised numpy HSV maths for speed.
    """
    img_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = img_rgb.shape[:2]
    pixels = img_rgb.reshape(-1, 3)

    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros(len(r), dtype=np.float32)
    mask = delta != 0
    rm = (cmax == r) & mask
    gm = (cmax == g) & mask
    bm = (cmax == b) & mask

    hue[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6
    hue[gm] = (b[gm] - r[gm]) / delta[gm] + 2
    hue[bm] = (r[bm] - g[bm]) / delta[bm] + 4
    hue /= 6.0

    sat = np.zeros_like(hue)
    sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    val = cmax

    # Rotate hue
    hue = (hue + degrees / 360.0) % 1.0

    # HSV → RGB
    h6 = hue * 6.0
    i = h6.astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)

    out = np.zeros((len(r), 3), dtype=np.float32)
    for idx, (rv, gv, bv) in enumerate(
        [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    ):
        sel = i == idx
        out[sel, 0] = rv[sel]
        out[sel, 1] = gv[sel]
        out[sel, 2] = bv[sel]

    result = (np.clip(out, 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)
    return Image.fromarray(result).convert(image.mode)


# ---------------------------------------------------------------------------
# Filters / effects
# ---------------------------------------------------------------------------


def blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur with the given radius."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def grayscale(image: Image.Image) -> Image.Image:
    """Convert image to grayscale (while preserving the original mode)."""
    return ImageOps.grayscale(image).convert(image.mode)


def sepia(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Apply a classic sepia tone effect.

    intensity: 0.0 = no change, 1.0 = full sepia.
    """
    img_array = np.array(image.convert("RGB"), dtype=np.float64)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    nr = np.clip(
        r * (1 - 0.607 * intensity)
        + g * (0.769 * intensity)
        + b * (0.189 * intensity),
        0,
        255,
    )
    ng = np.clip(
        r * (0.349 * intensity)
        + g * (1 - 0.314 * intensity)
        + b * (0.168 * intensity),
        0,
        255,
    )
    nb = np.clip(
        r * (0.272 * intensity)
        + g * (0.534 * intensity)
        + b * (1 - 0.869 * intensity),
        0,
        255,
    )

    result = np.stack([nr, ng, nb], axis=2).astype(np.uint8)
    out = Image.fromarray(result)
    return out.convert(image.mode)


def invert(image: Image.Image) -> Image.Image:
    """Invert (negate) all pixel colours."""
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        rgb = Image.merge("RGB", (r, g, b))
        inv = ImageOps.invert(rgb)
        ir, ig, ib = inv.split()
        return Image.merge("RGBA", (ir, ig, ib, a))
    return ImageOps.invert(image.convert("RGB")).convert(image.mode)


def vignette(image: Image.Image, intensity: float = 0.6) -> Image.Image:
    """Apply a radial vignette (dark edges) effect.

    intensity: 0.0 = no vignette, 1.0 = heavy vignette.
    """
    img_array = np.array(image.convert("RGB"), dtype=np.float64)
    h, w = img_array.shape[:2]

    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((X - cx) / (w / 2.0)) ** 2 + ((Y - cy) / (h / 2.0)) ** 2)
    mask = 1.0 - np.clip(intensity * dist, 0.0, 1.0)

    result = np.clip(img_array * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(result).convert(image.mode)


def auto_enhance(image: Image.Image) -> Image.Image:
    """Automatically enhance brightness, contrast, and sharpness."""
    img = ImageOps.autocontrast(image, cutoff=0.5)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


def noise_reduction(image: Image.Image, strength: int = 1) -> Image.Image:
    """Reduce noise using a median filter.

    strength: 1 = light, 2 = moderate, 3 = heavy.
    """
    size = max(3, 3 + (strength - 1) * 2)
    if size % 2 == 0:
        size += 1
    return image.filter(ImageFilter.MedianFilter(size=size))


def add_border(
    image: Image.Image,
    width: int = 20,
    color: str = "white",
) -> Image.Image:
    """Add a solid-colour border around the image."""
    return ImageOps.expand(image, border=width, fill=color)


def add_text(
    image: Image.Image,
    text: str = "Sample Text",
    position: str = "bottom-center",
    font_size: int = 40,
    color: str = "white",
) -> Image.Image:
    """Overlay *text* on the image at the given position.

    position: "top-left", "top-center", "top-right",
              "center", "bottom-left", "bottom-center" (default), "bottom-right".
    """
    img = image.convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Try to load a real font; fall back to the PIL default
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    iw, ih = img.size
    pad = max(10, font_size // 3)

    pos_lc = position.lower()
    if "top" in pos_lc and "left" in pos_lc:
        x, y = pad, pad
    elif "top" in pos_lc and "right" in pos_lc:
        x, y = iw - tw - pad, pad
    elif "top" in pos_lc:
        x, y = (iw - tw) // 2, pad
    elif "bottom" in pos_lc and "left" in pos_lc:
        x, y = pad, ih - th - pad
    elif "bottom" in pos_lc and "right" in pos_lc:
        x, y = iw - tw - pad, ih - th - pad
    elif "bottom" in pos_lc:
        x, y = (iw - tw) // 2, ih - th - pad
    else:  # center
        x, y = (iw - tw) // 2, (ih - th) // 2

    # Drop shadow
    shadow_off = max(2, font_size // 15)
    draw.text((x + shadow_off, y + shadow_off), text, fill=(0, 0, 0, 180), font=font)
    draw.text((x, y), text, fill=color, font=font)

    return img.convert(image.mode)


# ---------------------------------------------------------------------------
# Registry — maps tool name → callable
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, callable] = {
    "resize": resize,
    "crop": crop,
    "rotate": rotate,
    "flip": flip,
    "adjust_brightness": adjust_brightness,
    "adjust_contrast": adjust_contrast,
    "adjust_saturation": adjust_saturation,
    "adjust_sharpness": adjust_sharpness,
    "hue_rotate": hue_rotate,
    "blur": blur,
    "grayscale": grayscale,
    "sepia": sepia,
    "invert": invert,
    "vignette": vignette,
    "auto_enhance": auto_enhance,
    "noise_reduction": noise_reduction,
    "add_border": add_border,
    "add_text": add_text,
}

# OpenAI function-calling schema for every tool
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "resize",
            "description": (
                "Resize the image to specified pixel dimensions. "
                "Use when the user mentions changing the size, making it smaller/larger, "
                "or specifying pixel dimensions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Target width in pixels.",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Target height in pixels.",
                    },
                    "keep_aspect_ratio": {
                        "type": "boolean",
                        "description": "Preserve aspect ratio (default true).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop",
            "description": (
                "Crop the image using percentage-based coordinates (0–100). "
                "Use when the user wants to remove edges, zoom in, or focus on a region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {
                        "type": "number",
                        "description": "Left edge as % from the left (default 0).",
                    },
                    "top": {
                        "type": "number",
                        "description": "Top edge as % from the top (default 0).",
                    },
                    "right": {
                        "type": "number",
                        "description": "Right edge as % from the left (default 100).",
                    },
                    "bottom": {
                        "type": "number",
                        "description": "Bottom edge as % from the top (default 100).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": (
                "Rotate the image counter-clockwise by the given degrees. "
                "Use when the user says 'rotate', 'turn', 'tilt', or mentions an angle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "number",
                        "description": "Rotation angle in degrees (positive = counter-clockwise). E.g. 90, 180, 270, 45.",
                    },
                    "expand": {
                        "type": "boolean",
                        "description": "Expand canvas to avoid clipping (default true).",
                    },
                },
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flip",
            "description": (
                "Flip the image horizontally (mirror) or vertically. "
                "Use when the user says 'flip', 'mirror', 'reverse'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["horizontal", "vertical"],
                        "description": "'horizontal' mirrors left-right; 'vertical' flips top-bottom.",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_brightness",
            "description": (
                "Make the image brighter or darker. "
                "factor > 1.0 = brighter, factor < 1.0 = darker. "
                "Use when the user mentions brightness, lightness, or exposure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": (
                            "Brightness multiplier. "
                            "0.5 = half brightness, 1.0 = unchanged, 1.5 = 50% brighter, 2.0 = double."
                        ),
                    },
                },
                "required": ["factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_contrast",
            "description": (
                "Increase or decrease image contrast. "
                "factor > 1.0 = more contrast, factor < 1.0 = flat/washed out. "
                "Use when the user mentions contrast, pop, flat, or washed-out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Contrast multiplier. 1.0 = unchanged, 1.5 = moderately more contrast.",
                    },
                },
                "required": ["factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_saturation",
            "description": (
                "Adjust colour vividness/saturation. "
                "0 = grayscale, 1.0 = original, 2.0 = vibrant. "
                "Use for 'vivid', 'vibrant', 'dull', 'muted', 'desaturate'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Saturation multiplier. 0 = grayscale, 1.0 = original, 2.0 = very vivid.",
                    },
                },
                "required": ["factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_sharpness",
            "description": (
                "Sharpen or soften the image. "
                "factor > 1.0 = sharper, factor < 1.0 = softer/smoother. "
                "Use for 'sharpen', 'crisp', 'soft', 'smooth'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Sharpness multiplier. 0.0 = very soft, 1.0 = original, 2.0 = noticeably sharper.",
                    },
                },
                "required": ["factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hue_rotate",
            "description": (
                "Shift all hue values by a number of degrees, changing colours globally. "
                "Use when the user wants to change colours, shift hue, or get a colour effect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "number",
                        "description": "Hue shift in degrees (-180 to 180). 180 inverts all hues.",
                    },
                },
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blur",
            "description": (
                "Apply Gaussian blur to soften or defocus the image. "
                "Use for 'blur', 'defocus', 'soften', 'bokeh', or privacy masking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "Blur radius in pixels. 1 = subtle, 5 = heavy blur.",
                    },
                },
                "required": ["radius"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grayscale",
            "description": (
                "Convert the image to black and white (grayscale). "
                "Use for 'black and white', 'greyscale', 'monochrome', 'desaturate completely'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sepia",
            "description": (
                "Apply a warm sepia/vintage brown tone. "
                "Use for 'sepia', 'vintage', 'retro', 'old-photo', 'aged' look."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intensity": {
                        "type": "number",
                        "description": "Effect strength. 0.0 = no effect, 1.0 = full sepia.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invert",
            "description": (
                "Invert all colours (like a photo negative). "
                "Use for 'invert', 'negative', 'negate', 'complement'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vignette",
            "description": (
                "Add a dark vignette around the edges, drawing focus to the centre. "
                "Use for 'vignette', 'dark edges', 'focus centre', 'cinematic'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intensity": {
                        "type": "number",
                        "description": "Vignette strength. 0.0 = none, 0.5 = medium, 1.0 = heavy.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_enhance",
            "description": (
                "Automatically improve overall image quality — contrast, brightness, and sharpness. "
                "Use for 'enhance', 'improve', 'fix', 'auto', 'make it look better'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noise_reduction",
            "description": (
                "Reduce graininess or digital noise using a median filter. "
                "Use for 'denoise', 'remove noise', 'reduce grain', 'smooth out noise'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strength": {
                        "type": "integer",
                        "description": "Filter strength: 1 = light, 2 = moderate, 3 = heavy.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_border",
            "description": (
                "Add a solid-colour border/frame around the image. "
                "Use for 'add border', 'frame', 'white border', 'polaroid'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Border width in pixels (default 20).",
                    },
                    "color": {
                        "type": "string",
                        "description": "Border colour as a CSS colour name or hex code (e.g. 'white', '#000000').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_text",
            "description": (
                "Overlay text on the image. "
                "Use when the user wants to add a caption, label, watermark, or any text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text string to add.",
                    },
                    "position": {
                        "type": "string",
                        "description": (
                            "Where to place the text: "
                            "'top-left', 'top-center', 'top-right', "
                            "'center', 'bottom-left', 'bottom-center', 'bottom-right'."
                        ),
                    },
                    "font_size": {
                        "type": "integer",
                        "description": "Font size in points (default 40).",
                    },
                    "color": {
                        "type": "string",
                        "description": "Text colour as a CSS colour name or hex code (default 'white').",
                    },
                },
                "required": ["text"],
            },
        },
    },
]
