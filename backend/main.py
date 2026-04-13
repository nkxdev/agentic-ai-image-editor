"""
FastAPI application entry point for the Agentic AI Image Editor.

Start the server:
    uvicorn main:app --reload --port 8000

Or directly:
    python main.py
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic AI Image Editor",
    description="Edit images with natural language commands powered by AI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Serve everything under /static (including uploaded images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EditRequest(BaseModel):
    session_id: str
    request: str


class EditResponse(BaseModel):
    session_id: str
    original_url: str
    edited_url: str
    operations: list[dict]
    message: str
    used_ai: bool


# ---------------------------------------------------------------------------
# In-memory session store  {session_id: Path}
# ---------------------------------------------------------------------------

_sessions: dict[str, Path] = {}


def _session_path(session_id: str) -> Path:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload an image first.")
    return _sessions[session_id]


def _save_image(image: Image.Image, prefix: str = "img") -> Path:
    filename = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    path = UPLOADS_DIR / filename
    image.save(str(path), format="PNG")
    return path


def _url(path: Path) -> str:
    return f"/static/uploads/{path.name}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """Serve the single-page frontend."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a new image and start a session.

    Returns a session_id and the URL of the stored image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()  # Check integrity
        image = Image.open(io.BytesIO(data))  # Re-open after verify
        image = image.convert("RGBA") if image.mode in ("P", "PA") else image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}") from exc

    session_id = uuid.uuid4().hex
    path = _save_image(image, prefix="original")
    _sessions[session_id] = path

    return JSONResponse(
        {
            "session_id": session_id,
            "original_url": _url(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
    )


@app.post("/edit", response_model=EditResponse)
async def edit_image(body: EditRequest) -> EditResponse:
    """Apply an edit described in natural language to the current session image.

    The agent may call multiple image tools; the edited image replaces the
    session's current image so commands can be chained.
    """
    original_path = _session_path(body.session_id)

    try:
        image = Image.open(str(original_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot open session image: {exc}") from exc

    # Import here to benefit from dotenv loaded above
    from agent import openai_available, process_request  # noqa: PLC0415

    try:
        edited_image, operations = process_request(image, body.request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    # Persist edited image as new session image (enables chaining)
    edited_path = _save_image(edited_image, prefix="edited")
    _sessions[body.session_id] = edited_path

    # Human-readable summary
    if operations:
        parts = []
        for op in operations:
            params_str = ", ".join(f"{k}={v}" for k, v in op["params"].items())
            parts.append(f"**{op['tool']}**({params_str})" if params_str else f"**{op['tool']}**()")
        message = "Applied: " + " → ".join(parts)
    else:
        message = "No changes were applied."

    return EditResponse(
        session_id=body.session_id,
        original_url=_url(original_path),
        edited_url=_url(edited_path),
        operations=operations,
        message=message,
        used_ai=openai_available(),
    )


@app.get("/tools")
async def list_tools() -> JSONResponse:
    """Return the list of available image-editing tools with descriptions."""
    from image_tools import TOOLS_SCHEMA  # noqa: PLC0415

    tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
        }
        for t in TOOLS_SCHEMA
    ]
    return JSONResponse({"tools": tools, "count": len(tools)})


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    """Clean up a session (remove stored images)."""
    if session_id in _sessions:
        try:
            _sessions[session_id].unlink(missing_ok=True)
        except OSError:
            pass
        del _sessions[session_id]
    return JSONResponse({"deleted": session_id})


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
