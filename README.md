# Agentic AI Image Editor

A full-stack application that lets you edit images using **natural language commands**.
Describe what you want and the AI agent selects and applies the right image-processing
tools automatically — no manual sliders needed.

## Features

| Capability | Details |
|---|---|
| **18 image editing tools** | resize, crop, rotate, flip, brightness, contrast, saturation, sharpness, blur, grayscale, sepia, invert, hue rotate, vignette, auto-enhance, noise reduction, border, text overlay |
| **Agentic intent parsing** | Uses OpenAI GPT-4o tool-calling when `OPENAI_API_KEY` is set; falls back to a smart keyword parser |
| **Chainable edits** | Each command builds on the previous result — edit iteratively |
| **Before/After slider** | Drag the divider to compare original vs edited |
| **16 Quick prompts** | One-click presets for common edits |
| **Download** | Save the edited image with a single click |

## Quick Start

```bash
cd backend
pip install -r requirements.txt

# Optional: add your OpenAI key for GPT-4o powered parsing
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

python main.py
```

Open **http://localhost:8000** in your browser.

## Project Structure

```
backend/
├── main.py           # FastAPI application (upload / edit / tools endpoints)
├── image_tools.py    # All 18 image editing functions + OpenAI schema
├── agent.py          # Agentic layer: GPT-4o tool-calling + keyword fallback
├── requirements.txt
├── .env.example
└── static/
    └── index.html    # Single-page UI (Tailwind CSS, before/after slider, chat)
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/upload` | Upload an image; returns `session_id` |
| `POST` | `/edit` | Apply natural-language edits to a session image |
| `GET` | `/tools` | List all available tools with descriptions |
| `DELETE` | `/session/{id}` | Remove a session and its stored images |

## Example Edit Commands

- `"make it brighter"`
- `"vintage / retro look"`
- `"black and white"`
- `"rotate 90 degrees clockwise"`
- `"add a white border"`
- `"cinematic look"`
- `"sharpen the image"`
- `"add blur"` / `"bokeh effect"`
- `"flip horizontal"`
- `"add text 'Hello World' at the bottom"`
- `"warm tones"` / `"cool tones"`
- `"auto enhance"`

## Architecture

```
User request (text)
       │
       ▼
   agent.py
  ┌────────────────────────────────────────┐
  │  OpenAI GPT-4o tool-calling            │  ← if OPENAI_API_KEY is set
  │  (picks functions + parameters)        │
  └─────────────────┬──────────────────────┘
                    │  else
  ┌─────────────────▼──────────────────────┐
  │  Keyword/rule-based parser             │  ← always available
  │  (regex + intent matching)             │
  └─────────────────┬──────────────────────┘
                    │
                    ▼
           image_tools.py
           (Pillow + numpy)
                    │
                    ▼
           Edited image saved
           & URL returned to UI
```

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` (optional — the keyword fallback works without it)
