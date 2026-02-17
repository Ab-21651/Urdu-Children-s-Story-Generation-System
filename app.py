"""
app.py
======
FastAPI microservice for Urdu story generation.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then visit:
    http://localhost:8000/docs   ← interactive Swagger UI
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tokenizer_utils import BPETokenizer, TrigramLM, load_model_from_json

# ─── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "trigram_model.json"
MERGES_PATH = BASE_DIR / "tokenizer" / "bpe_merges.txt"

# ─── Load model & tokenizer at startup ────────────────────────────
print("Loading trigram model …")
model: TrigramLM = load_model_from_json(MODEL_PATH)
print(f"  ✅ Model loaded — vocab size: {len(model.vocab)}")

print("Loading BPE tokenizer …")
tokenizer: BPETokenizer = BPETokenizer(MERGES_PATH)
print(f"  ✅ Tokenizer loaded — {len(tokenizer.merges)} merge rules")

# ─── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Urdu Story Generator API",
    description="Generate Urdu children's stories using a Trigram Language Model + BPE tokenizer.",
    version="1.0.0",
)

# Allow frontend (Phase V) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────

class GenerateRequest(BaseModel):
    prefix: str = Field(
        default="",
        description="Optional Urdu text to start the story with (e.g. 'ایک دن').",
    )
    max_length: int = Field(
        default=300,
        ge=1,
        le=1000,
        description="Maximum number of BPE tokens to generate.",
    )
    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic).",
    )


class GenerateResponse(BaseModel):
    generated_text: str
    num_tokens: int
    seed_tokens: list[str]


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health-check / info endpoint."""
    return {
        "service": "Urdu Story Generator",
        "status": "running",
        "vocab_size": len(model.vocab),
        "merge_rules": len(tokenizer.merges),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_story(req: GenerateRequest):
    """
    Generate an Urdu children's story.

    - If `prefix` is provided, it is BPE-tokenized and used as seed.
    - The model generates token-by-token until EOT or `max_length`.
    - Returns the decoded Urdu text.
    """
    try:
        # Tokenize the prefix into BPE seed tokens
        seed_tokens = tokenizer.encode_text(req.prefix) if req.prefix.strip() else None

        # Generate
        generated_tokens = model.generate(
            max_tokens=req.max_length,
            temperature=req.temperature,
            seed_tokens=seed_tokens,
        )

        # Decode back to text
        text = TrigramLM.tokens_to_text(generated_tokens, eot_token=model.eot_token)

        return GenerateResponse(
            generated_text=text,
            num_tokens=len(generated_tokens),
            seed_tokens=seed_tokens or [],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
