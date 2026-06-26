"""
main.py — FastAPI Application Entry Point
==========================================
Responsibility: dependency detection, app wiring, and HTTP routes only.
All business logic lives in pipeline.py.
All model registration lives in registry.py.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Dependency detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect(module: str) -> bool:
    import importlib
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


TORCH_AVAILABLE = _detect("torch")
TL_AVAILABLE    = _detect("transformer_lens")
SAE_AVAILABLE   = _detect("sae_lens")
HF_AVAILABLE    = _detect("transformers")

print("\n" + "=" * 60)
print("  SAE Feature Dashboard Backend v2.0")
print(f"  PyTorch        : {'✅' if TORCH_AVAILABLE else '❌'}")
print(f"  TransformerLens: {'✅' if TL_AVAILABLE else '❌'}")
print(f"  SAELens        : {'✅' if SAE_AVAILABLE else '❌'}")
print(f"  HF transformers: {'✅' if HF_AVAILABLE else '❌'}")
print("=" * 60 + "\n")

# ── Inject dependency flags into pipeline ────────────────────────────────────
import pipeline as _pipeline  # noqa: E402
from config import vram_clear  # noqa: E402
from registry import MODEL_REGISTRY  # noqa: E402

_pipeline.set_deps(
    tl=TL_AVAILABLE,
    sae=SAE_AVAILABLE,
    hf=HF_AVAILABLE,
    torch_ok=TORCH_AVAILABLE,
)

FULL_PIPELINE = _pipeline.FULL_PIPELINE

# Pydantic request schema
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    selected_models: List[str] = Field(..., min_items=1, max_items=3)
    top_k: int = Field(default=10, ge=1, le=50)


# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "REAL" if FULL_PIPELINE else "MOCK"
    print(f"[STARTUP] 🚀 Pipeline mode: {mode}")
    print(f"[STARTUP] Registered models: {list(MODEL_REGISTRY.keys())}")
    yield
    print("[SHUTDOWN] Cleaning up …")
    vram_clear()


# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Model SAE Feature Dashboard API",
    description=(
        "Mechanistic interpretability backend. "
        "Model metadata auto-resolved from sae.cfg. "
        "Dual-path activation: TransformerLens → HuggingFace hook fallback."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# 开发模式：允许所有来源
# ⚠️  生产环境请改为具体的 frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "pipeline_mode": "real" if FULL_PIPELINE else "mock",
        "backends": {
            "torch":            TORCH_AVAILABLE,
            "transformer_lens": TL_AVAILABLE,
            "sae_lens":         SAE_AVAILABLE,
            "huggingface":      HF_AVAILABLE,
        },
        "registered_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models", tags=["models"])
async def list_models():
    """
    List all registered models with their resolved config (if cached).
    Config fields populate from cache after the first /analyze call.
    """
    cache = _pipeline._CONFIG_CACHE
    return {
        "models": [
            {
                "key":          key,
                "display_name": reg.get("display_name", key),
                "sae_release":  reg["sae_release"],
                "sae_id":       reg["sae_id"],
                # Populated after first /analyze; "auto" means not yet resolved
                "layer":        cache.get(key, {}).get("layer", "auto"),
                "hook_point":   cache.get(key, {}).get("hook_point", "auto"),
                "d_model":      cache.get(key, {}).get("d_model", "auto"),
                "hf_model_name": cache.get(key, {}).get("hf_model_name", "auto"),
            }
            for key, reg in MODEL_REGISTRY.items()
        ]
    }


@app.post("/analyze", tags=["analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Run SAE feature extraction across the selected models (hot-swapped
    sequentially to stay within VRAM budget).

    Returns Report 1 (global feature summary) + Report 2 (per-token top-50)
    for each model.
    """
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    unknown = [k for k in request.selected_models if k not in MODEL_REGISTRY]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model key(s): {unknown}. Valid: {list(MODEL_REGISTRY.keys())}",
        )

    print(f"\n[API] POST /analyze — models={request.selected_models} "
          f"top_k={request.top_k} prompt='{prompt[:60]}…'")

    models_data: Dict[str, Any] = {}
    for idx, model_key in enumerate(request.selected_models):
        print(f"\n[HOT-SWAP] ⚙️  {idx + 1}/{len(request.selected_models)}: '{model_key}'")
        t0 = time.time()
        models_data[model_key] = _pipeline.analyse_model(prompt, model_key, request.top_k)
        print(f"[HOT-SWAP] ✅ '{model_key}' — {time.time() - t0:.2f}s")
        vram_clear()

    return {
        "metadata": {
            "prompt":          prompt,
            "selected_models": request.selected_models,
            "top_k":           request.top_k,
            "pipeline_mode":   "real" if FULL_PIPELINE else "mock",
        },
        "models_data": models_data,
    }


# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
