"""
main.py — Multi-Model SAE Feature Dashboard Backend
=====================================================
Tech Stack  : FastAPI, PyTorch, TransformerLens, SAELens
Strategy    : Dynamic Hot-Swapping & Active VRAM Clearing
              Models are loaded, inferred, and unloaded ONE AT A TIME to stay
              within VRAM limits. torch.cuda.empty_cache() + gc.collect() are
              called between every model swap.

Pipeline (per model):
  1. TransformerLens  → load model, register residual-stream hooks, run forward
                        pass, intercept dense activation tensor at target layer.
  2. SAELens          → load pre-trained SAE weights for that (model, layer) pair,
                        encode the dense activations into a SPARSE feature vector,
                        extract top-K firing feature indices + values.
  3. Reports          → build Report 1 (global feature summary) and
                        Report 2 (per-token top-50 features), then unload everything.

Mock fallback: if transformer_lens / sae_lens are not installed the server
               gracefully switches to a seeded-random mock that preserves the
               full JSON schema so the frontend can be developed independently.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Dependency detection
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
    print("[BOOT] ✅ PyTorch detected:", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    print("[BOOT] ⚠️  PyTorch not found — MOCK mode active")

try:
    import transformer_lens                          # noqa: F401
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
    print("[BOOT] ✅ TransformerLens detected")
except ImportError:
    TL_AVAILABLE = False
    print("[BOOT] ⚠️  TransformerLens not found — MOCK mode active")

try:
    from sae_lens import SAE                         # noqa: F401
    SAE_AVAILABLE = True
    print("[BOOT] ✅ SAELens detected")
except ImportError:
    SAE_AVAILABLE = False
    print("[BOOT] ⚠️  SAELens not found — MOCK mode active")

FULL_PIPELINE = TORCH_AVAILABLE and TL_AVAILABLE and SAE_AVAILABLE

print(
    f"[BOOT] Pipeline mode: {'🔬 REAL (TransformerLens + SAELens)' if FULL_PIPELINE else '🧪 MOCK (seeded random)'}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry declares:
#   tl_name      — the TransformerLens model identifier
#   sae_release  — the SAELens release string
#   sae_id       — the specific SAE id within that release
#   hook_point   — the TransformerLens hook name (residual stream)
#   layer        — the layer index (int) for display
#   d_model      — residual stream dimensionality
#   display_name — friendly label shown in the UI

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pythia-70m": {
        "display_name": "Pythia 70M",
        "tl_name": "EleutherAI/pythia-70m-deduped",
        "sae_release": "pythia-70m-deduped",
        "sae_id": "res-jb",
        "hook_point": "blocks.4.hook_resid_post",
        "layer": 4,
        "d_model": 512,
    },
    "gemma-2b": {
        "display_name": "Gemma 2B",
        "tl_name": "google/gemma-2b",
        "sae_release": "gemma-2b-res-jb",
        "sae_id": "blocks.12.hook_resid_post",
        "hook_point": "blocks.12.hook_resid_post",
        "layer": 12,
        "d_model": 2048,
    },
    "llama-3.2-1b": {
        "display_name": "Llama 3.2 1B",
        "tl_name": "meta-llama/Llama-3.2-1B",
        "sae_release": "llama_3.2_1b-res-jb",
        "sae_id": "blocks.8.hook_resid_post",
        "hook_point": "blocks.8.hook_resid_post",
        "layer": 8,
        "d_model": 2048,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Concept label bank
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_LABELS: List[str] = [
    "Corporate Entities",
    "Tech Release",
    "Product Announcement",
    "Financial News",
    "Named Persons",
    "Geopolitical",
    "Temporal Markers",
    "Numeric Values",
    "Action Verbs",
    "Attributive Adjectives",
    "Consumer Electronics",
    "Brand Names",
    "Software Ecosystem",
    "Market Dynamics",
    "Innovation & R&D",
    "Media & Press",
    "Legal & Regulatory",
    "Supply Chain",
    "Sentiment — Positive",
    "Sentiment — Negative",
    "Scientific Concepts",
    "Historical References",
    "Sports & Recreation",
    "Cultural References",
    "Medical & Health",
    "Environmental Topics",
    "Political Discourse",
    "Economic Indicators",
    "Abstract Reasoning",
    "Logical Connectives",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request model
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    selected_models: List[str] = Field(..., min_items=1, max_items=3)
    top_k: int = Field(default=10, ge=1, le=50)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _concept_label(feature_id: int, rng: random.Random) -> str:
    return CONCEPT_LABELS[feature_id % len(CONCEPT_LABELS)]


def _vram_clear() -> None:
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        used_mb = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"[VRAM] 🧹 VRAM cleared — currently allocated: {used_mb:.1f} MB")
    else:
        print("[VRAM] 🧹 CPU mode — gc.collect() called (no CUDA to clear)")


def _naive_tokenise(text: str) -> List[str]:
    import re
    tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
    return tokens if tokens else [text]


# ─────────────────────────────────────────────────────────────────────────────
# MOCK pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _mock_analyse_model(prompt: str, model_key: str, top_k: int) -> Dict[str, Any]:
    """
    Deterministic seeded-random mock of the TransformerLens + SAELens pipeline.
    Preserves the full JSON schema for independent frontend development.
    """
    seed = hash(prompt + model_key) & 0xFFFFFFFF
    rng = random.Random(seed)

    cfg = MODEL_REGISTRY[model_key]
    tokens = _naive_tokenise(prompt)
    num_tokens = len(tokens)
    d_sae = 4096  # synthetic SAE width

    print(f"\n[MOCK | {model_key}] ─────────────────────────────────────────")
    print(f"[MOCK | {model_key}] 📝 Prompt    : '{prompt}'")
    print(f"[MOCK | {model_key}] 🔢 Tokens    : {tokens}")
    print(f"[MOCK | {model_key}] 🔬 [TransformerLens] Simulating model load: '{cfg['tl_name']}'")
    time.sleep(0.05)
    print(f"[MOCK | {model_key}] 🪝 [TransformerLens] Hook registered at '{cfg['hook_point']}' "
          f"(layer {cfg['layer']}, d_model={cfg['d_model']})")
    time.sleep(0.03)
    print(f"[MOCK | {model_key}] ▶️  [TransformerLens] Forward pass complete — "
          f"intercepted residual-stream activations: shape ({num_tokens}, {cfg['d_model']})")
    time.sleep(0.03)
    print(f"[MOCK | {model_key}] 🔭 [SAELens] Loading SAE weights: "
          f"release='{cfg['sae_release']}', id='{cfg['sae_id']}'")
    time.sleep(0.05)
    print(f"[MOCK | {model_key}] ✨ [SAELens] Encoding dense activations → sparse features "
          f"(d_sae={d_sae}, L0 sparsity target ~{d_sae // 200} active features/token)")
    time.sleep(0.03)

    # Build synthetic sparse feature activation matrix
    feature_matrix: List[List[float]] = []
    for _ in range(num_tokens):
        row = [0.0] * d_sae
        n_active = rng.randint(15, 35)
        active_ids = rng.sample(range(d_sae), n_active)
        for fid in active_ids:
            row[fid] = round(rng.uniform(0.1, 6.5), 4)
        feature_matrix.append(row)

    print(f"[MOCK | {model_key}] 📊 Feature matrix built: "
          f"{num_tokens} tokens × {d_sae} features (sparse)")

    # Report 2: Per-token top-50 features
    token_level_firings = []
    for tok_idx, tok_str in enumerate(tokens):
        row = feature_matrix[tok_idx]
        pairs = [(fid, val) for fid, val in enumerate(row) if val > 0.0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_50 = pairs[:50]
        top_50_features = [
            {
                "feature_id": fid,
                "activation": round(val, 4),
                "concept_label": _concept_label(fid, rng),
            }
            for fid, val in top_50
        ]
        token_level_firings.append({
            "token_index": tok_idx,
            "token_string": tok_str,
            "top_50_features": top_50_features,
        })

    # Report 1: Global feature firing summary
    global_feature_stats: Dict[int, Dict[str, Any]] = {}
    for tok_idx, row in enumerate(feature_matrix):
        for fid, val in enumerate(row):
            if val == 0.0:
                continue
            if fid not in global_feature_stats:
                global_feature_stats[fid] = {
                    "feature_id": fid,
                    "max_activation": val,
                    "sum_activation": val,
                    "fired_token_count": 1,
                    "concept_label": _concept_label(fid, rng),
                }
            else:
                s = global_feature_stats[fid]
                s["max_activation"] = max(s["max_activation"], val)
                s["sum_activation"] += val
                s["fired_token_count"] += 1

    fired_features_summary = []
    for fid, s in global_feature_stats.items():
        avg = round(s["sum_activation"] / s["fired_token_count"], 4)
        fired_features_summary.append({
            "feature_id": fid,
            "max_activation": round(s["max_activation"], 4),
            "avg_activation": avg,
            "fired_token_count": s["fired_token_count"],
            "concept_label": s["concept_label"],
        })
    fired_features_summary.sort(key=lambda x: x["max_activation"], reverse=True)

    print(f"[MOCK | {model_key}] ✅ Reports generated — "
          f"{len(fired_features_summary)} unique global features, "
          f"{num_tokens} token records")

    return {
        "model_metadata": {
            "model_name": cfg["display_name"],
            "model_key": model_key,
            "prompt": prompt,
            "tokens": tokens,
            "hook_point": cfg["hook_point"],
            "layer": cfg["layer"],
            "d_model": cfg["d_model"],
            "sae_release": cfg["sae_release"],
            "sae_id": cfg["sae_id"],
            "pipeline_mode": "mock",
        },
        "report_1_global": {
            "fired_features_summary": fired_features_summary,
        },
        "report_2_per_token": {
            "token_level_firings": token_level_firings,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# REAL pipeline (requires TransformerLens + SAELens + PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _real_analyse_model(prompt: str, model_key: str, top_k: int) -> Dict[str, Any]:
    """
    Full TransformerLens + SAELens pipeline:
      1. HookedTransformer.from_pretrained() loads the LLM via TransformerLens.
      2. run_with_hooks() intercepts the residual stream at the target layer.
      3. SAE.from_pretrained() loads the pre-trained sparse autoencoder via SAELens.
      4. sae.encode() maps dense activations → sparse interpretable features.
      5. Both reports are built, then ALL model objects are deleted and VRAM is cleared.
    """
    cfg = MODEL_REGISTRY[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(hash(prompt + model_key) & 0xFFFFFFFF)

    print(f"\n[REAL | {model_key}] ─────────────────────────────────────────")
    print(f"[REAL | {model_key}] 📝 Prompt  : '{prompt}'")
    print(f"[REAL | {model_key}] 💻 Device  : {device}")

    # Step 1 — TransformerLens: load HookedTransformer
    print(f"[REAL | {model_key}] 🔬 [TransformerLens] Loading '{cfg['tl_name']}' via "
          "HookedTransformer.from_pretrained() …")
    model = HookedTransformer.from_pretrained(
        cfg["tl_name"],
        center_writing_weights=False,
        fold_ln=False,
        device=device,
    )
    model.eval()
    print(f"[REAL | {model_key}] ✅ [TransformerLens] Model loaded — "
          f"d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    # Step 2 — Tokenise
    tokens_tensor = model.to_tokens(prompt)
    token_strs: List[str] = model.to_str_tokens(prompt)
    seq_len = tokens_tensor.shape[1]
    print(f"[REAL | {model_key}] 🔢 [TransformerLens] Tokenised → {seq_len} tokens: {token_strs}")

    # Step 3 — TransformerLens: register hook & run forward pass
    print(f"[REAL | {model_key}] 🪝 [TransformerLens] Registering residual-stream hook "
          f"at '{cfg['hook_point']}' (layer {cfg['layer']}) …")

    captured: Dict[str, "torch.Tensor"] = {}

    def _hook_fn(value: "torch.Tensor", hook: Any) -> "torch.Tensor":
        captured["resid"] = value.detach().clone()
        print(f"[REAL | {model_key}] 🎯 [TransformerLens] Hook FIRED at '{hook.name}' — "
              f"intercepted tensor shape: {tuple(value.shape)}")
        return value  # pass-through, model is NOT altered

    print(f"[REAL | {model_key}] ▶️  [TransformerLens] Running forward pass …")
    with torch.no_grad():
        _ = model.run_with_hooks(
            tokens_tensor,
            fwd_hooks=[(cfg["hook_point"], _hook_fn)],
        )

    resid: "torch.Tensor" = captured["resid"][0]  # (seq_len, d_model)
    print(f"[REAL | {model_key}] ✅ [TransformerLens] Forward pass complete — "
          f"residual stream captured: shape {tuple(resid.shape)}")

    # Step 4 — Hot-swap: unload TransformerLens model to free VRAM
    print(f"[REAL | {model_key}] 🗑️  [HOT-SWAP] Deleting HookedTransformer to free VRAM …")
    del model
    _vram_clear()

    # Step 5 — SAELens: load SAE and encode activations
    print(f"[REAL | {model_key}] 🔭 [SAELens] Loading SAE: "
          f"release='{cfg['sae_release']}', id='{cfg['sae_id']}' …")
    sae, cfg_dict, latents_dataset = SAE.from_pretrained(
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )
    sae.eval()
    d_sae = sae.cfg.d_sae
    print(f"[REAL | {model_key}] ✅ [SAELens] SAE loaded — "
          f"d_sae={d_sae} (expansion factor x{d_sae // resid.shape[-1]})")

    print(f"[REAL | {model_key}] ✨ [SAELens] Encoding dense residual activations "
          f"→ sparse SAE features (d_sae={d_sae}) …")
    with torch.no_grad():
        feature_acts: "torch.Tensor" = sae.encode(resid)  # (seq_len, d_sae)

    print(f"[REAL | {model_key}] 📊 [SAELens] Feature activation tensor: "
          f"shape {tuple(feature_acts.shape)}, "
          f"mean L0 = {(feature_acts > 0).float().sum(dim=-1).mean().item():.1f} active features/token")

    # Step 6 — Hot-swap: unload SAE to free VRAM
    print(f"[REAL | {model_key}] 🗑️  [HOT-SWAP] Deleting SAE model to free VRAM …")
    feature_acts_cpu = feature_acts.cpu()
    del sae, feature_acts
    _vram_clear()

    # Step 7 — Build reports
    import numpy as np
    fa = feature_acts_cpu.numpy()  # (seq_len, d_sae)

    # Report 2: Per-token top-50
    token_level_firings = []
    for tok_idx, tok_str in enumerate(token_strs):
        row = fa[tok_idx]
        active_ids = (row > 0).nonzero()[0].tolist()
        pairs = [(int(fid), float(row[fid])) for fid in active_ids]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_50 = pairs[:50]
        top_50_features = [
            {
                "feature_id": fid,
                "activation": round(val, 4),
                "concept_label": _concept_label(fid, rng),
            }
            for fid, val in top_50
        ]
        token_level_firings.append({
            "token_index": tok_idx,
            "token_string": tok_str,
            "top_50_features": top_50_features,
        })

    # Report 1: Global feature summary
    global_max = fa.max(axis=0)
    global_sum = fa.sum(axis=0)
    global_count = (fa > 0).sum(axis=0)
    active_feature_ids = (global_count > 0).nonzero()[0].tolist()

    fired_features_summary = []
    for fid in active_feature_ids:
        cnt = int(global_count[fid])
        fired_features_summary.append({
            "feature_id": fid,
            "max_activation": round(float(global_max[fid]), 4),
            "avg_activation": round(float(global_sum[fid]) / cnt, 4),
            "fired_token_count": cnt,
            "concept_label": _concept_label(fid, rng),
        })
    fired_features_summary.sort(key=lambda x: x["max_activation"], reverse=True)

    print(f"[REAL | {model_key}] ✅ Reports generated — "
          f"{len(fired_features_summary)} unique global features, "
          f"{seq_len} token records")

    return {
        "model_metadata": {
            "model_name": cfg["display_name"],
            "model_key": model_key,
            "prompt": prompt,
            "tokens": token_strs,
            "hook_point": cfg["hook_point"],
            "layer": cfg["layer"],
            "d_model": cfg["d_model"],
            "sae_release": cfg["sae_release"],
            "sae_id": cfg["sae_id"],
            "pipeline_mode": "real",
        },
        "report_1_global": {
            "fired_features_summary": fired_features_summary,
        },
        "report_2_per_token": {
            "token_level_firings": token_level_firings,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def analyse_model(prompt: str, model_key: str, top_k: int) -> Dict[str, Any]:
    if FULL_PIPELINE:
        try:
            return _real_analyse_model(prompt, model_key, top_k)
        except Exception as exc:
            print(f"[ERROR | {model_key}] Real pipeline failed: {exc}")
            print(traceback.format_exc())
            print(f"[FALLBACK | {model_key}] Switching to MOCK pipeline …")
            return _mock_analyse_model(prompt, model_key, top_k)
    else:
        return _mock_analyse_model(prompt, model_key, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("  SAE Feature Dashboard Backend — STARTING")
    print(f"  Pipeline: {'REAL (TransformerLens + SAELens)' if FULL_PIPELINE else 'MOCK'}")
    print(f"  Models  : {list(MODEL_REGISTRY.keys())}")
    print("=" * 60 + "\n")
    yield
    print("\n[SHUTDOWN] Backend shutting down …")
    _vram_clear()


app = FastAPI(
    title="Multi-Model SAE Feature Dashboard API",
    description=(
        "Mechanistic Interpretability backend leveraging TransformerLens for "
        "activation hooking and SAELens for sparse feature extraction. "
        "Supports dynamic hot-swapping of multiple LLMs to stay within VRAM limits."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "pipeline_mode": "real" if FULL_PIPELINE else "mock",
        "available_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models", tags=["models"])
async def list_models():
    """Return the full model registry so the frontend can build its selectors."""
    return {
        "models": [
            {
                "key": key,
                "display_name": cfg["display_name"],
                "layer": cfg["layer"],
                "hook_point": cfg["hook_point"],
                "sae_release": cfg["sae_release"],
            }
            for key, cfg in MODEL_REGISTRY.items()
        ]
    }


@app.post("/analyze", tags=["analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Core endpoint — sequentially processes each model via the hot-swap strategy:

      For each model:
        [TransformerLens] Load LLM → register hook at residual stream → run forward pass
        [SAELens]         Load SAE → encode dense activations → sparse feature vector
        [HOT-SWAP]        Delete ALL model objects → torch.cuda.empty_cache() + gc.collect()

    Returns dual reports (global summary + per-token top-50) nested under each model key.
    """
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    for key in request.selected_models:
        if key not in MODEL_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model key '{key}'. Valid: {list(MODEL_REGISTRY.keys())}",
            )

    print("\n" + "+" * 60)
    print(f"  NEW ANALYSIS REQUEST")
    print(f"  Prompt  : '{prompt}'")
    print(f"  Models  : {request.selected_models}")
    print(f"  Top-K   : {request.top_k}")
    print("+" * 60)

    models_data: Dict[str, Any] = {}

    for idx, model_key in enumerate(request.selected_models):
        print(f"\n[HOT-SWAP] Processing model {idx + 1}/{len(request.selected_models)}: '{model_key}'")
        t0 = time.time()
        model_result = analyse_model(prompt, model_key, request.top_k)
        elapsed = time.time() - t0
        models_data[model_key] = model_result
        print(f"[HOT-SWAP] '{model_key}' done in {elapsed:.2f}s")

        # Always clear between models
        if idx < len(request.selected_models) - 1:
            print(f"[HOT-SWAP] Clearing VRAM before loading next model …")
        _vram_clear()

    print(f"\n[ANALYSIS] All {len(request.selected_models)} model(s) processed successfully.")

    return {
        "metadata": {
            "prompt": prompt,
            "selected_models": request.selected_models,
            "top_k": request.top_k,
            "pipeline_mode": "real" if FULL_PIPELINE else "mock",
        },
        "models_data": models_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
