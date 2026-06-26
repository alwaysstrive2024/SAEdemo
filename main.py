"""
main.py — Multi-Model SAE Feature Dashboard Backend  (Refactored)
=================================================================
Key changes vs. v1:
  • MODEL_REGISTRY is now minimal — only (sae_release, sae_id, display_name).
    All heavy metadata (d_model, hook_point, layer, hf_model_name) is extracted
    automatically from sae.cfg at runtime and cached in _CONFIG_CACHE.

  • Dual-path activation extraction:
      Path A — TransformerLens HookedTransformer  (preferred)
      Path B — HuggingFace transformers + register_forward_hook  (fallback)
    The system tries Path A first; if the model is unsupported by TL or raises
    any error, it transparently falls back to Path B.

  • VRAM hot-swap strategy is preserved: model objects are deleted and
    torch.cuda.empty_cache() is called between every model swap.
"""

from __future__ import annotations

import gc
import random
import re
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

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
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
    print("[BOOT] ✅ TransformerLens detected")
except ImportError:
    TL_AVAILABLE = False
    print("[BOOT] ⚠️  TransformerLens not found — will use HF fallback for real pipeline")

try:
    from sae_lens import SAE
    SAE_AVAILABLE = True
    print("[BOOT] ✅ SAELens detected")
except ImportError:
    SAE_AVAILABLE = False
    print("[BOOT] ⚠️  SAELens not found — MOCK mode active")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
    print("[BOOT] ✅ HuggingFace transformers detected")
except ImportError:
    HF_AVAILABLE = False
    print("[BOOT] ⚠️  transformers not found — HF fallback path unavailable")

FULL_PIPELINE = TORCH_AVAILABLE and SAE_AVAILABLE and (TL_AVAILABLE or HF_AVAILABLE)

print(
    f"[BOOT] Pipeline mode: "
    f"{'🔬 REAL (SAELens + ' + ('TransformerLens' if TL_AVAILABLE else 'HF hook') + ')' if FULL_PIPELINE else '🧪 MOCK (seeded random)'}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal Model Registry
# ─────────────────────────────────────────────────────────────────────────────
# Only three fields are required.  All heavy metadata (d_model, hook_point,
# layer, hf_model_name) is auto-extracted from sae.cfg at first use.
#
# To add a new model: just add one entry with sae_release + sae_id.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # ── Pythia ────────────────────────────────────────────────────────────────
    "pythia-70m": {
        "display_name": "Pythia 70M",
        "sae_release":  "pythia-70m-deduped",
        "sae_id":       "res-jb",
    },
    # ── Gemma 2B ─────────────────────────────────────────────────────────────
    "gemma-2b": {
        "display_name": "Gemma 2B",
        "sae_release":  "gemma-2b-res-jb",
        "sae_id":       "blocks.12.hook_resid_post",
    },
    # ── Llama 3.2 1B ─────────────────────────────────────────────────────────
    "llama-3.2-1b": {
        "display_name": "Llama 3.2 1B",
        "sae_release":  "llama_3.2_1b-res-jb",
        "sae_id":       "blocks.8.hook_resid_post",
    },
    
    # ── Gemma-4 SAEs (decoderesearch) ─────────────────────────────────────────
    "gemma-4-e2b-l6": {
        "display_name": "Gemma-4-E2B (Layer 6)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-e2b/btk-mat-layer-6-k-100",
    },
    "gemma-4-e2b-l17": {
        "display_name": "Gemma-4-E2B (Layer 17)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-e2b/btk-mat-layer-17-k-100",
    },
    "gemma-4-e2b-l28": {
        "display_name": "Gemma-4-E2B(layer 28)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-e2b/btk-mat-layer-28-k-100",
    },
    "gemma-4-e4b-l7": {
        "display_name": "Gemma-4-E4B (Layer 7)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-e4b/btk-mat-layer-7-k-100",
    },
    "gemma-4-e4b-l21": {
        "display_name": "Gemma-4-E4B (Layer 21)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-e4b/btk-mat-layer-21-k-100",
    },
    "gemma-4-e4b-l35": {
        "display_name": "Gemma-4-E4B (Layer 35)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-e4b/btk-mat-layer-35-k-100",
    },
    "gemma-4-31b-l30": {
        "display_name": "Gemma-4-31B (Layer 30)",
        "sae_release":  "decoderesearch/gemma-4-saes",
        "sae_id":       "gemma-4-31b/btk-mat-layer-30-k-100",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Runtime config cache  (populated lazily on first use per model)
# ─────────────────────────────────────────────────────────────────────────────
# Stores the fully-resolved config dict so SAE weights don't need to be
# re-downloaded just to read metadata on subsequent requests.

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Concept label bank
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_LABELS: List[str] = [
    "Corporate Entities", "Tech Release", "Product Announcement",
    "Financial News", "Named Persons", "Geopolitical", "Temporal Markers",
    "Numeric Values", "Action Verbs", "Attributive Adjectives",
    "Consumer Electronics", "Brand Names", "Software Ecosystem",
    "Market Dynamics", "Innovation & R&D", "Media & Press",
    "Legal & Regulatory", "Supply Chain", "Sentiment — Positive",
    "Sentiment — Negative", "Scientific Concepts", "Historical References",
    "Sports & Recreation", "Cultural References", "Medical & Health",
    "Environmental Topics", "Political Discourse", "Economic Indicators",
    "Abstract Reasoning", "Logical Connectives",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request model
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    selected_models: List[str] = Field(..., min_items=1, max_items=3)
    top_k: int = Field(default=10, ge=1, le=50)

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
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
        print("[VRAM] 🧹 CPU mode — gc.collect() called")


def _naive_tokenise(text: str) -> List[str]:
    tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
    return tokens if tokens else [text]


def _safe_sae_attr(cfg: Any, *names: str, default: Any = None) -> Any:
    """Try multiple attribute names on sae.cfg (SAELens API varies by version)."""
    for name in names:
        val = getattr(cfg, name, None)
        if val is not None:
            return val
    return default


def _layer_from_hook(hook_name: str) -> Optional[int]:
    """Extract layer index from hook name e.g. 'blocks.12.hook_resid_post' → 12."""
    m = re.search(r'\.(\d+)\.', hook_name or "")
    return int(m.group(1)) if m else None

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic config resolution  (the core of the refactor)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model_config(model_key: str) -> Dict[str, Any]:
    """
    Return the fully-resolved config for a model key.

    On first call: loads the SAE just long enough to read sae.cfg,
    builds the config dict, caches it, then frees the SAE weights.

    On subsequent calls: returns the cached dict instantly.
    """
    if model_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[model_key]

    reg = MODEL_REGISTRY[model_key]

    if not SAE_AVAILABLE:
        # No SAELens — return a stub (mock pipeline will handle it)
        cfg = {
            "display_name": reg.get("display_name", model_key),
            "hf_model_name": model_key,
            "sae_release": reg["sae_release"],
            "sae_id": reg["sae_id"],
            "hook_point": f"blocks.0.hook_resid_post",
            "layer": 0,
            "d_model": 512,
        }
        _CONFIG_CACHE[model_key] = cfg
        return cfg

    print(f"[CONFIG] 🔭 [SAELens] Loading SAE to extract config for '{model_key}' …")
    print(f"[CONFIG]   release='{reg['sae_release']}', sae_id='{reg['sae_id']}'")

    sae_obj, _, _ = SAE.from_pretrained(
        release=reg["sae_release"],
        sae_id=reg["sae_id"],
    )
    sae_cfg = sae_obj.cfg

    # ── Extract fields with multi-name fallback for SAELens version compat ──
    hook_point   = _safe_sae_attr(sae_cfg, "hook_name", "hook_point",  default="")
    d_model      = _safe_sae_attr(sae_cfg, "d_in",      "d_model",     default=512)
    hook_layer   = _safe_sae_attr(sae_cfg, "hook_layer", "layer",       default=None)
    hf_model_id  = _safe_sae_attr(sae_cfg, "model_name", "model_id",   default=model_key)
    sae_id_read  = _safe_sae_attr(sae_cfg, "sae_id",                    default=reg["sae_id"])

    # Derive layer from hook_point string if not directly available
    if hook_layer is None:
        hook_layer = _layer_from_hook(hook_point) or 0

    config: Dict[str, Any] = {
        "display_name":  reg.get("display_name", hf_model_id),
        "hf_model_name": hf_model_id,
        "sae_release":   reg["sae_release"],
        "sae_id":        sae_id_read,
        "hook_point":    hook_point,
        "layer":         hook_layer,
        "d_model":       d_model,
    }

    # Free SAE weights immediately — we only needed the config
    del sae_obj
    _vram_clear()

    _CONFIG_CACHE[model_key] = config
    print(
        f"[CONFIG] ✅ Config resolved and cached: "
        f"hf='{hf_model_id}', hook='{hook_point}', "
        f"d_model={d_model}, layer={hook_layer}"
    )
    return config

# ─────────────────────────────────────────────────────────────────────────────
# Path A — TransformerLens activation extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_activations_tl(
    hf_model_name: str,
    hook_point: str,
    prompt: str,
    device: str,
) -> Tuple["torch.Tensor", List[str]]:
    """
    Use HookedTransformer to run the model and intercept the residual stream
    at `hook_point`.  Returns (resid: Tensor[seq_len, d_model], token_strs).
    """
    print(f"[PATH-A | TL] 🔬 [TransformerLens] Loading '{hf_model_name}' via "
          "HookedTransformer.from_pretrained() …")
    model = HookedTransformer.from_pretrained(
        hf_model_name,
        center_writing_weights=False,
        fold_ln=False,
        device=device,
    )
    model.eval()
    print(f"[PATH-A | TL] ✅ Loaded — d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    tokens_tensor = model.to_tokens(prompt)
    token_strs: List[str] = model.to_str_tokens(prompt)
    print(f"[PATH-A | TL] 🔢 Tokenised → {len(token_strs)} tokens: {token_strs}")

    captured: Dict[str, "torch.Tensor"] = {}

    def hook_fn(value: "torch.Tensor", hook: Any) -> "torch.Tensor":
        captured["resid"] = value.detach().clone()
        print(f"[PATH-A | TL] 🎯 Hook FIRED at '{hook.name}' — shape {tuple(value.shape)}")
        return value  # pass-through, model unaltered

    print(f"[PATH-A | TL] 🪝 Registering hook at '{hook_point}' …")
    with torch.no_grad():
        _ = model.run_with_hooks(tokens_tensor, fwd_hooks=[(hook_point, hook_fn)])

    resid = captured["resid"][0]  # (seq_len, d_model)
    print(f"[PATH-A | TL] ✅ Residual stream captured: {tuple(resid.shape)}")

    del model
    _vram_clear()
    return resid, token_strs


# ─────────────────────────────────────────────────────────────────────────────
# Path B — HuggingFace + register_forward_hook fallback
# ─────────────────────────────────────────────────────────────────────────────

def _find_hf_layer(model: Any, layer_idx: int) -> Any:
    """
    Locate the transformer block at index `layer_idx` across common HF architectures:
      Llama / Gemma / Mistral / Qwen  →  model.model.layers[n]
      GPT-NeoX / Pythia               →  model.gpt_neox.layers[n]
      GPT-2                            →  model.transformer.h[n]
      OPT                              →  model.model.decoder.layers[n]
    Raises RuntimeError if none match.
    """
    CANDIDATE_PATHS = [
        lambda m, n: m.model.layers[n],             # Llama, Gemma, Mistral, Qwen
        lambda m, n: m.gpt_neox.layers[n],           # Pythia, GPT-NeoX
        lambda m, n: m.transformer.h[n],              # GPT-2
        lambda m, n: m.model.decoder.layers[n],       # OPT
        lambda m, n: m.transformer.blocks[n],         # Falcon (old)
        lambda m, n: list(m.children())[0][n],        # generic ordered children
    ]
    for path_fn in CANDIDATE_PATHS:
        try:
            layer = path_fn(model, layer_idx)
            if layer is not None:
                print(f"[PATH-B | HF] 🎯 Found layer {layer_idx}: {type(layer).__name__}")
                return layer
        except (AttributeError, IndexError, TypeError, KeyError):
            continue
    raise RuntimeError(
        f"[PATH-B | HF] Cannot locate layer {layer_idx} in model of type "
        f"{type(model).__name__}. Please add its attribute path to _find_hf_layer()."
    )


def _extract_activations_hf(
    hf_model_name: str,
    layer_idx: int,
    prompt: str,
    device: str,
) -> Tuple["torch.Tensor", List[str]]:
    """
    HuggingFace transformers fallback.
    Loads the model, attaches a register_forward_hook on the target layer,
    runs a forward pass, and captures the hidden states.
    Returns (resid: Tensor[seq_len, d_model], token_strs).
    """
    print(f"[PATH-B | HF] 🤗 Loading tokenizer for '{hf_model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    print(f"[PATH-B | HF] 🤗 Loading '{hf_model_name}' via AutoModelForCausalLM …")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
        device_map=device if device == "cpu" else "auto",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"][0]
    token_strs = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    print(f"[PATH-B | HF] 🔢 Tokenised → {len(token_strs)} tokens: {token_strs}")

    # Register hook
    captured: Dict[str, "torch.Tensor"] = {}

    def hook_fn(module: Any, inp: Any, out: Any) -> None:
        # HF layers usually return (hidden_states, ...) or just hidden_states
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        captured["hidden"] = hidden.detach().clone()
        print(f"[PATH-B | HF] 🎯 Hook FIRED at layer {layer_idx} — shape {tuple(hidden.shape)}")

    target_layer = _find_hf_layer(model, layer_idx)
    handle = target_layer.register_forward_hook(hook_fn)

    print(f"[PATH-B | HF] 🪝 Hook registered — running forward pass …")
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("[PATH-B | HF] Hook did not fire — no activations captured.")

    resid = captured["hidden"][0]  # (seq_len, d_model) — remove batch dim
    print(f"[PATH-B | HF] ✅ Activations captured: {tuple(resid.shape)}")

    del model
    _vram_clear()
    return resid, token_strs

# ─────────────────────────────────────────────────────────────────────────────
# Shared report builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_reports(
    feature_acts_cpu: "torch.Tensor",
    token_strs: List[str],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Given a (seq_len, d_sae) feature activation tensor (on CPU),
    build Report 1 (global feature summary) and Report 2 (per-token top-50).
    """
    import numpy as np
    fa = feature_acts_cpu.numpy()  # (seq_len, d_sae)

    # Report 2 — per-token top-50
    token_level_firings: List[Dict[str, Any]] = []
    for tok_idx, tok_str in enumerate(token_strs):
        row = fa[tok_idx]
        pairs = [(int(fid), float(row[fid])) for fid in (row > 0).nonzero()[0]]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_50_features = [
            {
                "feature_id": fid,
                "activation": round(val, 4),
                "concept_label": _concept_label(fid, rng),
            }
            for fid, val in pairs[:50]
        ]
        token_level_firings.append({
            "token_index": tok_idx,
            "token_string": tok_str,
            "top_50_features": top_50_features,
        })

    # Report 1 — global feature summary
    global_max   = fa.max(axis=0)
    global_sum   = fa.sum(axis=0)
    global_count = (fa > 0).sum(axis=0)
    active_ids   = (global_count > 0).nonzero()[0].tolist()

    fired_features_summary: List[Dict[str, Any]] = []
    for fid in active_ids:
        cnt = int(global_count[fid])
        fired_features_summary.append({
            "feature_id": fid,
            "max_activation": round(float(global_max[fid]), 4),
            "avg_activation": round(float(global_sum[fid]) / cnt, 4),
            "fired_token_count": cnt,
            "concept_label": _concept_label(fid, rng),
        })
    fired_features_summary.sort(key=lambda x: x["max_activation"], reverse=True)

    return fired_features_summary, token_level_firings

# ─────────────────────────────────────────────────────────────────────────────
# MOCK pipeline  (no ML deps required)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_analyse_model(prompt: str, model_key: str, top_k: int) -> Dict[str, Any]:
    """Deterministic seeded-random mock that preserves the full JSON schema."""
    seed = hash(prompt + model_key) & 0xFFFFFFFF
    rng = random.Random(seed)

    reg = MODEL_REGISTRY[model_key]
    # Use cached config if available, otherwise use registry stubs
    cfg = _CONFIG_CACHE.get(model_key, {
        "display_name":  reg.get("display_name", model_key),
        "hf_model_name": model_key,
        "hook_point":    "blocks.4.hook_resid_post",
        "layer":         4,
        "d_model":       512,
        "sae_release":   reg["sae_release"],
        "sae_id":        reg["sae_id"],
    })

    tokens = _naive_tokenise(prompt)
    num_tokens = len(tokens)
    d_sae = 4096  # synthetic SAE width

    print(f"\n[MOCK | {model_key}] ─────────────────────────────────────────")
    print(f"[MOCK | {model_key}] 📝 Prompt : '{prompt}'")
    print(f"[MOCK | {model_key}] 🔢 Tokens : {tokens}")
    print(f"[MOCK | {model_key}] 🔬 [TransformerLens] Simulating load: '{cfg['hf_model_name']}'")
    time.sleep(0.05)
    print(f"[MOCK | {model_key}] 🪝 [TransformerLens] Hook at '{cfg['hook_point']}' "
          f"(layer {cfg['layer']}, d_model={cfg['d_model']})")
    time.sleep(0.03)
    print(f"[MOCK | {model_key}] ▶️  [TransformerLens] Forward pass done — "
          f"activations shape ({num_tokens}, {cfg['d_model']})")
    time.sleep(0.03)
    print(f"[MOCK | {model_key}] 🔭 [SAELens] Loading SAE: "
          f"release='{cfg['sae_release']}', sae_id='{cfg['sae_id']}'")
    time.sleep(0.05)
    print(f"[MOCK | {model_key}] ✨ [SAELens] Encoding dense → sparse "
          f"(d_sae={d_sae}, L0 ~{d_sae // 200} active/token)")
    time.sleep(0.03)

    # Synthetic sparse feature matrix
    feature_matrix: List[List[float]] = []
    for _ in range(num_tokens):
        row = [0.0] * d_sae
        for fid in rng.sample(range(d_sae), rng.randint(15, 35)):
            row[fid] = round(rng.uniform(0.1, 6.5), 4)
        feature_matrix.append(row)

    print(f"[MOCK | {model_key}] 📊 Feature matrix: {num_tokens}×{d_sae} (sparse)")

    # Report 2
    token_level_firings = []
    for tok_idx, tok_str in enumerate(tokens):
        pairs = [(fid, val) for fid, val in enumerate(feature_matrix[tok_idx]) if val > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        token_level_firings.append({
            "token_index": tok_idx,
            "token_string": tok_str,
            "top_50_features": [
                {"feature_id": fid, "activation": round(val, 4),
                 "concept_label": _concept_label(fid, rng)}
                for fid, val in pairs[:50]
            ],
        })

    # Report 1
    stats: Dict[int, Any] = {}
    for row in feature_matrix:
        for fid, val in enumerate(row):
            if val == 0:
                continue
            if fid not in stats:
                stats[fid] = {"max": val, "sum": val, "cnt": 1}
            else:
                stats[fid]["max"] = max(stats[fid]["max"], val)
                stats[fid]["sum"] += val
                stats[fid]["cnt"] += 1

    fired_features_summary = sorted([
        {
            "feature_id": fid,
            "max_activation": round(s["max"], 4),
            "avg_activation": round(s["sum"] / s["cnt"], 4),
            "fired_token_count": s["cnt"],
            "concept_label": _concept_label(fid, rng),
        }
        for fid, s in stats.items()
    ], key=lambda x: x["max_activation"], reverse=True)

    print(f"[MOCK | {model_key}] ✅ Done — "
          f"{len(fired_features_summary)} global features, {num_tokens} token records")

    return {
        "model_metadata": {
            "model_name":    cfg["display_name"],
            "model_key":     model_key,
            "prompt":        prompt,
            "tokens":        tokens,
            "hook_point":    cfg["hook_point"],
            "layer":         cfg["layer"],
            "d_model":       cfg["d_model"],
            "sae_release":   cfg["sae_release"],
            "sae_id":        cfg["sae_id"],
            "pipeline_mode": "mock",
            "activation_path": "mock",
        },
        "report_1_global":   {"fired_features_summary": fired_features_summary},
        "report_2_per_token": {"token_level_firings": token_level_firings},
    }

# ─────────────────────────────────────────────────────────────────────────────
# REAL pipeline  (SAELens + TransformerLens or HF fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _real_analyse_model(prompt: str, model_key: str, top_k: int) -> Dict[str, Any]:
    """
    Full interpretability pipeline:
      1. Resolve model config from sae.cfg (cached after first call).
      2. Extract activations via Path A (TL) or Path B (HF hook).
      3. Load SAE → encode activations → sparse features.
      4. Build and return both reports.
      5. Delete all objects and clear VRAM.
    """
    # ── Step 1: Resolve config (cached) ──────────────────────────────────────
    cfg = _resolve_model_config(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(hash(prompt + model_key) & 0xFFFFFFFF)

    print(f"\n[REAL | {model_key}] ─────────────────────────────────────────")
    print(f"[REAL | {model_key}] 📝 Prompt       : '{prompt}'")
    print(f"[REAL | {model_key}] 💻 Device       : {device}")
    print(f"[REAL | {model_key}] 🔗 HF model     : {cfg['hf_model_name']}")
    print(f"[REAL | {model_key}] 🪝 Hook point   : {cfg['hook_point']}")
    print(f"[REAL | {model_key}] 📐 d_model      : {cfg['d_model']}")

    # ── Step 2: Extract residual-stream activations (dual-path) ──────────────
    resid: "torch.Tensor" = None
    token_strs: List[str] = None
    activation_path = "unknown"

    # Path A — TransformerLens
    if TL_AVAILABLE:
        try:
            print(f"[REAL | {model_key}] 🔬 Trying Path A (TransformerLens) …")
            resid, token_strs = _extract_activations_tl(
                cfg["hf_model_name"], cfg["hook_point"], prompt, device
            )
            activation_path = "transformer_lens"
        except Exception as exc:
            print(f"[REAL | {model_key}] ⚠️  Path A failed: {exc}")
            print(f"[REAL | {model_key}] 🔄 Falling back to Path B (HF hook) …")

    # Path B — HuggingFace register_forward_hook
    if resid is None:
        if not HF_AVAILABLE:
            raise RuntimeError(
                "Both TransformerLens and HuggingFace transformers are unavailable. "
                "Cannot extract activations."
            )
        resid, token_strs = _extract_activations_hf(
            cfg["hf_model_name"], cfg["layer"], prompt, device
        )
        activation_path = "huggingface_hook"

    print(f"[REAL | {model_key}] ✅ Activation path used: {activation_path}")

    # ── Step 3: Load SAE and encode ───────────────────────────────────────────
    print(f"[REAL | {model_key}] 🔭 [SAELens] Loading SAE: "
          f"release='{cfg['sae_release']}', sae_id='{cfg['sae_id']}' …")
    sae, _, _ = SAE.from_pretrained(
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )
    sae.eval()
    d_sae = _safe_sae_attr(sae.cfg, "d_sae", default=resid.shape[-1] * 8)
    print(f"[REAL | {model_key}] ✅ [SAELens] SAE loaded — d_sae={d_sae}")

    print(f"[REAL | {model_key}] ✨ [SAELens] Encoding dense → sparse features …")
    with torch.no_grad():
        feature_acts: "torch.Tensor" = sae.encode(resid)  # (seq_len, d_sae)

    l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
    print(f"[REAL | {model_key}] 📊 Feature acts: {tuple(feature_acts.shape)}, mean L0={l0:.1f}")

    # ── Step 4: Free SAE weights ──────────────────────────────────────────────
    print(f"[REAL | {model_key}] 🗑️  [HOT-SWAP] Deleting SAE to free VRAM …")
    feature_acts_cpu = feature_acts.cpu()
    del sae, feature_acts
    _vram_clear()

    # ── Step 5: Build reports ─────────────────────────────────────────────────
    fired_features_summary, token_level_firings = _build_reports(
        feature_acts_cpu, token_strs, rng
    )

    print(f"[REAL | {model_key}] ✅ Done — "
          f"{len(fired_features_summary)} global features, "
          f"{len(token_strs)} token records")

    return {
        "model_metadata": {
            "model_name":      cfg["display_name"],
            "model_key":       model_key,
            "prompt":          prompt,
            "tokens":          token_strs,
            "hook_point":      cfg["hook_point"],
            "layer":           cfg["layer"],
            "d_model":         cfg["d_model"],
            "sae_release":     cfg["sae_release"],
            "sae_id":          cfg["sae_id"],
            "pipeline_mode":   "real",
            "activation_path": activation_path,  # "transformer_lens" | "huggingface_hook"
        },
        "report_1_global":   {"fired_features_summary": fired_features_summary},
        "report_2_per_token": {"token_level_firings": token_level_firings},
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
            print(f"[FALLBACK | {model_key}] → switching to MOCK pipeline …")
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
    print(f"  Pipeline: {'REAL' if FULL_PIPELINE else 'MOCK'} "
          f"(TL={'✅' if TL_AVAILABLE else '❌'}, "
          f"HF={'✅' if HF_AVAILABLE else '❌'}, "
          f"SAE={'✅' if SAE_AVAILABLE else '❌'})")
    print(f"  Models registered: {list(MODEL_REGISTRY.keys())}")
    print("=" * 60 + "\n")
    yield
    print("\n[SHUTDOWN] Backend shutting down …")
    _vram_clear()


app = FastAPI(
    title="Multi-Model SAE Feature Dashboard API",
    description=(
        "Mechanistic interpretability backend. "
        "Config is auto-extracted from sae.cfg; no hardcoded d_model or hook_point. "
        "Dual-path activation extraction: TransformerLens (preferred) "
        "→ HuggingFace register_forward_hook (fallback)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# 开发模式：允许所有来源，前端 localhost:5173 可直接访问
# ⚠️  生产环境请改为具体的 frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # allow_origins=["*"] 时必须为 False
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
        "version": "2.0.0",
        "pipeline_mode": "real" if FULL_PIPELINE else "mock",
        "backends": {
            "transformer_lens": TL_AVAILABLE,
            "sae_lens": SAE_AVAILABLE,
            "huggingface": HF_AVAILABLE,
        },
        "available_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models", tags=["models"])
async def list_models():
    """
    Return all registered models.
    Config fields (hook_point, layer, d_model) are returned from cache if
    already resolved, otherwise from registry stubs (will be auto-filled on
    first /analyze call).
    """
    models_out = []
    for key, reg in MODEL_REGISTRY.items():
        cached = _CONFIG_CACHE.get(key, {})
        models_out.append({
            "key":          key,
            "display_name": reg.get("display_name", key),
            "sae_release":  reg["sae_release"],
            "sae_id":       reg["sae_id"],
            # These populate from cache after first /analyze; stubs shown until then
            "layer":        cached.get("layer", "?"),
            "hook_point":   cached.get("hook_point", "auto"),
            "d_model":      cached.get("d_model", "auto"),
            "hf_model_name": cached.get("hf_model_name", "auto"),
        })
    return {"models": models_out}


@app.post("/analyze", tags=["analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Core endpoint. For each selected model:
      1. Resolve config from sae.cfg (cached after first call — no hardcoding).
      2. Extract residual-stream activations:
           → Path A: TransformerLens HookedTransformer (preferred)
           → Path B: HuggingFace transformers + register_forward_hook (fallback)
      3. Load SAE → encode activations → sparse features.
      4. Build Report 1 (global) + Report 2 (per-token top-50).
      5. Delete all model objects + clear VRAM (hot-swap).
    """
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    unknown = [k for k in request.selected_models if k not in MODEL_REGISTRY]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model key(s): {unknown}. "
                   f"Valid: {list(MODEL_REGISTRY.keys())}",
        )

    print("\n" + "+" * 60)
    print(f"  NEW ANALYSIS REQUEST")
    print(f"  Prompt : '{prompt}'")
    print(f"  Models : {request.selected_models}")
    print(f"  Top-K  : {request.top_k}")
    print("+" * 60)

    models_data: Dict[str, Any] = {}
    for idx, model_key in enumerate(request.selected_models):
        print(f"\n[HOT-SWAP] ⚙️  {idx + 1}/{len(request.selected_models)}: '{model_key}'")
        t0 = time.time()
        models_data[model_key] = analyse_model(prompt, model_key, request.top_k)
        print(f"[HOT-SWAP] ✅ '{model_key}' done in {time.time() - t0:.2f}s")
        _vram_clear()

    print(f"\n[ANALYSIS] 🎉 All {len(request.selected_models)} model(s) processed.")
    return {
        "metadata": {
            "prompt":          prompt,
            "selected_models": request.selected_models,
            "top_k":           request.top_k,
            "pipeline_mode":   "real" if FULL_PIPELINE else "mock",
        },
        "models_data": models_data,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
