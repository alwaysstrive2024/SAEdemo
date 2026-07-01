"""
pipeline.py — Analysis Engine
==============================
Handles all computation:
  • _resolve_model_config()  — lazy config extraction from sae.cfg (cached)
  • _extract_activations_tl()  — Path A: TransformerLens HookedTransformer
  • _extract_activations_hf()  — Path B: HuggingFace + register_forward_hook
  • _build_reports()           — Report 1 (global) + Report 2 (per-token)
  • _mock_analyse_model()      — Seeded-random mock (no ML deps needed)
  • _real_analyse_model()      — Full dual-path real pipeline
  • analyse_model()            — Public dispatch entry point
"""

from __future__ import annotations

import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from config import (
    layer_from_hook,
    naive_tokenise,
    safe_sae_attr,
    vram_clear,
)
from registry import MODEL_REGISTRY
from neuronpedia import get_concept_labels_batch

# Dependency flags — injected by main.py at startup
# ─────────────────────────────────────────────────────────────────────────────

TL_AVAILABLE:   bool = False
SAE_AVAILABLE:  bool = False
HF_AVAILABLE:   bool = False
TORCH_AVAILABLE: bool = False
FULL_PIPELINE:  bool = False


def set_deps(tl: bool, sae: bool, hf: bool, torch_ok: bool) -> None:
    """Called once by main.py after dependency detection."""
    global TL_AVAILABLE, SAE_AVAILABLE, HF_AVAILABLE, TORCH_AVAILABLE, FULL_PIPELINE
    TL_AVAILABLE    = tl
    SAE_AVAILABLE   = sae
    HF_AVAILABLE    = hf
    TORCH_AVAILABLE = torch_ok
    FULL_PIPELINE   = torch_ok and sae and (tl or hf)


# Config cache  (model_key → resolved config dict)
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def _stub_config(model_key: str) -> Dict[str, Any]:
    """Minimal config when SAELens is unavailable (mock mode only)."""
    reg = MODEL_REGISTRY[model_key]
    return {
        "display_name":  reg.get("display_name", model_key),
        "hf_model_name": model_key,
        "sae_release":   reg["sae_release"],
        "sae_id":        reg["sae_id"],
        "hook_point":    "blocks.4.hook_resid_post",
        "layer":         4,
        "d_model":       512,
    }


def _resolve_model_config(model_key: str) -> Dict[str, Any]:
    """
    Return fully-resolved config for a model key.

    First call: loads SAE just long enough to read sae.cfg, builds the config
    dict, caches it, then immediately frees SAE weights.
    Subsequent calls: returns from cache in O(1).
    """
    if model_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[model_key]

    reg = MODEL_REGISTRY[model_key]

    if not SAE_AVAILABLE:
        cfg = _stub_config(model_key)
        _CONFIG_CACHE[model_key] = cfg
        return cfg

    print(f"[CONFIG] 🔭 [SAELens] Fetching config for '{model_key}' …")
    print(f"[CONFIG]   release='{reg['sae_release']}'  sae_id='{reg['sae_id']}'")

    from sae_lens import SAE as _SAE
    sae_obj, _, _ = _SAE.from_pretrained(
        release=reg["sae_release"],
        sae_id=reg["sae_id"],
    )
    sae_cfg = sae_obj.cfg

    hook_point    = safe_sae_attr(sae_cfg, "hook_name", "hook_point",  default="")
    d_model       = safe_sae_attr(sae_cfg, "d_in",      "d_model",     default=512)
    hook_layer    = safe_sae_attr(sae_cfg, "hook_layer", "layer",       default=None)
    hf_model_name = safe_sae_attr(sae_cfg, "model_name", "model_id",   default=model_key)
    sae_id_read   = safe_sae_attr(sae_cfg, "sae_id",                    default=reg["sae_id"])

    # ── Registry overrides take priority over sae.cfg ───────────────────────
    # sae.cfg often returns wrong/empty values (e.g. 'gemma-2b' instead of
    # 'google/gemma-2b', or hook_name='').  Explicit registry entries win.
    if reg.get("hf_model_name"):
        hf_model_name = reg["hf_model_name"]
    if reg.get("hook_point"):
        hook_point = reg["hook_point"]
    if reg.get("hook_layer") is not None:
        hook_layer = reg["hook_layer"]
    # ────────────────────────────────────────────────────────────────────────

    if hook_layer is None:
        hook_layer = layer_from_hook(hook_point) or 0

    config: Dict[str, Any] = {
        "display_name":  reg.get("display_name", hf_model_name),
        "hf_model_name": hf_model_name,
        "sae_release":   reg["sae_release"],
        "sae_id":        sae_id_read,
        "hook_point":    hook_point,
        "layer":         hook_layer,
        "d_model":       d_model,
        "np_model_id":   reg.get("np_model_id"),
        "np_sae_id":     reg.get("np_sae_id"),
    }

    del sae_obj
    vram_clear()

    _CONFIG_CACHE[model_key] = config
    print(
        f"[CONFIG] ✅ Cached: hf='{hf_model_name}', "
        f"hook='{hook_point}', d_model={d_model}, layer={hook_layer}"
    )
    return config


# Path A — TransformerLens
# ─────────────────────────────────────────────────────────────────────────────

def _extract_activations_tl(
    hf_model_name: str,
    hook_point: str,
    prompt: str,
    device: str,
) -> Tuple[Any, List[str]]:
    """
    Use HookedTransformer.run_with_hooks() to capture residual-stream
    activations at `hook_point`.
    Returns (resid Tensor[seq, d_model], token_strs).
    """
    import torch
    from transformer_lens import HookedTransformer

    print(f"[PATH-A | TL] 🔬 [TransformerLens] Loading '{hf_model_name}' …")
    model = HookedTransformer.from_pretrained(
        hf_model_name,
        center_writing_weights=False,
        fold_ln=False,
        device=device,
    )
    model.eval()
    print(f"[PATH-A | TL] ✅ d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    tokens_tensor = model.to_tokens(prompt)
    token_strs: List[str] = model.to_str_tokens(prompt)
    print(f"[PATH-A | TL] 🔢 {len(token_strs)} tokens: {token_strs}")

    captured: Dict[str, Any] = {}

    def hook_fn(value: Any, hook: Any) -> Any:
        captured["resid"] = value.detach().clone()
        print(f"[PATH-A | TL] 🎯 Hook fired @ '{hook.name}' — {tuple(value.shape)}")
        return value

    with torch.no_grad():
        model.run_with_hooks(tokens_tensor, fwd_hooks=[(hook_point, hook_fn)])

    resid = captured["resid"][0]  # (seq, d_model)
    del model
    vram_clear()
    return resid, token_strs


# Path B — HuggingFace + register_forward_hook
# ─────────────────────────────────────────────────────────────────────────────

def _find_hf_layer(model: Any, layer_idx: int) -> Any:
    """
    Locate transformer block `layer_idx` across common HF architectures:
      Llama / Gemma / Mistral / Qwen  →  model.model.layers[n]
      GPT-NeoX / Pythia               →  model.gpt_neox.layers[n]
      GPT-2                            →  model.transformer.h[n]
      OPT                              →  model.model.decoder.layers[n]
    """
    candidates = [
        lambda m, n: m.model.layers[n],
        lambda m, n: m.gpt_neox.layers[n],
        lambda m, n: m.transformer.h[n],
        lambda m, n: m.model.decoder.layers[n],
        lambda m, n: m.transformer.blocks[n],
    ]
    for fn in candidates:
        try:
            layer = fn(model, layer_idx)
            if layer is not None:
                print(f"[PATH-B | HF] 🎯 Layer {layer_idx}: {type(layer).__name__}")
                return layer
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError(
        f"Cannot locate layer {layer_idx} in {type(model).__name__}. "
        "Add its attribute path to _find_hf_layer() in pipeline.py."
    )


def _extract_activations_hf(
    hf_model_name: str,
    layer_idx: int,
    prompt: str,
    device: str,
) -> Tuple[Any, List[str]]:
    """
    HuggingFace fallback: AutoModelForCausalLM + register_forward_hook.
    Returns (resid Tensor[seq, d_model], token_strs).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[PATH-B | HF] 🤗 Loading '{hf_model_name}' via AutoModelForCausalLM …")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
        device_map=device if device == "cpu" else "auto",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    token_strs = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    print(f"[PATH-B | HF] 🔢 {len(token_strs)} tokens: {token_strs}")

    captured: Dict[str, Any] = {}

    def hook_fn(module: Any, inp: Any, out: Any) -> None:
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        captured["hidden"] = hidden.detach().clone()
        print(f"[PATH-B | HF] 🎯 Hook fired @ layer {layer_idx} — {tuple(hidden.shape)}")

    target_layer = _find_hf_layer(model, layer_idx)
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)
    handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Hook did not fire — no activations captured.")

    resid = captured["hidden"][0]  # (seq, d_model)
    del model
    vram_clear()
    return resid, token_strs


# Report builder  (shared by real and mock pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def _build_reports(
    feature_acts_cpu: Any,
    token_strs: List[str],
    rng: random.Random,
    np_model_id: str,
    np_sae_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert a (seq_len, d_sae) activation tensor (CPU numpy) into:
      fired_features_summary  — Report 1 global feature list
      token_level_firings     — Report 2 per-token top-50
    """
    import numpy as _np
    fa = feature_acts_cpu.numpy()  # (seq, d_sae)

    # Find all active feature IDs
    active_mask = fa > 0
    active_ids = active_mask.any(axis=0).nonzero()[0].tolist()

    print(f"[Neuronpedia] Fetching labels for {len(active_ids)} unique features...")
    labels_map = get_concept_labels_batch(np_model_id, np_sae_id, active_ids)

    # Report 2 — per-token
    token_level_firings: List[Dict[str, Any]] = []
    for tok_idx, tok_str in enumerate(token_strs):
        row = fa[tok_idx]
        pairs = [(int(fid), float(row[fid])) for fid in (row > 0).nonzero()[0]]
        pairs.sort(key=lambda x: x[1], reverse=True)
        token_level_firings.append({
            "token_index":    tok_idx,
            "token_string":   tok_str,
            "top_50_features": [
                {
                    "feature_id":    fid,
                    "activation":    round(val, 4),
                    "concept_label": labels_map.get(fid, f"Concept {fid}"),
                }
                for fid, val in pairs[:50]
            ],
        })

    # Report 1 — global aggregation
    global_max   = fa.max(axis=0)
    global_sum   = fa.sum(axis=0)
    global_count = active_mask.sum(axis=0)

    fired_features_summary: List[Dict[str, Any]] = sorted([
        {
            "feature_id":       fid,
            "max_activation":   round(float(global_max[fid]), 4),
            "avg_activation":   round(float(global_sum[fid]) / int(global_count[fid]), 4),
            "fired_token_count": int(global_count[fid]),
            "concept_label":    labels_map.get(fid, f"Concept {fid}"),
        }
        for fid in active_ids
    ], key=lambda x: x["max_activation"], reverse=True)

    return fired_features_summary, token_level_firings


# Mock pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _mock_analyse_model(
    prompt: str, model_key: str, top_k: int
) -> Dict[str, Any]:
    """Deterministic seeded-random mock — full JSON schema, no ML deps."""
    seed = hash(prompt + model_key) & 0xFFFFFFFF
    rng  = random.Random(seed)

    cfg = _CONFIG_CACHE.get(model_key) or _stub_config(model_key)
    tokens = naive_tokenise(prompt)
    d_sae  = 4096

    print(f"\n[MOCK | {model_key}] prompt='{prompt}', tokens={tokens}")
    print(f"[MOCK | {model_key}] 🔬 [TransformerLens] Simulating load '{cfg['hf_model_name']}'")
    time.sleep(0.05)
    print(f"[MOCK | {model_key}] 🪝 hook='{cfg['hook_point']}' "
          f"layer={cfg['layer']} d_model={cfg['d_model']}")
    print(f"[MOCK | {model_key}] 🔭 [SAELens] Encoding → sparse (d_sae={d_sae})")
    time.sleep(0.03)

    # Sparse feature matrix
    matrix: List[List[float]] = []
    for _ in tokens:
        row = [0.0] * d_sae
        for fid in rng.sample(range(d_sae), rng.randint(15, 35)):
            row[fid] = round(rng.uniform(0.1, 6.5), 4)
        matrix.append(row)

    active_ids = set()
    for row in matrix:
        for fid, val in enumerate(row):
            if val > 0:
                active_ids.add(fid)
    
    labels_map = get_concept_labels_batch(cfg.get("np_model_id"), cfg.get("np_sae_id"), list(active_ids))

    # Report 2
    token_level_firings = []
    for tok_idx, tok_str in enumerate(tokens):
        pairs = [(fid, val) for fid, val in enumerate(matrix[tok_idx]) if val > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        token_level_firings.append({
            "token_index":    tok_idx,
            "token_string":   tok_str,
            "top_50_features": [
                {"feature_id": fid, "activation": round(val, 4),
                 "concept_label": labels_map.get(fid, f"Concept {fid}")}
                for fid, val in pairs[:50]
            ],
        })

    # Report 1
    stats: Dict[int, Any] = {}
    for row in matrix:
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
            "feature_id":       fid,
            "max_activation":   round(s["max"], 4),
            "avg_activation":   round(s["sum"] / s["cnt"], 4),
            "fired_token_count": s["cnt"],
            "concept_label":    labels_map.get(fid, f"Concept {fid}"),
        }
        for fid, s in stats.items()
    ], key=lambda x: x["max_activation"], reverse=True)

    print(f"[MOCK | {model_key}] ✅ {len(fired_features_summary)} features, "
          f"{len(tokens)} tokens")

    return {
        "model_metadata": {
            "model_name":      cfg["display_name"],
            "model_key":       model_key,
            "prompt":          prompt,
            "tokens":          tokens,
            "hook_point":      cfg["hook_point"],
            "layer":           cfg["layer"],
            "d_model":         cfg["d_model"],
            "sae_release":     cfg["sae_release"],
            "sae_id":          cfg["sae_id"],
            "pipeline_mode":   "mock",
            "activation_path": "mock",
        },
        "report_1_global":    {"fired_features_summary": fired_features_summary},
        "report_2_per_token": {"token_level_firings": token_level_firings},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _real_analyse_model(
    prompt: str, model_key: str, top_k: int
) -> Dict[str, Any]:
    """
    Full dual-path pipeline:
      1. Resolve config (cached after first call)
      2. Extract activations: Path A (TL) → Path B (HF) on failure
      3. Load SAE → encode → sparse features
      4. Build reports
      5. Free all objects + clear VRAM
    """
    import torch

    cfg    = _resolve_model_config(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng    = random.Random(hash(prompt + model_key) & 0xFFFFFFFF)

    print(f"\n[REAL | {model_key}] prompt='{prompt}' device={device}")
    print(f"[REAL | {model_key}] hf='{cfg['hf_model_name']}' "
          f"hook='{cfg['hook_point']}' d_model={cfg['d_model']}")

    # ── Activation extraction ─────────────────────────────────────────────────
    resid: Any        = None
    token_strs: List[str] = []
    activation_path   = "unknown"

    if TL_AVAILABLE:
        try:
            print(f"[REAL | {model_key}] 🔬 Trying Path A (TransformerLens) …")
            resid, token_strs = _extract_activations_tl(
                cfg["hf_model_name"], cfg["hook_point"], prompt, device
            )
            activation_path = "transformer_lens"
        except Exception as exc:
            print(f"[REAL | {model_key}] ⚠️  Path A failed: {exc}")

    if resid is None:
        if not HF_AVAILABLE:
            raise RuntimeError(
                "TransformerLens unavailable and HuggingFace transformers not installed."
            )
        print(f"[REAL | {model_key}] 🤗 Trying Path B (HuggingFace hook) …")
        resid, token_strs = _extract_activations_hf(
            cfg["hf_model_name"], cfg["layer"], prompt, device
        )
        activation_path = "huggingface_hook"

    print(f"[REAL | {model_key}] ✅ Activation path: {activation_path}")

    # ── SAE encode ────────────────────────────────────────────────────────────
    from sae_lens import SAE as _SAE

    print(f"[REAL | {model_key}] 🔭 [SAELens] Loading SAE "
          f"release='{cfg['sae_release']}' sae_id='{cfg['sae_id']}' …")
    sae, _, _ = _SAE.from_pretrained(
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )
    sae.eval()
    d_sae = safe_sae_attr(sae.cfg, "d_sae", default=resid.shape[-1] * 8)
    print(f"[REAL | {model_key}] ✅ [SAELens] d_sae={d_sae}")

    with torch.no_grad():
        feature_acts = sae.encode(resid)  # (seq, d_sae)

    l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
    print(f"[REAL | {model_key}] 📊 {tuple(feature_acts.shape)}, mean L0={l0:.1f}")

    print(f"[REAL | {model_key}] 🗑️  [HOT-SWAP] Freeing SAE …")
    feature_acts_cpu = feature_acts.cpu()
    del sae, feature_acts
    vram_clear()

    # ── Build reports ─────────────────────────────────────────────────────────
    fired_features_summary, token_level_firings = _build_reports(
        feature_acts_cpu, token_strs, rng, cfg.get("np_model_id"), cfg.get("np_sae_id")
    )

    print(f"[REAL | {model_key}] ✅ Done — "
          f"{len(fired_features_summary)} features, {len(token_strs)} tokens")

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
            "activation_path": activation_path,
        },
        "report_1_global":    {"fired_features_summary": fired_features_summary},
        "report_2_per_token": {"token_level_firings": token_level_firings},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatch
# ─────────────────────────────────────────────────────────────────────────────

def analyse_model(prompt: str, model_key: str, top_k: int) -> Dict[str, Any]:
    """
    Entry point called by the API route.
    Tries real pipeline first; falls back to mock on any error.
    """
    if FULL_PIPELINE:
        try:
            return _real_analyse_model(prompt, model_key, top_k)
        except Exception as exc:
            print(f"[ERROR | {model_key}] Real pipeline failed: {exc}")
            print(traceback.format_exc())
            print(f"[FALLBACK | {model_key}] → mock pipeline")
            return _mock_analyse_model(prompt, model_key, top_k)
    return _mock_analyse_model(prompt, model_key, top_k)
