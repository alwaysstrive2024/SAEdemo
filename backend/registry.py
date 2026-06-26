"""
registry.py — Model Registry
=============================
★ THIS IS THE ONLY FILE YOU NEED TO EDIT TO ADD A NEW MODEL ★

Each entry requires exactly three fields:
  display_name  — human-readable name shown in the UI
  sae_release   — SAELens release string (HuggingFace repo or named release)
  sae_id        — SAELens SAE identifier within that release

All other metadata (d_model, hook_point, layer, hf_model_name) is extracted
automatically from sae.cfg at runtime and cached — no hardcoding required.

──────────────────────────────────────────────────────────────────────────────
How to find sae_release / sae_id for a new model:
  • Named SAELens releases: https://jbloomaus.github.io/SAELens/api/#saelens.pretrained_saes
  • HuggingFace repos:      from sae_lens import SAE
                            sae, _, _ = SAE.from_pretrained("<release>", "<sae_id>")
                            print(sae.cfg.hook_name, sae.cfg.d_in, sae.cfg.model_name)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Dict

# ─── Type alias ──────────────────────────────────────────────────────────────
ModelEntry = Dict[str, str]

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, ModelEntry] = {

    # ── EleutherAI Pythia ─────────────────────────────────────────────────────
    "pythia-70m": {
        "display_name": "Pythia 70M",
        "sae_release":  "pythia-70m-deduped",
        "sae_id":       "res-jb",
    },

    # ── Google Gemma 2B ───────────────────────────────────────────────────────
    "gemma-2b": {
        "display_name": "Gemma 2B",
        "sae_release":  "gemma-2b-res-jb",
        "sae_id":       "blocks.12.hook_resid_post",
    },

    # ── Meta Llama 3.2 1B ─────────────────────────────────────────────────────
    "llama-3.2-1b": {
        "display_name": "Llama 3.2 1B",
        "sae_release":  "llama_3.2_1b-res-jb",
        "sae_id":       "blocks.8.hook_resid_post",
    },

    # ── Google Gemma-4 (decoderesearch SAEs) ──────────────────────────────────
    # release: "decoderesearch/gemma-4-saes"
    # All metadata auto-resolved from sae.cfg.

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
        "display_name": "Gemma-4-E2B (Layer 28)",
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

    # ── Add new models below ──────────────────────────────────────────────────
    # Template:
    # "your-model-key": {
    #     "display_name": "Your Model Name",
    #     "sae_release":  "hf-org/repo-name",   # or named SAELens release
    #     "sae_id":       "path/to/sae-id",
    # },

}
