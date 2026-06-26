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
        "np_model_id":  "gemma-2-2b",
        "np_sae_id":    "12-res-jb",
    },

    # ── Meta Llama 3.2 1B ─────────────────────────────────────────────────────
    "llama-3.2-1b": {
        "display_name": "Llama 3.2 1B",
        "sae_release":  "chanind/llama_3.2_1b-res-jb",
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

    # —— GPT-2 Small ————————————————————————————————
    "gpt2-small-l4": {
        "display_name": "GPT-2 Small(Layer4)",
        "sae_release":  "gpt2-small-res-jb",
        "sae_id":       "blocks.4.hook_resid_pre",
    },

    "gpt2-small-l8": {
        "display_name": "GPT-2 Small(Layer8)",
        "sae_release":  "gpt2-small-res-jb",
        "sae_id":       "blocks.8.hook_resid_pre",
    },

    "gpt2-small-l12": {
        "display_name": "GPT-2 Small(Layer12)",
        "sae_release":  "gpt2-small-res-jb",
        "sae_id":       "blocks.12.hook_resid_pre",
    },
    # ── Olmo 3 7B ─────────────────────────────────────────────
    "olmo-3-7b-l4": {
        "display_name": "Olmo 3 7B (Layer 4)",
        "sae_release":  "decoderesearch/olmo-3-saes",
        "sae_id":       "olmo-3-1025-7b/btk-mat-layer-4-k-100",
    },

    "olmo-3-7b-l16": {
        "display_name": "Olmo 3 7B (Layer 16)",
        "sae_release":  "decoderesearch/olmo-3-saes",
        "sae_id":       "olmo-3-1025-7b/btk-mat-layer-16-k-100",
    },

    "olmo-3-7b-l28": {
        "display_name": "Olmo 3 7B (Layer 28)",
        "sae_release":  "decoderesearch/olmo-3-saes",
        "sae_id":       "olmo-3-1025-7b/btk-mat-layer-28-k-100",
    },

    # "olmo-3-7b-l31": {
    #     "display_name": "Olmo 3 7B (Layer 31)",
    #     "sae_release":  "decoderesearch/olmo-3-saes",
    #     "sae_id":       "layer_31/width_163k/topk_128_raw",
    # },


    # "olmo-3-7b-l15": {
    #     "display_name": "Olmo 3 7B (Layer 15)",
    #     "sae_release":  "dams2005/olmo-3-7b-think-saes",
    #     "sae_id":       "layer_15/width_163k/topk_64_raw",
    # },


    # —— Deepseek R1 ————————————————————————————————
    "deepseek-r1(llama-distill)": {
        "display_name": "Deepseek R1(llama-distill) (Layer 19)",
        "sae_release":  "andreuka18/sae-deepseek-r1-llama-8b",
        "sae_id":       "model.layers.19",
    },

    "deepseek-r1": {
        "display_name": "Deepseek R1(Layer 16)",
        "sae_release":  "Farmerobot/deepseek-r1-1.5b-sae-l16-topk32-v2",
        "sae_id":       "deepseek_base_l16_topk32_0M",
    },

    # —— Qwen 3.5 0.8B ————————————————————————————————
    "qwen-3.5-0.8b": {
        "display_name": "Qwen 3.5 0.8B",
        "sae_release":  "decoderesearch/qwen-3.5-saes",
        "sae_id":       "blocks.12.hook_resid_post",
    },

    # —— Qwen 3.5 4B base ————————————————————————————————

    # ——— Qwen 3.5 4B ————————————————————————————————

    # ── Add new models below ──────────────────────────────────────────────────
    # Template:
    # "your-model-key": {
    #     "display_name": "Your Model Name",
    #     "sae_release":  "hf-org/repo-name",   # or named SAELens release
    #     "sae_id":       "path/to/sae-id",
    # },

}
