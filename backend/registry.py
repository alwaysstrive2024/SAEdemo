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

    # ──EleutherAI Pythia─────────────────────────────────────────────────
    "pythia": {
        "display_name":  "Pythia",
        "sae_release":   "timhua/pythia1b_deduped_saes",
        "sae_id":        "pythia1b_614mtoks",
        "hf_model_name": "EleutherAI/pythia-70m-deduped",
        "hook_point":    "blocks.4.hook_resid_post",
        "hook_layer":    4,
    },

    # ── Google Gemma 2B ───────────────────────────────────────────────────────
    "gemma-2b": {
        "display_name":  "Gemma 2B",
        "sae_release":   "gemma-2b-res-jb",
        "sae_id":        "blocks.12.hook_resid_post",
        "hf_model_name": "google/gemma-2b",
        "hook_point":    "blocks.12.hook_resid_post",
        "hook_layer":    12,
        "np_model_id":   "gemma-2-2b",
        "np_sae_id":     "12-res-jb",
    },

    # ── Meta Llama 3.2 1B ─────────────────────────────────────────────────────
    "llama-3.2-1b": {
        "display_name":  "Llama 3.2 1B",
        "sae_release":   "seonglae/Llama-3.2-1B-sae",
        "sae_id":        "Llama-3.2-1B_blocks.12.hook_resid_pre_14336_topk_48_0.0002_49_fineweb_512",
        "hf_model_name": "meta-llama/Llama-3.2-1B",
        "hook_point":    "blocks.8.hook_resid_post",
        "hook_layer":    8,
    },

    # ── GPT-2 Small ───────────────────────────────────────────────────────────
    # TransformerLens 支持 "gpt2"，hook_point 用 resid_pre（SAE 训练时用的是 pre）
    "gpt2-small-l4": {
        "display_name":  "GPT-2 Small (Layer 4)",
        "sae_release":   "gpt2-small-res-jb",
        "sae_id":        "blocks.4.hook_resid_pre",
        "hf_model_name": "gpt2",
        "hook_point":    "blocks.4.hook_resid_pre",
        "hook_layer":    4,
    },

    "gpt2-small-l8": {
        "display_name":  "GPT-2 Small (Layer 8)",
        "sae_release":   "gpt2-small-res-jb",
        "sae_id":        "blocks.8.hook_resid_pre",
        "hf_model_name": "gpt2",
        "hook_point":    "blocks.8.hook_resid_pre",
        "hook_layer":    8,
    },
    "gpt2-small-l12": {
        "display_name":  "GPT-2 Small (Layer 12)",
        "sae_release":   "gpt2-small-res-jb",
        "sae_id":        "blocks.12.hook_resid_pre",
        "hf_model_name": "gpt2",
        "hook_point":    "blocks.12.hook_resid_pre",
        "hook_layer":    12,
    },

    # ── Google Gemma-4 E2B (decoderesearch SAEs) ──────────────────────────────
    # 基座模型: google/gemma-4-E2B（Gemma 4 第二代，2B 参数）
    "gemma-4-e2b-l6": {
        "display_name":  "Gemma-4 E2B (Layer 6)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-e2b/btk-mat-layer-6-k-100",
        "hf_model_name": "google/gemma-4-E2B",
        "hook_point":    "blocks.6.hook_resid_post",
        "hook_layer":    6,
    },
    "gemma-4-e2b-l17": {
        "display_name":  "Gemma-4 E2B (Layer 17)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-e2b/btk-mat-layer-17-k-100",
        "hf_model_name": "google/gemma-4-E2B",
        "hook_point":    "blocks.17.hook_resid_post",
        "hook_layer":    17,
    },
    "gemma-4-e2b-l28": {
        "display_name":  "Gemma-4 E2B (Layer 28)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-e2b/btk-mat-layer-28-k-100",
        "hf_model_name": "google/gemma-4-E2B",
        "hook_point":    "blocks.28.hook_resid_post",
        "hook_layer":    28,
    },

    # ── Google Gemma-4 E4B (decoderesearch SAEs) ──────────────────────────────
    # 基座模型: google/gemma-4-4b（Gemma 4 第二代，4B 参数）
    # ⚠️  如果下载时报 404，请改为 google/gemma-4-4b-it（instruction-tuned 版本）
    "gemma-4-e4b-l7": {
        "display_name":  "Gemma-4 E4B (Layer 7)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-e4b/btk-mat-layer-7-k-100",
        "hf_model_name": "google/gemma-4-E4B-it",
        "hook_point":    "blocks.7.hook_resid_post",
        "hook_layer":    7,
    },
    "gemma-4-e4b-l21": {
        "display_name":  "Gemma-4 E4B (Layer 21)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-e4b/btk-mat-layer-21-k-100",
        "hf_model_name": "google/gemma-4-E4B",
        "hook_point":    "blocks.21.hook_resid_post",
        "hook_layer":    21,
    },
    "gemma-4-e4b-l35": {
        "display_name":  "Gemma-4 E4B (Layer 35)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-e4b/btk-mat-layer-35-k-100",
        "hf_model_name": "google/gemma-4-E4B-it",
        "hook_point":    "blocks.35.hook_resid_post",
        "hook_layer":    35,
    },

    # ── Google Gemma-4 31B (decoderesearch SAEs) ──────────────────────────────
    # 基座模型: google/gemma-4-31B-it
    "gemma-4-31b-l30": {
        "display_name":  "Gemma-4 31B (Layer 30)",
        "sae_release":   "decoderesearch/gemma-4-saes",
        "sae_id":        "gemma-4-31b/btk-mat-layer-30-k-100",
        "hf_model_name": "google/gemma-4-31B-it",
        "hook_point":    "blocks.30.hook_resid_post",
        "hook_layer":    30,
    },

    # ── AllenAI OLMo 3 7B (decoderesearch SAEs) ──────────────────────────────
    "olmo-3-7b-l4": {
        "display_name":  "OLMo 3 7B (Layer 4)",
        "sae_release":   "decoderesearch/olmo-3-saes",
        "sae_id":        "olmo-3-1025-7b/btk-mat-layer-4-k-100",
        "hf_model_name": "allenai/OLMo-2-1124-7B",
        "hook_point":    "blocks.4.hook_resid_post",
        "hook_layer":    4,
    },
    "olmo-3-7b-l16": {
        "display_name":  "OLMo 3 7B (Layer 16)",
        "sae_release":   "decoderesearch/olmo-3-saes",
        "sae_id":        "olmo-3-1025-7b/btk-mat-layer-16-k-100",
        "hf_model_name": "allenai/OLMo-2-1124-7B",
        "hook_point":    "blocks.16.hook_resid_post",
        "hook_layer":    16,
    },
    "olmo-3-7b-l28": {
        "display_name":  "OLMo 3 7B (Layer 28)",
        "sae_release":   "decoderesearch/olmo-3-saes",
        "sae_id":        "olmo-3-1025-7b/btk-mat-layer-28-k-100",
        "hf_model_name": "allenai/OLMo-2-1124-7B",
        "hook_point":    "blocks.28.hook_resid_post",
        "hook_layer":    28,
    },

    # ── DeepSeek R1 ───────────────────────────────────────────────────────────
    "deepseek-r1(llama-distill)": {
        "display_name":  "DeepSeek R1 Llama Distill (Layer 19)",
        "sae_release":   "andreuka18/sae-deepseek-r1-llama-8b",
        "sae_id":        "model.layers.19",
        "hf_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hook_point":    "blocks.19.hook_resid_post",
        "hook_layer":    19,
    },
    "deepseek-r1": {
        "display_name":  "DeepSeek R1 1.5B (Layer 16)",
        "sae_release":   "Farmerobot/deepseek-r1-1.5b-sae-l16-topk32-v2",
        "sae_id":        "deepseek_base_l16_topk32_0M",
        "hf_model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "hook_point":    "blocks.16.hook_resid_post",
        "hook_layer":    16,
    },

    # ── Qwen 3.5 0.8B (decoderesearch SAEs) ──────────────────────────────────
    "qwen-3.5-0.8b": {
        "display_name":  "Qwen 3.5 0.8B (Layer 11)",
        "sae_release":   "decoderesearch/qwen-3.5-saes",
        "sae_id":        "qwen-3.5-0.8b/btk-mat-layer-11-k-100",
        "hf_model_name": "Qwen/Qwen2.5-0.5B",
        "hook_point":    "blocks.11.hook_resid_post",
        "hook_layer":    11,
    },

    # ── Add new models below ──────────────────────────────────────────────────
    # Template:
    # "your-model-key": {
    #     "display_name":  "Your Model Name",
    #     "sae_release":   "hf-org/repo-name",
    #     "sae_id":        "path/to/sae-id",
    #     "hf_model_name": "org/model-name",       # 真实 HF 基座模型 ID
    #     "hook_point":    "blocks.N.hook_resid_post",
    #     "hook_layer":    N,
    # },

}

