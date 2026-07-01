"""
test_model_loading.py
=====================
Standalone diagnostic script for running on Google Colab.

Tests:
  1. SAELens  — SAE.from_pretrained()  → prints raw sae.cfg fields
  2. TransformerLens — HookedTransformer.from_pretrained()  (Path A)
  3. HuggingFace     — AutoModelForCausalLM.from_pretrained() (Path B fallback)
  4. End-to-end activation extraction on a test prompt

Usage on Colab:
  !pip install sae-lens transformer-lens transformers accelerate torch -q
  # Put registry.py in the same folder, then:
  !python test_model_loading.py

Configure which models to test by editing MODELS_TO_TEST below.
"""

import gc
import sys
import traceback
from typing import Any, Dict, List, Optional

# ─── 配置：选择你要测试的模型 key（和 registry.py 里的 key 完全一致）──────────
MODELS_TO_TEST: List[str] = [
    "gemma-2b",
    "pythia-70m",
    # "gpt2-small-l4",
    # "llama-3.2-1b",
    # "gemma-4-e2b-l6",
    # "gemma-4-e2b-l17",
    # "gemma-4-e2b-l28",
    # "gemma-4-e4b-l7",
    # "gemma-4-e4b-l21",
    # "gemma-4-e4b-l35",
    # "gemma-4-31b-l30",
    # "olmo-3-7b-l4",
    # "deepseek-r1",
    # "qwen-3.5-0.8b",
]

# 用来测试 activation 提取的 prompt
TEST_PROMPT = "The quick brown fox jumps over the lazy dog."

# 是否跑完整的激活提取测试（会真正加载大模型，Colab T4 可能 OOM）
# 建议先设为 False 跑名字/cfg 测试，确认没问题后再设为 True
RUN_ACTIVATION_TEST = False

# ─── 颜色输出辅助 ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ  {msg}{RESET}")

# ─── 加载 registry ────────────────────────────────────────────────────────────
try:
    from registry import MODEL_REGISTRY
    ok(f"registry.py loaded — {len(MODEL_REGISTRY)} models registered")
except ImportError:
    fail("registry.py not found in the current directory. Make sure it's in the same folder.")
    sys.exit(1)

# ─── 检测运行环境 ─────────────────────────────────────────────────────────────
print(f"\n{BOLD}=== Environment Check ==={RESET}")

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok(f"PyTorch {torch.__version__} — device={device}")
    if device == "cuda":
        info(f"GPU: {torch.cuda.get_device_name(0)} | "
             f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
except ImportError:
    fail("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

try:
    import sae_lens
    ok(f"SAELens {sae_lens.__version__}")
    SAE_OK = True
except ImportError:
    fail("SAELens not installed. Run: pip install sae-lens")
    SAE_OK = False

try:
    import transformer_lens
    ok(f"TransformerLens {transformer_lens.__version__}")
    TL_OK = True
except ImportError:
    warn("TransformerLens not installed (Path A will be skipped). Run: pip install transformer-lens")
    TL_OK = False

try:
    import transformers
    ok(f"Transformers {transformers.__version__}")
    HF_OK = True
except ImportError:
    warn("HuggingFace Transformers not installed. Run: pip install transformers")
    HF_OK = False


def vram_clear():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─── 单个模型测试逻辑 ─────────────────────────────────────────────────────────

def test_model(model_key: str) -> Dict[str, Any]:
    """
    Run all diagnostic checks for one registry entry.
    Returns a result dict with keys:
      model_key, registry_ok, sae_ok, cfg_match, tl_ok, hf_ok, activation_ok, errors
    """
    result = {
        "model_key":     model_key,
        "registry_ok":   False,
        "sae_ok":        False,
        "sae_cfg":       {},
        "cfg_match":     False,
        "tl_ok":         False,
        "hf_ok":         False,
        "activation_ok": False,
        "errors":        [],
    }

    sep = "─" * 60
    print(f"\n{BOLD}{sep}")
    print(f"  Testing: {model_key}")
    print(f"{sep}{RESET}")

    # ── 1. Registry check ────────────────────────────────────────────────────
    print(f"\n{BOLD}[1] Registry{RESET}")
    if model_key not in MODEL_REGISTRY:
        fail(f"'{model_key}' not in MODEL_REGISTRY")
        result["errors"].append("Key not in registry")
        return result

    reg = MODEL_REGISTRY[model_key]
    result["registry_ok"] = True
    ok(f"Found in registry")
    info(f"display_name : {reg.get('display_name')}")
    info(f"sae_release  : {reg.get('sae_release')}")
    info(f"sae_id       : {reg.get('sae_id')}")
    info(f"hf_model_name: {reg.get('hf_model_name', '⚠️  NOT SET')}")
    info(f"hook_point   : {reg.get('hook_point',    '⚠️  NOT SET')}")
    info(f"hook_layer   : {reg.get('hook_layer',    '⚠️  NOT SET')}")

    if not reg.get("hf_model_name"):
        warn("hf_model_name is missing — model loading will fall back to key name (likely wrong)")
    if not reg.get("hook_point"):
        warn("hook_point is missing — will try to read from sae.cfg")

    # ── 2. SAELens check ─────────────────────────────────────────────────────
    print(f"\n{BOLD}[2] SAELens — SAE.from_pretrained(){RESET}")
    if not SAE_OK:
        warn("Skipped — SAELens not installed")
    else:
        try:
            from sae_lens import SAE
            print(f"  Loading SAE: release='{reg['sae_release']}' sae_id='{reg['sae_id']}'")
            sae_obj, _, _ = SAE.from_pretrained(
                release=reg["sae_release"],
                sae_id=reg["sae_id"],
            )
            cfg = sae_obj.cfg

            # Print all cfg fields that are non-empty
            cfg_fields = {}
            for attr in ["model_name", "hook_name", "hook_layer", "d_in", "d_sae",
                         "hook_point", "layer", "model_id", "dtype"]:
                val = getattr(cfg, attr, None)
                if val is not None and val != "":
                    cfg_fields[attr] = val

            result["sae_ok"] = True
            result["sae_cfg"] = cfg_fields
            ok(f"SAE loaded successfully")

            print(f"\n  {BOLD}Raw sae.cfg fields:{RESET}")
            for k, v in cfg_fields.items():
                print(f"    cfg.{k:20s} = {repr(v)}")

            # ── Compare sae.cfg vs registry ──────────────────────────────────
            print(f"\n  {BOLD}Registry vs sae.cfg comparison:{RESET}")
            mismatches = []

            sae_model_name = cfg_fields.get("model_name") or cfg_fields.get("model_id", "")
            reg_hf = reg.get("hf_model_name", "")
            if sae_model_name and reg_hf:
                match = sae_model_name.lower() in reg_hf.lower() or reg_hf.lower().endswith(sae_model_name.lower())
                sym = f"{GREEN}✅" if match else f"{RED}⚠️ "
                print(f"    {sym} hf_model_name: registry='{reg_hf}' | sae.cfg='{sae_model_name}'{RESET}")
                if not match:
                    mismatches.append(f"hf_model_name mismatch: cfg='{sae_model_name}' vs registry='{reg_hf}'")
            elif not sae_model_name:
                warn(f"sae.cfg.model_name is empty — registry value '{reg_hf}' will be used")

            sae_hook = cfg_fields.get("hook_name") or cfg_fields.get("hook_point", "")
            reg_hook = reg.get("hook_point", "")
            if sae_hook and reg_hook:
                sym = f"{GREEN}✅" if sae_hook == reg_hook else f"{RED}⚠️ "
                print(f"    {sym} hook_point   : registry='{reg_hook}' | sae.cfg='{sae_hook}'{RESET}")
                if sae_hook != reg_hook:
                    mismatches.append(f"hook_point mismatch: cfg='{sae_hook}' vs registry='{reg_hook}'")
            elif not sae_hook:
                warn(f"sae.cfg.hook_name is empty — registry value '{reg_hook}' will be used")

            sae_layer = cfg_fields.get("hook_layer") or cfg_fields.get("layer")
            reg_layer = reg.get("hook_layer")
            if sae_layer is not None and reg_layer is not None:
                sym = f"{GREEN}✅" if int(sae_layer) == int(reg_layer) else f"{RED}⚠️ "
                print(f"    {sym} hook_layer   : registry={reg_layer} | sae.cfg={sae_layer}{RESET}")
                if int(sae_layer) != int(reg_layer):
                    mismatches.append(f"hook_layer mismatch: cfg={sae_layer} vs registry={reg_layer}")

            sae_d_in = cfg_fields.get("d_in")
            if sae_d_in:
                info(f"d_model (d_in) : {sae_d_in}")

            result["cfg_match"] = len(mismatches) == 0
            if mismatches:
                for m in mismatches:
                    warn(m)
            else:
                ok("Registry values match sae.cfg (or sae.cfg is empty → registry overrides will be used)")

            del sae_obj
            vram_clear()

        except Exception as e:
            fail(f"SAE loading failed: {e}")
            result["errors"].append(f"SAE: {e}")
            traceback.print_exc()

    # ── 3. TransformerLens check (Path A) ────────────────────────────────────
    print(f"\n{BOLD}[3] TransformerLens — Path A{RESET}")
    hf_name = reg.get("hf_model_name") or model_key
    hook_pt  = reg.get("hook_point", "blocks.0.hook_resid_post")

    if not TL_OK:
        warn("Skipped — TransformerLens not installed")
    elif not RUN_ACTIVATION_TEST:
        info("Skipped (RUN_ACTIVATION_TEST=False) — only checking model name recognition")
        # Just check if TL recognizes the model name without downloading
        try:
            from transformer_lens import HookedTransformer
            # get_pretrained_model_config doesn't download weights
            cfg_tl = HookedTransformer.get_pretrained_model_config(hf_name)
            ok(f"TransformerLens recognizes '{hf_name}'")
            info(f"TL config: d_model={cfg_tl.d_model}, n_layers={cfg_tl.n_layers}, n_heads={cfg_tl.n_heads}")
            result["tl_ok"] = True
        except Exception as e:
            fail(f"TransformerLens does NOT recognize '{hf_name}': {e}")
            info("This means Path A will fail at runtime — Path B (HuggingFace) will be used instead")
            result["errors"].append(f"TL name not recognized: {e}")
    else:
        try:
            from transformer_lens import HookedTransformer
            print(f"  Loading '{hf_name}' via HookedTransformer …")
            model = HookedTransformer.from_pretrained(
                hf_name,
                center_writing_weights=False,
                fold_ln=False,
                device=device,
            )
            model.eval()
            ok(f"HookedTransformer loaded: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")
            result["tl_ok"] = True

            # Test forward pass with hook
            tokens = model.to_tokens(TEST_PROMPT)
            captured = {}
            def hook_fn(value, hook):
                captured["resid"] = value.detach().clone()
                return value

            import torch
            with torch.no_grad():
                model.run_with_hooks(tokens, fwd_hooks=[(hook_pt, hook_fn)])

            if "resid" in captured:
                ok(f"Activation captured at '{hook_pt}' — shape: {tuple(captured['resid'].shape)}")
                result["activation_ok"] = True
            else:
                fail(f"Hook did not fire at '{hook_pt}' — hook_point may be wrong")
                result["errors"].append(f"Hook did not fire at '{hook_pt}'")

            del model
            vram_clear()

        except Exception as e:
            fail(f"TransformerLens loading failed: {e}")
            result["errors"].append(f"TL: {e}")

    # ── 4. HuggingFace check (Path B) ────────────────────────────────────────
    print(f"\n{BOLD}[4] HuggingFace — Path B (AutoModelForCausalLM){RESET}")
    if not HF_OK:
        warn("Skipped — transformers not installed")
    elif not RUN_ACTIVATION_TEST:
        info("Skipped (RUN_ACTIVATION_TEST=False) — checking model name via HuggingFace Hub API")
        try:
            from huggingface_hub import model_info
            mi = model_info(hf_name)
            ok(f"Model '{hf_name}' exists on HuggingFace Hub")
            info(f"Model ID: {mi.id} | Downloads: {mi.downloads}")
            result["hf_ok"] = True
        except Exception as e:
            fail(f"Model '{hf_name}' not found on HuggingFace Hub: {e}")
            info("Double check the hf_model_name in registry.py")
            result["errors"].append(f"HF Hub: {e}")
    else:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            def find_layer(model, layer_idx):
                for fn in [
                    lambda m, n: m.model.layers[n],
                    lambda m, n: m.gpt_neox.layers[n],
                    lambda m, n: m.transformer.h[n],
                    lambda m, n: m.model.decoder.layers[n],
                ]:
                    try:
                        return fn(model, layer_idx)
                    except (AttributeError, IndexError, TypeError):
                        continue
                raise RuntimeError(f"Cannot locate layer {layer_idx}")

            layer_idx = reg.get("hook_layer", 0)
            print(f"  Loading '{hf_name}' via AutoModelForCausalLM …")
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            ok(f"HuggingFace model loaded")
            result["hf_ok"] = True

            inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            captured = {}
            target_layer = find_layer(model, layer_idx)
            def hook_fn(module, inp, out):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                captured["resid"] = hidden.detach().clone()
            handle = target_layer.register_forward_hook(hook_fn)

            with torch.no_grad():
                model(**inputs)
            handle.remove()

            if "resid" in captured:
                ok(f"Activation captured at layer {layer_idx} — shape: {tuple(captured['resid'].shape)}")
                result["activation_ok"] = True
            else:
                fail(f"Hook did not fire at layer {layer_idx}")

            del model, tokenizer
            vram_clear()

        except Exception as e:
            fail(f"HuggingFace loading failed: {e}")
            result["errors"].append(f"HF: {e}")

    return result


# ─── 主测试循环 ───────────────────────────────────────────────────────────────

all_results = []

for key in MODELS_TO_TEST:
    r = test_model(key)
    all_results.append(r)

# ─── 汇总报告 ─────────────────────────────────────────────────────────────────
print(f"\n\n{BOLD}{'═' * 70}")
print("  SUMMARY")
print(f"{'═' * 70}{RESET}")

header = f"{'Model Key':<32} {'Registry':^8} {'SAE':^8} {'CFG Match':^10} {'TL':^8} {'HF':^8} {'Errors'}"
print(f"\n{BOLD}{header}{RESET}")
print("─" * 90)

for r in all_results:
    def sym(b): return f"{GREEN}✅{RESET}" if b else f"{RED}❌{RESET}"
    errors_str = "; ".join(r["errors"])[:40] if r["errors"] else ""
    print(
        f"{r['model_key']:<32} "
        f"{sym(r['registry_ok']):^8} "
        f"{sym(r['sae_ok']):^8} "
        f"{sym(r['cfg_match']):^10} "
        f"{sym(r['tl_ok']):^8} "
        f"{sym(r['hf_ok']):^8} "
        f"{RED}{errors_str}{RESET}"
    )

print(f"\n{BOLD}Tips:{RESET}")
print("  • If SAE ❌ → check sae_release / sae_id in registry.py")
print("  • If CFG Match ❌ → sae.cfg returns different values; update hf_model_name/hook_point in registry.py")
print("  • If TL ❌ and HF ✅ → TransformerLens doesn't support this model; Path B will be used at runtime")
print("  • If HF ❌ → hf_model_name in registry.py is wrong; check the exact name on huggingface.co/models")
print("  • Set RUN_ACTIVATION_TEST=True to also test actual forward pass (requires enough VRAM)")
