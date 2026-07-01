"""
test_backend.py — 后端全链路诊断脚本
=====================================
与现有代码完全共享同一套逻辑：
  - 用 main.py 同款的依赖检测 & set_deps() 初始化 pipeline
  - 用 pipeline._resolve_model_config() 加载真实 sae.cfg
  - 用 pipeline._extract_activations_tl() / _extract_activations_hf() 提取激活
  - 用 pipeline._real_analyse_model() 跑完整分析流程

在运行实际代码之前，先做两类前置检查：
  1. HuggingFace Hub 确认 hf_model_name 真实存在
  2. TransformerLens 确认是否支持该模型名（不下载权重）

用法 (Colab):
  !pip install sae-lens transformer-lens transformers accelerate huggingface_hub torch -q
  # 将 backend/ 整个目录上传，或把所有 .py 放同一目录，然后：
  !python test_backend.py

注意：这里不设置 HF_ENDPOINT 镜像，因为 Colab 可以直接访问 HuggingFace。
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import Any, Dict, List, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# ① 配置：选择要测试的模型和测试 prompt
# ══════════════════════════════════════════════════════════════════════════════

# 要测试的模型 key（必须和 registry.py 中的 key 完全一致）
MODELS_TO_TEST: List[str] = [
    "gemma-2b",
    "pythia",
    "gpt2-small-l4",
    "llama-3.2-1b",
    "gemma-4-e2b-l6",
    "gemma-4-e2b-l17",
    "gemma-4-e2b-l28",
    "gemma-4-e4b-l7",
    "gemma-4-e4b-l21",
    "gemma-4-e4b-l35",
    "gemma-4-31b-l30",
    "olmo-3-7b-l4",
    "deepseek-r1",
    "qwen-3.5-0.8b",
]

TEST_PROMPT = "Apple announced the iPhone 18 at WWDC."

# 测试阶段开关
CHECK_HF_HUB      = True   # 预检：hf_model_name 在 HF Hub 上是否存在
CHECK_TL_NAME     = True   # 预检：TransformerLens 是否认识该模型名
RUN_CONFIG        = True   # 跑 _resolve_model_config()（加载 SAE 读取 cfg）
RUN_ACTIVATION    = True   # 跑 _extract_activations_*()（加载 LLM 提取激活）
RUN_SAE_ENCODE    = False  # 跑 SAE encode（feature_acts）
RUN_FULL_PIPELINE = False  # 跑完整 _real_analyse_model()（含 report 生成）

# ══════════════════════════════════════════════════════════════════════════════
# ② 覆盖 config.py 里的镜像源（Colab 不需要镜像）
# ══════════════════════════════════════════════════════════════════════════════
# 在 import config 之前就覆盖，确保生效
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# ══════════════════════════════════════════════════════════════════════════════
# ③ 颜色 & 格式辅助
# ══════════════════════════════════════════════════════════════════════════════
GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET  = "\033[0m"
DIMMED = "\033[2m"

def _ok(msg: str):   print(f"  {GREEN}✅ {msg}{RESET}")
def _fail(msg: str): print(f"  {RED}❌ {msg}{RESET}")
def _warn(msg: str): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def _info(msg: str): print(f"  {CYAN}   {msg}{RESET}")
def _dim(msg: str):  print(f"{DIMMED}{msg}{RESET}")
def _sep(title: str = ""):
    if title:
        pad = "═" * max(0, 68 - len(title) - 2)
        print(f"\n{BOLD}╔══ {title} {pad}╗{RESET}")
    else:
        print("─" * 72)

# ══════════════════════════════════════════════════════════════════════════════
# ④ 完全复用 main.py 的依赖检测逻辑
# ══════════════════════════════════════════════════════════════════════════════
_sep("Environment Detection (same as main.py)")

def _detect(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False

TORCH_AVAILABLE = _detect("torch")
TL_AVAILABLE    = _detect("transformer_lens")
SAE_AVAILABLE   = _detect("sae_lens")
HF_AVAILABLE    = _detect("transformers")
HUB_AVAILABLE   = _detect("huggingface_hub")

print(f"  PyTorch        : {'✅' if TORCH_AVAILABLE else '❌ MISSING'}")
print(f"  TransformerLens: {'✅' if TL_AVAILABLE else '❌ MISSING'}")
print(f"  SAELens        : {'✅' if SAE_AVAILABLE else '❌ MISSING'}")
print(f"  HF Transformers: {'✅' if HF_AVAILABLE else '❌ MISSING'}")
print(f"  HF Hub         : {'✅' if HUB_AVAILABLE else '❌ (pip install huggingface_hub)'}")

if not TORCH_AVAILABLE:
    print(f"\n{RED}Fatal: PyTorch not installed. Run: pip install torch{RESET}")
    sys.exit(1)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
_ok(f"PyTorch {torch.__version__} | device={device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    _info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1e9:.1f} GB")

# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 导入后端模块（与 main.py 完全相同，且按相同顺序）
# ══════════════════════════════════════════════════════════════════════════════
_sep("Importing Backend Modules")

try:
    import pipeline as _pipeline
    _ok("pipeline.py imported")
except ImportError as e:
    _fail(f"Cannot import pipeline.py: {e}")
    _info("Make sure test_backend.py is in the same directory as pipeline.py")
    sys.exit(1)

try:
    from config import vram_clear, safe_sae_attr
    _ok("config.py imported")
except ImportError as e:
    _fail(f"Cannot import config.py: {e}")
    sys.exit(1)

try:
    from registry import MODEL_REGISTRY
    _ok(f"registry.py imported — {len(MODEL_REGISTRY)} models registered")
except ImportError as e:
    _fail(f"Cannot import registry.py: {e}")
    sys.exit(1)

# 用与 main.py 完全相同的方式初始化 pipeline 依赖标志
_pipeline.set_deps(
    tl       = TL_AVAILABLE,
    sae      = SAE_AVAILABLE,
    hf       = HF_AVAILABLE,
    torch_ok = TORCH_AVAILABLE,
)
_info(f"pipeline.FULL_PIPELINE = {_pipeline.FULL_PIPELINE}")
_info(f"pipeline.TL_AVAILABLE  = {_pipeline.TL_AVAILABLE}")
_info(f"pipeline.SAE_AVAILABLE = {_pipeline.SAE_AVAILABLE}")
_info(f"pipeline.HF_AVAILABLE  = {_pipeline.HF_AVAILABLE}")

if not _pipeline.FULL_PIPELINE:
    _warn("FULL_PIPELINE=False — at least one of [torch + sae + (tl or hf)] is missing")
    _warn("Real pipeline will NOT run. Install missing packages first.")


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ 结果收集
# ══════════════════════════════════════════════════════════════════════════════
class TestResult:
    def __init__(self, model_key: str):
        self.model_key       = model_key
        self.registry_ok     = False
        self.hf_hub_ok       = False
        self.tl_name_ok      = False
        self.config_ok       = False
        self.activation_path = ""      # "transformer_lens" / "huggingface_hook" / ""
        self.activation_ok   = False
        self.sae_encode_ok   = False
        self.full_ok         = False
        self.errors: List[Tuple[str, str]] = []   # [(stage, message)]

    def add_error(self, stage: str, msg: str):
        self.errors.append((stage, msg))
        _fail(f"[{stage}] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ 单模型测试函数
# ══════════════════════════════════════════════════════════════════════════════

def run_test(model_key: str) -> TestResult:
    r = TestResult(model_key)
    print(f"\n\n{BOLD}{'╔' + '═'*68 + '╗'}")
    print(f"║  Testing: {model_key:<57}║")
    print(f"{'╚' + '═'*68 + '╝'}{RESET}")

    # ── A. Registry 检查 ──────────────────────────────────────────────────────
    print(f"\n{BOLD}[A] Registry{RESET}")
    if model_key not in MODEL_REGISTRY:
        r.add_error("Registry", f"Key '{model_key}' not found in MODEL_REGISTRY")
        return r

    reg  = MODEL_REGISTRY[model_key]
    r.registry_ok = True
    _ok("Key found")

    required = ["sae_release", "sae_id"]
    recommended = ["hf_model_name", "hook_point", "hook_layer"]
    for field in required:
        if not reg.get(field):
            r.add_error("Registry", f"Required field '{field}' is missing or empty")
        else:
            _info(f"{field:20s} = {reg[field]}")

    for field in recommended:
        val = reg.get(field)
        if val is None:
            _warn(f"Recommended field '{field}' is not set — will fall back to sae.cfg (unreliable)")
        else:
            _info(f"{field:20s} = {val}")

    hf_name   = reg.get("hf_model_name") or model_key
    hook_pt   = reg.get("hook_point", "")
    hook_layer = reg.get("hook_layer", 0)

    if r.errors:
        return r  # Registry errors are fatal

    # ── B. 预检 1: HuggingFace Hub 模型是否存在 ───────────────────────────────
    print(f"\n{BOLD}[B] Pre-check: HF Hub — '{hf_name}'{RESET}")
    if not CHECK_HF_HUB:
        _warn("Skipped (CHECK_HF_HUB=False)")
    elif not HUB_AVAILABLE:
        _warn("Skipped — huggingface_hub not installed")
    else:
        try:
            from huggingface_hub import model_info
            mi = model_info(hf_name, timeout=15)
            r.hf_hub_ok = True
            _ok(f"Model exists on HF Hub: {mi.id}")
            _info(f"Downloads: {mi.downloads or 'N/A'} | Likes: {mi.likes or 'N/A'}")
        except Exception as e:
            err_msg = str(e)
            r.add_error("HF Hub", f"'{hf_name}' not found or not accessible: {err_msg}")
            _info("Common causes:")
            _info("  • Wrong model name (check huggingface.co/models)")
            _info("  • Model requires login token (use: huggingface-cli login)")
            _info("  • Network issue (check internet connectivity)")

    # ── C. 预检 2: TransformerLens 模型名认知 ────────────────────────────────
    print(f"\n{BOLD}[C] Pre-check: TransformerLens name recognition — '{hf_name}'{RESET}")
    if not CHECK_TL_NAME:
        _warn("Skipped (CHECK_TL_NAME=False)")
    elif not TL_AVAILABLE:
        _warn("Skipped — TransformerLens not installed")
    else:
        try:
            from transformer_lens import HookedTransformer
            tl_cfg = HookedTransformer.get_pretrained_model_config(hf_name)
            r.tl_name_ok = True
            _ok(f"TransformerLens recognizes '{hf_name}'")
            _info(f"TL cfg: d_model={tl_cfg.d_model}, n_layers={tl_cfg.n_layers}, n_heads={tl_cfg.n_heads}")
        except Exception as e:
            r.tl_name_ok = False
            _warn(f"TransformerLens does NOT recognize '{hf_name}': {type(e).__name__}: {e}")
            _warn("Path A (TransformerLens) will be SKIPPED at runtime → Path B (HuggingFace) will be used")

    # ── D. _resolve_model_config()（与 pipeline.py 完全相同的调用） ────────────
    print(f"\n{BOLD}[D] _resolve_model_config() — reads SAE, extracts cfg{RESET}")
    if not RUN_CONFIG:
        _warn("Skipped (RUN_CONFIG=False)")
        cfg = None
    elif not SAE_AVAILABLE:
        r.add_error("Config", "SAELens not available — cannot resolve config")
        cfg = None
    else:
        try:
            cfg = _pipeline._resolve_model_config(model_key)
            r.config_ok = True
            _ok("Config resolved successfully")
            _info(f"hf_model_name : {cfg['hf_model_name']}")
            _info(f"hook_point    : {cfg['hook_point']}")
            _info(f"hook_layer    : {cfg['layer']}")
            _info(f"d_model       : {cfg['d_model']}")
            _info(f"sae_release   : {cfg['sae_release']}")
            _info(f"sae_id        : {cfg['sae_id']}")

            # 精准比对：cfg 最终值 vs registry 原始值
            print(f"\n  {BOLD}Final resolved values vs registry:{RESET}")
            def _cmp(label, resolved, registered):
                if registered is None:
                    print(f"    {YELLOW}◎ {label}: resolved='{resolved}' | registry=NOT SET (using sae.cfg){RESET}")
                elif str(resolved) == str(registered):
                    print(f"    {GREEN}✅ {label}: '{resolved}'{RESET}")
                else:
                    print(f"    {RED}⚠️  {label}: resolved='{resolved}' ≠ registry='{registered}'{RESET}")
                    r.add_error("ConfigMismatch",
                                f"{label}: resolved='{resolved}' but registry has '{registered}'")

            _cmp("hf_model_name", cfg["hf_model_name"],  reg.get("hf_model_name"))
            _cmp("hook_point",    cfg["hook_point"],      reg.get("hook_point"))
            _cmp("hook_layer",    cfg["layer"],           reg.get("hook_layer"))

        except Exception as e:
            r.add_error("Config", f"{type(e).__name__}: {e}")
            _info("Full traceback:")
            traceback.print_exc()
            cfg = None

    # ── E. Activation extraction（直接调用 pipeline 内部函数）─────────────────
    print(f"\n{BOLD}[E] Activation extraction — same code as _real_analyse_model(){RESET}")
    resid = None
    token_strs = []

    if not RUN_ACTIVATION:
        _warn("Skipped (RUN_ACTIVATION=False)")
    elif cfg is None:
        _warn("Skipped — config not resolved (see step D)")
    elif not _pipeline.FULL_PIPELINE:
        _warn("Skipped — FULL_PIPELINE=False")
    else:
        # Path A: TransformerLens
        if _pipeline.TL_AVAILABLE:
            print(f"\n  {BOLD}Path A — _extract_activations_tl('{cfg['hf_model_name']}', '{cfg['hook_point']}'){RESET}")
            try:
                resid, token_strs = _pipeline._extract_activations_tl(
                    cfg["hf_model_name"], cfg["hook_point"], TEST_PROMPT, device
                )
                r.activation_path = "transformer_lens"
                r.activation_ok   = True
                _ok(f"✅ Path A SUCCESS — resid shape={tuple(resid.shape)}, tokens={token_strs}")
            except Exception as e:
                _warn(f"Path A FAILED: {type(e).__name__}: {e}")
                _dim(traceback.format_exc())
                r.add_error("PathA", f"{type(e).__name__}: {e}")

        # Path B: HuggingFace (fallback or sole path)
        if resid is None:
            if not _pipeline.HF_AVAILABLE:
                r.add_error("PathB", "HF Transformers not available; both paths failed")
            else:
                print(f"\n  {BOLD}Path B — _extract_activations_hf('{cfg['hf_model_name']}', layer={cfg['layer']}){RESET}")
                try:
                    resid, token_strs = _pipeline._extract_activations_hf(
                        cfg["hf_model_name"], cfg["layer"], TEST_PROMPT, device
                    )
                    r.activation_path = "huggingface_hook"
                    r.activation_ok   = True
                    _ok(f"✅ Path B SUCCESS — resid shape={tuple(resid.shape)}, tokens={token_strs}")
                except Exception as e:
                    r.add_error("PathB", f"{type(e).__name__}: {e}")
                    _dim(traceback.format_exc())

    # ── F. SAE encode ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}[F] SAE encode (sae.encode(resid)){RESET}")
    if not RUN_SAE_ENCODE:
        _warn("Skipped (RUN_SAE_ENCODE=False)")
    elif resid is None:
        _warn("Skipped — no activation tensor (see step E)")
    elif cfg is None:
        _warn("Skipped — config not available")
    else:
        try:
            from sae_lens import SAE as _SAE
            print(f"  Loading SAE: release='{cfg['sae_release']}' sae_id='{cfg['sae_id']}'")
            sae, _, _ = _SAE.from_pretrained(
                release=cfg["sae_release"],
                sae_id=cfg["sae_id"],
                device=device,
            )
            sae.eval()
            d_sae = safe_sae_attr(sae.cfg, "d_sae", default=None)
            _info(f"SAE loaded: d_sae={d_sae}")

            import torch
            with torch.no_grad():
                feature_acts = sae.encode(resid.to(device))

            fired = (feature_acts > 0).float()
            l0    = fired.sum(dim=-1).mean().item()
            sparsity = fired.mean().item()
            _ok(f"Encode success — shape={tuple(feature_acts.shape)}, mean L0={l0:.1f}, sparsity={sparsity:.4f}")
            r.sae_encode_ok = True

            del sae, feature_acts
            vram_clear()

        except Exception as e:
            r.add_error("SAEEncode", f"{type(e).__name__}: {e}")
            _dim(traceback.format_exc())

    # ── G. Full pipeline（可选）───────────────────────────────────────────────
    print(f"\n{BOLD}[G] Full pipeline — _real_analyse_model(){RESET}")
    if not RUN_FULL_PIPELINE:
        _warn("Skipped (RUN_FULL_PIPELINE=False)")
    elif not _pipeline.FULL_PIPELINE:
        _warn("Skipped — FULL_PIPELINE=False")
    else:
        try:
            out = _pipeline._real_analyse_model(TEST_PROMPT, model_key, top_k=5)
            r.full_ok = True
            meta = out["model_metadata"]
            fired = out["report_1_global"]["fired_features_summary"]
            _ok(f"Full pipeline success")
            _info(f"activation_path : {meta['activation_path']}")
            _info(f"tokens          : {meta['tokens']}")
            _info(f"features fired  : {len(fired)}")
            vram_clear()
        except Exception as e:
            r.add_error("FullPipeline", f"{type(e).__name__}: {e}")
            _dim(traceback.format_exc())

    return r


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ 运行所有测试 & 打印汇总表
# ══════════════════════════════════════════════════════════════════════════════

all_results: List[TestResult] = []
for key in MODELS_TO_TEST:
    all_results.append(run_test(key))
    vram_clear()  # 每个模型测完清一次 VRAM

print(f"\n\n{BOLD}{'╔' + '═'*100 + '╗'}")
print(f"║{'SUMMARY':^100}║")
print(f"{'╚' + '═'*100 + '╝'}{RESET}")

col = f"{'Model Key':<32} {'Reg':^5} {'Hub':^5} {'TL?':^5} {'Cfg':^5} {'Act':^5} {'path':^16} {'SAE':^5} {'Full':^5}  Errors"
print(f"\n{BOLD}{col}{RESET}")
print("─" * 110)

for r in all_results:
    def S(b): return f"{GREEN}✅{RESET}" if b else f"{RED}❌{RESET}"
    path_str = r.activation_path[:14] if r.activation_path else "—"
    err_str  = " | ".join(f"{s}: {m[:30]}" for s, m in r.errors)[:50] if r.errors else ""
    print(
        f"{r.model_key:<32} "
        f"{S(r.registry_ok):^5} {S(r.hf_hub_ok):^5} {S(r.tl_name_ok):^5} "
        f"{S(r.config_ok):^5} {S(r.activation_ok):^5} {path_str:^16} "
        f"{S(r.sae_encode_ok):^5} {S(r.full_ok):^5}  "
        f"{RED}{err_str}{RESET}"
    )

print(f"\n{BOLD}Column guide:{RESET}")
print("  Reg  = registry key & required fields OK")
print("  Hub  = hf_model_name exists on HuggingFace Hub")
print("  TL?  = TransformerLens recognizes the model name (Path A)")
print("  Cfg  = _resolve_model_config() success (loads SAE to read cfg)")
print("  Act  = activation extraction success (Path A or B)")
print("  SAE  = sae.encode(resid) success")
print("  Full = complete _real_analyse_model() success")
