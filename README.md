# SAE Feature Dashboard

Multi-model Sparse Autoencoder (SAE) feature visualization dashboard.  
Powered by **TransformerLens** + **SAELens** with HuggingFace fallback.

---

## Project Structure

```
SAEdemo/
├── backend/
│   ├── main.py          # FastAPI app + HTTP routes (thin layer)
│   ├── registry.py      # ★ Model registry — edit this to add new models
│   ├── pipeline.py      # Analysis engine: dual-path activation extraction
│   ├── config.py        # Constants (CONCEPT_LABELS) + shared utilities
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── constants.js         # API_BASE + per-model color tokens
│   │   ├── components/
│   │   │   ├── ControlPanel.jsx
│   │   │   ├── LoadingOverlay.jsx
│   │   │   ├── ModelColumn.jsx
│   │   │   ├── ConceptCluster.jsx
│   │   │   ├── FeatureBars.jsx
│   │   │   ├── TextAttribution.jsx
│   │   │   └── DivergenceSummary.jsx
│   │   └── utils/
│   │       └── download.js
│   ├── vite.config.js
│   └── package.json
└── README.md
```

---

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# → http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Adding a New Model

Edit **`backend/registry.py`** — the only file you need to touch:

```python
MODEL_REGISTRY = {
    # existing models ...

    "your-model-key": {
        "display_name": "Your Model Name",
        "sae_release":  "hf-org/repo-name",   # HuggingFace repo or named SAELens release
        "sae_id":       "path/to/sae-id",
    },
}
```

All other metadata (`d_model`, `hook_point`, `layer`, `hf_model_name`) is
**auto-extracted from `sae.cfg`** at runtime — no hardcoding required.

To find the correct `sae_release` / `sae_id`:
```python
from sae_lens import SAE
sae, _, _ = SAE.from_pretrained("<release>", "<sae_id>")
print(sae.cfg.hook_name, sae.cfg.d_in, sae.cfg.model_name)
```

After adding a model to `registry.py`, restart the backend. The new model
appears automatically in the frontend dropdown.

---

## Pipeline Architecture

```
POST /analyze
      │
      ▼
_resolve_model_config()      ← reads sae.cfg once, caches forever
      │
      ├─ Path A: TransformerLens HookedTransformer.run_with_hooks()
      └─ Path B: HuggingFace AutoModelForCausalLM + register_forward_hook()
                 (auto-fallback if TL doesn't support the model)
      │
      ▼
SAE.encode(residual_stream)  → sparse feature activations
      │
      ▼
Report 1: global feature summary (max/avg activation, fired token count)
Report 2: per-token top-50 feature firings
      │
      ▼
del model / del sae + torch.cuda.empty_cache()   ← VRAM hot-swap
```

---

## Hardware Requirements

| Models | Min VRAM | Recommended |
|---|---|---|
| Pythia 70M + 1B models | 8 GB | 16 GB |
| + Gemma 2B / E2B | 16 GB | 24 GB |
| + Gemma-4-E4B | 24 GB | A100 40 GB |
| + Gemma-4-31B | A100 80 GB | H100 80 GB |

**Software**: PyTorch ≥ 2.2, CUDA ≥ 12.1, Python ≥ 3.10
