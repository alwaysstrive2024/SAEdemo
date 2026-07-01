// ─── API ────────────────────────────────────────────────────────────────────
// Local dev    → http://localhost:8000  (direct to FastAPI)
// K8s / prod   → /api  (nginx in the frontend pod reverse-proxies /api/* to
//                       the internal sae-backend-svc K8s service)
export const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// Mirrors backend/registry.py so the selector can still show every registered
// model while the backend is starting or temporarily unavailable.
export const FALLBACK_MODELS = [
  {
    key: 'pythia-70m',
    display_name: 'Pythia 70M',
    sae_release: 'pythia-70m-deduped-res-jb',
    sae_id: 'res-jb',
  },
  {
    key: 'gemma-2b',
    display_name: 'Gemma 2B',
    sae_release: 'gemma-2b-res-jb',
    sae_id: 'blocks.12.hook_resid_post',
  },
  {
    key: 'llama-3.2-1b',
    display_name: 'Llama 3.2 1B',
    sae_release: 'llama-3.2-1b-res-jb',
    sae_id: 'blocks.8.hook_resid_post',
  },
  {
    key: 'gemma-4-e2b-l6',
    display_name: 'Gemma-4-E2B (Layer 6)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-e2b/btk-mat-layer-6-k-100',
  },
  {
    key: 'gemma-4-e2b-l17',
    display_name: 'Gemma-4-E2B (Layer 17)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-e2b/btk-mat-layer-17-k-100',
  },
  {
    key: 'gemma-4-e2b-l28',
    display_name: 'Gemma-4-E2B (Layer 28)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-e2b/btk-mat-layer-28-k-100',
  },
  {
    key: 'gemma-4-e4b-l7',
    display_name: 'Gemma-4-E4B (Layer 7)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-e4b/btk-mat-layer-7-k-100',
  },
  {
    key: 'gemma-4-e4b-l21',
    display_name: 'Gemma-4-E4B (Layer 21)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-e4b/btk-mat-layer-21-k-100',
  },
  {
    key: 'gemma-4-e4b-l35',
    display_name: 'Gemma-4-E4B (Layer 35)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-e4b/btk-mat-layer-35-k-100',
  },
  {
    key: 'gemma-4-31b-l30',
    display_name: 'Gemma-4-31B (Layer 30)',
    sae_release: 'decoderesearch/gemma-4-saes',
    sae_id: 'gemma-4-31b/btk-mat-layer-30-k-100',
  },
];

// ─── Per-model colour tokens (Slot-based) ────────────────────────────────────
export const SLOT_COLORS = [
  {
    // Slot 0: Purple / Indigo
    accent: '#a855f7',
    accentDark: '#7c3aed',
    gradient: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)',
    gradientBar: 'linear-gradient(90deg, #7c3aed, #a855f7, #c084fc)',
    glow: 'rgba(168, 85, 247, 0.35)',
    ring: 'rgba(168, 85, 247, 0.6)',
    bg: 'rgba(168, 85, 247, 0.07)',
    bgHover: 'rgba(168, 85, 247, 0.12)',
    text: '#c084fc',
    border: 'rgba(168, 85, 247, 0.4)',
    tag: 'TL+SAE',
  },
  {
    // Slot 1: Cyan / Emerald
    accent: '#06b6d4',
    accentDark: '#0284c7',
    gradient: 'linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)',
    gradientBar: 'linear-gradient(90deg, #0284c7, #06b6d4, #22d3ee)',
    glow: 'rgba(6, 182, 212, 0.35)',
    ring: 'rgba(6, 182, 212, 0.6)',
    bg: 'rgba(6, 182, 212, 0.07)',
    bgHover: 'rgba(6, 182, 212, 0.12)',
    text: '#22d3ee',
    border: 'rgba(6, 182, 212, 0.4)',
    tag: 'TL+SAE',
  },
  {
    // Slot 2: Emerald / Green
    accent: '#10b981',
    accentDark: '#059669',
    gradient: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
    gradientBar: 'linear-gradient(90deg, #059669, #10b981, #34d399)',
    glow: 'rgba(16, 185, 129, 0.35)',
    ring: 'rgba(16, 185, 129, 0.6)',
    bg: 'rgba(16, 185, 129, 0.07)',
    bgHover: 'rgba(16, 185, 129, 0.12)',
    text: '#34d399',
    border: 'rgba(16, 185, 129, 0.4)',
    tag: 'TL+SAE',
  },
];

export const DEFAULT_COLOR = {
  accent: '#6366f1',
  accentDark: '#4f46e5',
  gradient: 'linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)',
  gradientBar: 'linear-gradient(90deg, #4f46e5, #6366f1, #818cf8)',
  glow: 'rgba(99, 102, 241, 0.35)',
  ring: 'rgba(99, 102, 241, 0.6)',
  bg: 'rgba(99, 102, 241, 0.07)',
  bgHover: 'rgba(99, 102, 241, 0.12)',
  text: '#818cf8',
  border: 'rgba(99, 102, 241, 0.4)',
  tag: 'TL+SAE',
};

export function getModelColor(slotIndex) {
  if (typeof slotIndex !== 'number' || slotIndex < 0) return DEFAULT_COLOR;
  return SLOT_COLORS[slotIndex % SLOT_COLORS.length];
}

// ─── Loading overlay pipeline steps ──────────────────────────────────────────
// Each step belongs to a 'tool' (TransformerLens | SAELens | HOT-SWAP)
export const PIPELINE_STEPS = [
  {
    icon: '🔬',
    tool: 'TransformerLens',
    toolColor: '#a78bfa',
    text: (modelName) => `Loading ${modelName} via HookedTransformer.from_pretrained()…`,
  },
  {
    icon: '🪝',
    tool: 'TransformerLens',
    toolColor: '#a78bfa',
    text: () => 'Registering residual-stream hook at target layer…',
  },
  {
    icon: '▶️',
    tool: 'TransformerLens',
    toolColor: '#a78bfa',
    text: () => 'Running forward pass — intercepting activation tensor…',
  },
  {
    icon: '🔭',
    tool: 'SAELens',
    toolColor: '#34d399',
    text: () => 'Loading pre-trained SAE weights from HuggingFace…',
  },
  {
    icon: '✨',
    tool: 'SAELens',
    toolColor: '#34d399',
    text: () => 'Encoding dense activations → sparse interpretable features…',
  },
  {
    icon: '📊',
    tool: 'SAELens',
    toolColor: '#34d399',
    text: () => 'Building Report 1 (global) and Report 2 (per-token)…',
  },
  {
    icon: '🗑️',
    tool: 'HOT-SWAP',
    toolColor: '#fb923c',
    text: () => 'Deleting model objects — clearing VRAM for next model…',
  },
];
