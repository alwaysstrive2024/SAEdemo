// ─── API ────────────────────────────────────────────────────────────────────
// export const API_BASE = 'http://localhost:8000';
export const API_BASE = 'https://8000-gpu-g4-s-kkb-euw4a0-1kdtao9e835yx-a.europe-west4-0.prod.colab.dev';

// ─── Per-model colour tokens ─────────────────────────────────────────────────
export const MODEL_COLORS = {
  'pythia-70m': {
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
  'gemma-2b': {
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
  'llama-3.2-1b': {
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
};

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

export function getModelColor(modelKey) {
  return MODEL_COLORS[modelKey] ?? DEFAULT_COLOR;
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
