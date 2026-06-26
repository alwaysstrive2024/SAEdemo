import { useState, useEffect, useCallback } from 'react';
import { Brain, Cpu, Zap, AlertCircle, RefreshCw } from 'lucide-react';

import { API_BASE, FALLBACK_MODELS, getModelColor } from './constants';
import ControlPanel from './components/ControlPanel';
import ModelColumn from './components/ModelColumn';
import DivergenceSummary from './components/DivergenceSummary';
import LoadingOverlay from './components/LoadingOverlay';

export default function App() {
  // ── State ────────────────────────────────────────────────────────────────
  const [availableModels, setAvailableModels] = useState(FALLBACK_MODELS);
  const [selectedModels, setSelectedModels] = useState([]); // [{id, modelKey}]
  const [prompt, setPrompt] = useState('Apple announced the iPhone 18 at WWDC.');
  const [topK, setTopK] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [pipelineMode, setPipelineMode] = useState(null); // 'real' | 'mock' | null

  // ── Fetch available models on mount ──────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then((r) => r.json())
      .then((data) => setAvailableModels(data.models?.length ? data.models : FALLBACK_MODELS))
      .catch(() => setAvailableModels(FALLBACK_MODELS));

    // Also ping root for pipeline mode
    fetch(`${API_BASE}/`, { credentials: 'include' })
      .then((r) => r.json())
      .then((d) => setPipelineMode(d.pipeline_mode))
      .catch(() => setPipelineMode('offline'));
  }, []);

  // ── Run analysis ─────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    const modelKeys = selectedModels.map((m) => m.modelKey).filter(Boolean);
    if (!modelKeys.length || !prompt.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          selected_models: modelKeys,
          top_k: topK,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
      setPipelineMode(data.metadata?.pipeline_mode ?? pipelineMode);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [selectedModels, prompt, topK, pipelineMode]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const activeModelKeys = selectedModels.map((m) => m.modelKey).filter(Boolean);
  const loadingModelKeys = isLoading ? activeModelKeys : [];

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col font-outfit">
      {/* Loading overlay */}
      {isLoading && (
        <LoadingOverlay
          models={loadingModelKeys}
          availableModels={availableModels}
        />
      )}

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-white/[0.06]"
        style={{ background: 'rgba(8,9,13,0.88)', backdropFilter: 'blur(16px)' }}
      >
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-3">
          {/* Logo */}
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: 'linear-gradient(135deg, #4f46e5, #7c3aed)' }}
          >
            <Brain size={16} className="text-white" />
          </div>

          <div className="leading-tight">
            <h1 className="text-sm font-bold text-white">SAE Feature Dashboard</h1>
            <p className="text-[10px] text-white/35">Multi-Model Mechanistic Interpretability</p>
          </div>

          {/* Pipeline badges */}
          <div className="ml-4 flex items-center gap-2 flex-wrap">
            <PipelineBadge label="TransformerLens" color="#a78bfa" />
            <span className="text-white/20 text-xs">+</span>
            <PipelineBadge label="SAELens" color="#34d399" />
            <span className="text-white/20 text-xs">|</span>
            <PipelineBadge label="VRAM Hot-Swap" color="#fb923c" />
          </div>

          {/* Backend status */}
          <div className="ml-auto flex items-center gap-2">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                pipelineMode === 'offline' ? 'bg-red-500' : 'bg-emerald-400 animate-pulse'
              }`}
            />
            <span className="text-[10px] text-white/35">
              {pipelineMode === null
                ? 'Connecting…'
                : pipelineMode === 'offline'
                ? 'Backend offline'
                : pipelineMode === 'real'
                ? '🔬 Real pipeline'
                : '🧪 Mock pipeline'}
            </span>
          </div>
        </div>
      </header>

      {/* ── Main ──────────────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-screen-2xl mx-auto w-full px-6 py-6 space-y-5">

        {/* Control panel */}
        <ControlPanel
          availableModels={availableModels}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          prompt={prompt}
          setPrompt={setPrompt}
          topK={topK}
          setTopK={setTopK}
          onRun={handleRun}
          isLoading={isLoading}
        />

        {/* Error banner */}
        {error && (
          <div
            className="flex items-start gap-3 p-4 rounded-xl border animate-fade-up"
            style={{ background: 'rgba(239,68,68,0.08)', borderColor: 'rgba(239,68,68,0.3)' }}
          >
            <AlertCircle size={16} className="text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-semibold text-red-300">Analysis failed</p>
              <p className="mono text-[10px] text-red-400/70 mt-0.5">{error}</p>
            </div>
            <button
              onClick={handleRun}
              className="flex items-center gap-1 text-[10px] text-red-300 hover:text-red-200"
            >
              <RefreshCw size={11} /> Retry
            </button>
          </div>
        )}

        {/* ── Results grid or empty state ─────────────────────────────────── */}
        {results ? (
          <>
            {/* Metadata bar */}
            <div
              className="flex items-center gap-3 px-4 py-2.5 rounded-xl text-[10px] text-white/40 animate-fade-up"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <Zap size={11} className="text-indigo-400 flex-shrink-0" />
              <span className="font-semibold text-white/60">Prompt:</span>
              <span className="italic truncate">&quot;{results.metadata?.prompt}&quot;</span>
              <span className="ml-auto flex-shrink-0">
                {activeModelKeys.length} model{activeModelKeys.length !== 1 ? 's' : ''} · Top-{topK}
              </span>
            </div>

            {/* Model columns */}
            <div
              className="grid gap-5"
              style={{
                gridTemplateColumns: `repeat(${activeModelKeys.length}, minmax(0, 1fr))`,
              }}
            >
              {activeModelKeys.map((key, index) => (
                <ModelColumn
                  key={key}
                  modelKey={key}
                  data={results.models_data?.[key]}
                  topK={topK}
                  modelColor={getModelColor(index)}
                />
              ))}
            </div>

            {/* Divergence summary (2+ models) */}
            {activeModelKeys.length >= 2 && (
              <DivergenceSummary results={results} />
            )}
          </>
        ) : (
          !isLoading && <EmptyState hasModels={activeModelKeys.length > 0} />
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-white/[0.05] py-4 px-6">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between text-[10px] text-white/20">
          <span>
            SAE Feature Dashboard · Powered by{' '}
            <span className="text-violet-400/60">TransformerLens</span> &{' '}
            <span className="text-emerald-400/60">SAELens</span>
          </span>
          <span className="flex items-center gap-1">
            <Cpu size={9} /> Sequential VRAM Hot-Swap Strategy
          </span>
        </div>
      </footer>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────────────────────────── */

function PipelineBadge({ label, color }) {
  return (
    <span
      className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[9px] font-semibold"
      style={{
        background: `${color}18`,
        border: `1px solid ${color}40`,
        color,
      }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

function EmptyState({ hasModels }) {
  return (
    <div
      className="flex flex-col items-center justify-center py-20 text-center animate-fade-up rounded-2xl"
      style={{ background: 'rgba(255,255,255,0.015)', border: '1px dashed rgba(255,255,255,0.07)' }}
    >
      <div
        className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
        style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.15), rgba(167,139,250,0.1))' }}
      >
        <Brain size={28} className="text-indigo-400 opacity-60" />
      </div>

      <h3 className="text-base font-bold text-white/50 mb-1">
        {hasModels ? 'Ready to analyse' : 'Select models to begin'}
      </h3>
      <p className="text-sm text-white/25 max-w-sm leading-relaxed">
        {hasModels
          ? 'Click "Run Comparison" to extract SAE features via TransformerLens & SAELens.'
          : 'Click "Add Model" in the control panel to select up to 3 LLMs, then hit Run Comparison.'}
      </p>

      {/* Pipeline diagram */}
      <div className="mt-8 flex items-center gap-3 text-[11px] text-white/25">
        <PipelineStep emoji="🔬" label="TransformerLens" sublabel="Hook activations" color="#a78bfa" />
        <Arrow />
        <PipelineStep emoji="✨" label="SAELens" sublabel="Encode features" color="#34d399" />
        <Arrow />
        <PipelineStep emoji="📊" label="Reports" sublabel="Global + Per-token" color="#60a5fa" />
        <Arrow />
        <PipelineStep emoji="🗑️" label="VRAM Clear" sublabel="Hot-swap" color="#fb923c" />
      </div>
    </div>
  );
}

function PipelineStep({ emoji, label, sublabel, color }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center text-lg"
        style={{ background: `${color}15`, border: `1px solid ${color}30` }}
      >
        {emoji}
      </div>
      <span style={{ color }} className="font-semibold text-[10px]">{label}</span>
      <span className="text-[9px] text-white/20">{sublabel}</span>
    </div>
  );
}

function Arrow() {
  return (
    <svg width="20" height="12" viewBox="0 0 20 12" fill="none" className="opacity-20 flex-shrink-0">
      <path d="M0 6h16M12 2l4 4-4 4" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
