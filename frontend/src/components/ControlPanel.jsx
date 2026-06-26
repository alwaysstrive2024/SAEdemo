import { useId } from 'react';
import { Plus, X, ChevronDown, Cpu, SlidersHorizontal, Play } from 'lucide-react';

/**
 * ControlPanel
 * Top section of the dashboard containing:
 *  - Prompt textarea
 *  - Dynamic model selector (up to 3, each with dropdown + remove)
 *  - Top-K slider
 *  - Run button
 *
 * Props:
 *   availableModels  {object[]}
 *   selectedModels   {object[]}  — [{id, modelKey}]
 *   setSelectedModels
 *   prompt / setPrompt
 *   topK / setTopK
 *   onRun / isLoading
 */
export default function ControlPanel({
  availableModels,
  selectedModels,
  setSelectedModels,
  prompt,
  setPrompt,
  topK,
  setTopK,
  onRun,
  isLoading,
}) {
  const baseId = useId();
  const MAX_MODELS = 3;

  // ── Handlers ───────────────────────────────────────────────────────────────
  const addModel = () => {
    if (selectedModels.length >= MAX_MODELS) return;
    setSelectedModels((prev) => [...prev, { id: `${baseId}-${Date.now()}`, modelKey: '' }]);
  };

  const removeModel = (id) => {
    setSelectedModels((prev) => prev.filter((m) => m.id !== id));
  };

  const updateModelKey = (id, modelKey) => {
    setSelectedModels((prev) => prev.map((m) => (m.id === id ? { ...m, modelKey } : m)));
  };

  // Which model keys are already chosen (for disabling duplicates in other selectors)
  const chosenKeys = selectedModels.map((m) => m.modelKey).filter(Boolean);
  const canRun =
    !isLoading &&
    prompt.trim().length > 0 &&
    chosenKeys.length > 0 &&
    selectedModels.every((m) => m.modelKey);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="glass-card p-6 space-y-5 animate-fade-up">
      {/* Section label */}
      <div className="flex items-center gap-2">
        <div
          className="w-1 h-5 rounded-full"
          style={{ background: 'linear-gradient(180deg, #6366f1, #a78bfa)' }}
        />
        <h2 className="text-sm font-semibold text-white/70 uppercase tracking-wider">
          Control Panel
        </h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-5">
        {/* ── Left: prompt + model selectors ── */}
        <div className="space-y-4">
          {/* Prompt textarea */}
          <div>
            <label
              htmlFor="prompt-input"
              className="block text-xs font-semibold text-white/50 mb-2 uppercase tracking-wider"
            >
              Input Prompt
            </label>
            <textarea
              id="prompt-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              className="input-base w-full px-4 py-3 text-sm resize-none leading-relaxed"
              placeholder="Enter a sentence to analyse… e.g. Apple announced iPhone 18 at WWDC."
            />
          </div>

          {/* Model selectors */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-white/50 uppercase tracking-wider flex items-center gap-1.5">
                <Cpu size={11} className="opacity-60" />
                Models to Compare
              </label>
              <button
                id="add-model-btn"
                onClick={addModel}
                disabled={selectedModels.length >= MAX_MODELS}
                className="btn-ghost text-xs flex items-center gap-1 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Plus size={12} />
                Add Model
                {selectedModels.length > 0 && (
                  <span
                    className="ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold"
                    style={{ background: 'rgba(99,102,241,0.2)', color: '#a5b4fc' }}
                  >
                    {selectedModels.length}/{MAX_MODELS}
                  </span>
                )}
              </button>
            </div>

            {/* Empty state hint */}
            {selectedModels.length === 0 && (
              <div
                className="flex items-center justify-center gap-2 rounded-xl py-4 text-sm text-white/25"
                style={{ border: '1px dashed rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.02)' }}
              >
                <Plus size={14} className="opacity-50" />
                Click &quot;Add Model&quot; to select up to 3 LLMs to compare
              </div>
            )}

            {/* Model selector rows */}
            <div className="space-y-2">
              {selectedModels.map((m, i) => (
                <ModelSelectorRow
                  key={m.id}
                  rowId={m.id}
                  index={i}
                  modelKey={m.modelKey}
                  availableModels={availableModels}
                  chosenKeys={chosenKeys}
                  onUpdate={updateModelKey}
                  onRemove={removeModel}
                />
              ))}
            </div>
          </div>
        </div>

        {/* ── Right: top-K + run button ── */}
        <div className="flex flex-col gap-4 lg:w-52">
          {/* Top-K control */}
          <div
            className="rounded-xl p-4 space-y-3"
            style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
          >
            <div className="flex items-center gap-1.5">
              <SlidersHorizontal size={11} className="text-white/40" />
              <span className="text-xs font-semibold text-white/50 uppercase tracking-wider">
                Top-K Features
              </span>
            </div>

            <div className="flex items-center gap-3">
              <input
                id="topk-slider"
                type="range"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="flex-1 accent-indigo-500"
                style={{ accentColor: '#6366f1' }}
              />
              <span
                className="mono text-sm font-bold w-8 text-center rounded-lg px-1 py-0.5"
                style={{ background: 'rgba(99,102,241,0.18)', color: '#a5b4fc' }}
              >
                {topK}
              </span>
            </div>

            <p className="text-[10px] text-white/25 leading-snug">
              Tokens shown per model column. Range 1–50.
            </p>
          </div>

          {/* Run button */}
          <button
            id="run-comparison-btn"
            onClick={onRun}
            disabled={!canRun}
            className="btn-primary flex-1 text-sm font-bold relative overflow-hidden group disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: canRun
                ? 'linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%)'
                : 'rgba(255,255,255,0.07)',
              color: 'white',
              boxShadow: canRun ? '0 0 24px rgba(99,102,241,0.4)' : 'none',
              minHeight: '52px',
            }}
          >
            {/* Shimmer on hover */}
            <span
              className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
              style={{
                background:
                  'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.12) 50%, transparent 100%)',
                backgroundSize: '200% auto',
              }}
            />
            <Play size={15} className="relative z-10" />
            <span className="relative z-10">Run Comparison</span>
          </button>

          {/* Pipeline mode legend */}
          <div className="text-[10px] text-white/25 text-center space-y-1">
            <div className="flex items-center justify-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />
              <span>TransformerLens</span>
            </div>
            <div className="flex items-center justify-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>SAELens</span>
            </div>
            <div className="flex items-center justify-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />
              <span>VRAM Hot-Swap</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Model selector row ────────────────────────────────────────────────────── */
const MODEL_ACCENT = {
  'pythia-70m': { color: '#a855f7', bg: 'rgba(168,85,247,0.12)' },
  'gemma-2b':   { color: '#06b6d4', bg: 'rgba(6,182,212,0.12)' },
  'llama-3.2-1b': { color: '#10b981', bg: 'rgba(16,185,129,0.12)' },
};

function ModelSelectorRow({ rowId, index, modelKey, availableModels, chosenKeys, onUpdate, onRemove }) {
  const accent = MODEL_ACCENT[modelKey] ?? { color: '#6366f1', bg: 'rgba(99,102,241,0.12)' };
  const labelColors = ['#a855f7', '#06b6d4', '#10b981'];

  return (
    <div
      className="flex items-center gap-2 animate-fade-up"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {/* Index badge */}
      <span
        className="mono text-[10px] font-bold w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0"
        style={{ background: `${labelColors[index]}22`, color: labelColors[index] }}
      >
        {index + 1}
      </span>

      {/* Dropdown */}
      <div className="relative flex-1">
        <select
          id={`model-select-${rowId}`}
          value={modelKey}
          onChange={(e) => onUpdate(rowId, e.target.value)}
          className="w-full appearance-none input-base px-3 py-2 pr-8 text-sm cursor-pointer"
          style={
            modelKey
              ? { borderColor: accent.color + '66', background: accent.bg, color: 'white' }
              : {}
          }
        >
          <option value="" disabled>
            Select a model…
          </option>
          {availableModels.map((m) => (
            <option
              key={m.key}
              value={m.key}
              disabled={chosenKeys.includes(m.key) && m.key !== modelKey}
              style={{ background: '#12141a', color: 'white' }}
            >
              {m.display_name ?? m.key}
            </option>
          ))}
        </select>
        <ChevronDown
          size={13}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-white/40 pointer-events-none"
        />
      </div>

      {/* Remove button */}
      <button
        id={`remove-model-${rowId}`}
        onClick={() => onRemove(rowId)}
        className="w-7 h-7 rounded-lg flex items-center justify-center text-white/30 hover:text-red-400 hover:bg-red-500/10 transition-all duration-150 flex-shrink-0"
        aria-label="Remove model"
      >
        <X size={13} />
      </button>
    </div>
  );
}
