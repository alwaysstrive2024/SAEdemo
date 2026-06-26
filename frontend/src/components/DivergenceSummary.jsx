import { useMemo } from 'react';
import { getModelColor } from '../constants';

/**
 * DivergenceSummary
 * Bottom section comparing which SAE concepts are shared vs. unique across models.
 *
 * Props:
 *   results — full API response object
 */
export default function DivergenceSummary({ results }) {
  const { metadata, models_data } = results;
  const modelKeys = metadata?.selected_models ?? [];

  // For each model, build a Map<concept_label → max_activation> for global features
  const modelConceptMaps = useMemo(() => {
    return modelKeys.map((key) => {
      const features = models_data[key]?.report_1_global?.fired_features_summary ?? [];
      const map = new Map();
      for (const f of features) {
        const existing = map.get(f.concept_label);
        if (!existing || f.max_activation > existing) {
          map.set(f.concept_label, f.max_activation);
        }
      }
      return { key, map };
    });
  }, [results]); // eslint-disable-line

  // Collect all labels
  const allLabels = useMemo(() => {
    const set = new Set();
    modelConceptMaps.forEach(({ map }) => map.forEach((_, label) => set.add(label)));
    return [...set];
  }, [modelConceptMaps]);

  // Classify each label
  const { sharedAll, uniquePerModel } = useMemo(() => {
    const sharedAll = [];
    const uniquePerModel = modelKeys.map(() => []);

    for (const label of allLabels) {
      const presentIn = modelConceptMaps
        .map(({ map }, i) => ({ i, active: map.has(label), val: map.get(label) ?? 0 }))
        .filter((x) => x.active);

      if (presentIn.length === modelKeys.length) {
        // Shared by all
        const avgVal =
          presentIn.reduce((s, x) => s + x.val, 0) / presentIn.length;
        sharedAll.push({ label, avgVal });
      } else if (presentIn.length === 1) {
        // Unique to one model
        const { i, val } = presentIn[0];
        uniquePerModel[i].push({ label, val });
      }
      // If shared by some but not all — we skip for simplicity
    }

    // Sort by value desc, take top 8
    sharedAll.sort((a, b) => b.avgVal - a.avgVal);
    uniquePerModel.forEach((arr) => arr.sort((a, b) => b.val - a.val));

    return {
      sharedAll: sharedAll.slice(0, 8),
      uniquePerModel: uniquePerModel.map((arr) => arr.slice(0, 8)),
    };
  }, [modelConceptMaps, allLabels, modelKeys]);

  if (modelKeys.length < 2) return null;

  return (
    <div className="glass-card p-6 animate-fade-up">
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div
          className="w-1 h-5 rounded-full"
          style={{ background: 'linear-gradient(180deg, #f59e0b, #ef4444, #8b5cf6)' }}
        />
        <div>
          <h2 className="text-sm font-bold text-white">Concept Divergence Summary</h2>
          <p className="text-[10px] text-white/35 mt-0.5">
            Comparing SAE-extracted concepts across {modelKeys.length} models
          </p>
        </div>
        <div className="ml-auto mono text-[10px] text-white/25">
          {allLabels.length} total unique concepts
        </div>
      </div>

      <div
        className="grid gap-4"
        style={{ gridTemplateColumns: `repeat(${modelKeys.length + 1}, 1fr)` }}
      >
        {/* ── Per-model unique columns ── */}
        {modelKeys.map((key, i) => {
          const color = getModelColor(i);
          const modelName = results.models_data[key]?.model_metadata?.model_name ?? key;
          const items = uniquePerModel[i];

          return (
            <div
              key={key}
              className="rounded-xl p-4"
              style={{ background: color.bg, border: `1px solid ${color.border}` }}
            >
              <div className="flex items-center gap-1.5 mb-3">
                <span
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: color.accent }}
                />
                <span className="text-[11px] font-bold" style={{ color: color.text }}>
                  Only in {modelName}
                </span>
              </div>

              {items.length === 0 ? (
                <p className="text-[10px] text-white/20">No unique concepts</p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {items.map(({ label, val }) => (
                    <ConceptTag key={label} label={label} value={val} color={color} />
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {/* ── Shared by all ── */}
        <div
          className="rounded-xl p-4"
          style={{
            background: 'rgba(99,102,241,0.07)',
            border: '1px solid rgba(99,102,241,0.25)',
          }}
        >
          <div className="flex items-center gap-1.5 mb-3">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: 'linear-gradient(135deg, #6366f1, #a78bfa)' }}
            />
            <span className="text-[11px] font-bold text-indigo-300">
              Shared by All Models
            </span>
          </div>

          {sharedAll.length === 0 ? (
            <p className="text-[10px] text-white/20">No shared concepts</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {sharedAll.map(({ label, avgVal }) => (
                <ConceptTag
                  key={label}
                  label={label}
                  value={avgVal}
                  color={{
                    accent: '#6366f1',
                    text: '#a5b4fc',
                    bg: 'rgba(99,102,241,0.18)',
                    border: 'rgba(99,102,241,0.4)',
                    gradientBar: 'linear-gradient(90deg,#4f46e5,#6366f1,#a78bfa)',
                  }}
                  shared
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Concept tag pill ────────────────────────────────────────────────────────── */
function ConceptTag({ label, value, color, shared }) {
  return (
    <div
      className="flex items-center gap-1 px-2 py-1 rounded-lg cursor-default
                 transition-all duration-150 group"
      style={{
        background: color.bg,
        border: `1px solid ${color.border}`,
      }}
      title={`Activation: ${value.toFixed(3)}`}
      onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-1px)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}
    >
      {shared && (
        <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ background: color.accent }} />
      )}
      <span
        className="text-[10px] font-medium leading-tight"
        style={{ color: color.text }}
      >
        {label}
      </span>
      <span
        className="mono text-[9px] opacity-0 group-hover:opacity-60 transition-opacity"
        style={{ color: color.text }}
      >
        {value.toFixed(1)}
      </span>
    </div>
  );
}
