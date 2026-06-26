import { Download, Layers, GitBranch, Activity } from 'lucide-react';
import ConceptCluster from './ConceptCluster';
import FeatureBars from './FeatureBars';
import TextAttribution from './TextAttribution';
import { downloadJSON } from '../utils/download';

/**
 * ModelColumn
 * A single model's analysis column containing:
 *  - Header (model name, layer badge, SAE info, download buttons)
 *  - Concept Cluster (SVG radial graph)
 *  - Token Firing Bars
 *  - Text Attribution map
 *
 * Props:
 *   modelKey   — string key from registry
 *   data       — models_data[modelKey] from API response
 *   topK       — integer, controls TokenBars display
 *   modelColor — colour tokens
 */
export default function ModelColumn({ modelKey, data, topK, modelColor }) {
  if (!data) {
    return (
      <div className="glass-card p-6 flex items-center justify-center text-white/20 text-sm">
        No data for {modelKey}
      </div>
    );
  }

  const meta = data.model_metadata;

  // ── Download handlers ─────────────────────────────────────────────────────
  const handleDownloadGlobal = () => {
    downloadJSON(
      { model_metadata: meta, ...data.report_1_global },
      `sae_report1_global_${modelKey}`
    );
  };

  const handleDownloadToken = () => {
    downloadJSON(
      { model_metadata: meta, ...data.report_2_per_token },
      `sae_report2_per_token_${modelKey}`
    );
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      className="glass-card overflow-hidden flex flex-col animate-fade-up"
      style={{
        borderTop: `2px solid ${modelColor.accent}`,
        boxShadow: `0 0 40px ${modelColor.glow}, 0 0 1px ${modelColor.accent}33`,
      }}
    >
      {/* ── Column Header ─────────────────────────────────────────────── */}
      <div
        className="px-5 pt-5 pb-4 border-b border-white/[0.06]"
        style={{ background: modelColor.bg }}
      >
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            {/* Model name */}
            <h3
              className="text-base font-bold leading-tight"
              style={{ color: modelColor.text }}
            >
              {meta.model_name}
            </h3>

            {/* Badges row */}
            <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
              <Badge icon={<Layers size={9} />} label={`Layer ${meta.layer}`} color={modelColor} />
              <Badge icon={<GitBranch size={9} />} label={meta.hook_point?.split('.').slice(-1)[0] ?? 'hook'} color={modelColor} />
              <Badge
                icon={<Activity size={9} />}
                label={`${data.report_1_global?.fired_features_summary?.length ?? 0} features`}
                color={modelColor}
              />
            </div>
          </div>

          {/* Pipeline mode pill */}
          <span
            className="mono text-[9px] px-2 py-1 rounded-lg font-semibold flex-shrink-0 whitespace-nowrap"
            style={{
              background: meta.pipeline_mode === 'real'
                ? 'rgba(52,211,153,0.15)'
                : 'rgba(251,146,60,0.12)',
              color: meta.pipeline_mode === 'real' ? '#34d399' : '#fb923c',
              border: `1px solid ${meta.pipeline_mode === 'real' ? 'rgba(52,211,153,0.3)' : 'rgba(251,146,60,0.25)'}`,
            }}
          >
            {meta.pipeline_mode === 'real' ? '🔬 REAL' : '🧪 MOCK'}
          </span>
        </div>

        {/* SAE info row */}
        <div className="mono text-[10px] text-white/30 flex flex-wrap gap-x-3 gap-y-0.5">
          <span>SAE: {meta.sae_release}</span>
          <span className="opacity-40">|</span>
          <span>d_model: {meta.d_model}</span>
        </div>

        {/* ── Download Buttons ──────────────────────────────────────── */}
        <div className="flex gap-2 mt-3">
          <DownloadButton
            id={`dl-global-${modelKey}`}
            label="Global Feature Report"
            onClick={handleDownloadGlobal}
            color={modelColor}
          />
          <DownloadButton
            id={`dl-token-${modelKey}`}
            label="Token Sequence Report"
            onClick={handleDownloadToken}
            color={modelColor}
          />
        </div>
      </div>

      {/* ── Concept Cluster ────────────────────────────────────────────── */}
      <div className="px-4 pt-4 pb-2">
        <SectionLabel
          icon={<span className="text-[10px]">🧠</span>}
          text="Concept Cluster"
          color={modelColor}
        />
        <div className="mt-2">
          <ConceptCluster modelData={data} modelColor={modelColor} />
        </div>
      </div>

      {/* ── Token Firing Bars ──────────────────────────────────────────── */}
      <div
        className="mx-4 mb-3 p-3 rounded-xl"
        style={{ background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(255,255,255,0.05)' }}
      >
        <SectionLabel
          icon={<span className="text-[10px]">⚡</span>}
          text={`Top-${topK} SAE Feature Firings`}
          color={modelColor}
        />
        <div className="mt-2.5">
          <FeatureBars modelData={data} modelColor={modelColor} topK={topK} />
        </div>
      </div>

      {/* ── Text Attribution ───────────────────────────────────────────── */}
      <div
        className="mx-4 mb-4 p-3 rounded-xl"
        style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)' }}
      >
        <SectionLabel
          icon={<span className="text-[10px]">🎨</span>}
          text="Token Attribution Map"
          color={modelColor}
        />
        <div className="mt-2.5">
          <TextAttribution modelData={data} modelColor={modelColor} />
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────────────────────────── */

function Badge({ icon, label, color }) {
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] font-semibold"
      style={{
        background: color.bg,
        border: `1px solid ${color.border}`,
        color: color.text,
      }}
    >
      {icon}
      {label}
    </span>
  );
}

function SectionLabel({ icon, text, color }) {
  return (
    <div className="flex items-center gap-1.5">
      {icon}
      <span
        className="text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: color.text, opacity: 0.8 }}
      >
        {text}
      </span>
      <div className="flex-1 h-px" style={{ background: `${color.accent}22` }} />
    </div>
  );
}

function DownloadButton({ id, label, onClick, color }) {
  return (
    <button
      id={id}
      onClick={onClick}
      className="flex-1 flex items-center justify-center gap-1.5 text-[10px] font-semibold
                 py-1.5 px-2 rounded-lg transition-all duration-150"
      style={{
        background: 'rgba(255,255,255,0.04)',
        border: `1px solid ${color.border}`,
        color: color.text,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = color.bgHover;
        e.currentTarget.style.boxShadow = `0 0 12px ${color.glow}`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <Download size={10} />
      <span className="truncate">{label}</span>
    </button>
  );
}
