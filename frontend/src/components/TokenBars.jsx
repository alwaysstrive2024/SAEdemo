import { useMemo } from 'react';
import { Zap } from 'lucide-react';

/**
 * TokenBars
 * Ranked horizontal bar chart of the top-K tokens by peak SAE feature activation.
 *
 * Props:
 *   modelData  — full model data object
 *   modelColor — color tokens
 *   topK       — number of top tokens to display
 */
export default function TokenBars({ modelData, modelColor, topK }) {
  const tokenRows = useMemo(() => {
    const firings = modelData?.report_2_per_token?.token_level_firings ?? [];

    // For each token, find the max activation across its top features
    const rows = firings.map((t) => {
      const maxAct = t.top_50_features?.[0]?.activation ?? 0;
      const topLabel = t.top_50_features?.[0]?.concept_label ?? '—';
      return {
        token: t.token_string,
        index: t.token_index,
        maxAct,
        topLabel,
        featureId: t.top_50_features?.[0]?.feature_id ?? null,
      };
    });

    // Sort by maxAct descending, take topK
    return rows.sort((a, b) => b.maxAct - a.maxAct).slice(0, topK);
  }, [modelData, topK]);

  const globalMax = useMemo(
    () => Math.max(...tokenRows.map((r) => r.maxAct), 1),
    [tokenRows]
  );

  if (!tokenRows.length) {
    return (
      <div className="text-white/20 text-sm text-center py-6">No activation data</div>
    );
  }

  return (
    <div className="space-y-1.5">
      {tokenRows.map((row, i) => {
        const pct = (row.maxAct / globalMax) * 100;
        return (
          <div
            key={`${row.index}-${i}`}
            className="group flex items-center gap-2 animate-fade-up"
            style={{ animationDelay: `${i * 30}ms` }}
          >
            {/* Rank */}
            <span
              className="mono text-[9px] font-bold w-4 text-right flex-shrink-0"
              style={{ color: i === 0 ? modelColor.accent : 'rgba(255,255,255,0.25)' }}
            >
              {i + 1}
            </span>

            {/* Token chip */}
            <span
              className="mono text-[10px] font-semibold px-1.5 py-0.5 rounded-md flex-shrink-0 min-w-[40px] text-center"
              style={{
                background: i === 0 ? modelColor.bg : 'rgba(255,255,255,0.04)',
                border: `1px solid ${i === 0 ? modelColor.border : 'rgba(255,255,255,0.07)'}`,
                color: i === 0 ? modelColor.text : 'rgba(255,255,255,0.6)',
                maxWidth: '72px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={row.token}
            >
              {row.token}
            </span>

            {/* Bar */}
            <div
              className="relative flex-1 h-4 rounded-md overflow-hidden"
              style={{ background: 'rgba(255,255,255,0.04)' }}
            >
              {/* Fill */}
              <div
                className="absolute left-0 top-0 h-full rounded-md animate-progress-fill"
                style={{
                  '--fill-width': `${pct}%`,
                  width: `${pct}%`,
                  background: modelColor.gradientBar,
                  opacity: 0.75 + (pct / 100) * 0.25,
                  backgroundSize: '200% auto',
                }}
              />
              {/* Shimmer overlay on top bar */}
              {i === 0 && (
                <div
                  className="absolute inset-0 animate-shimmer opacity-30"
                  style={{
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
                    backgroundSize: '200% auto',
                  }}
                />
              )}
              {/* Activation value */}
              <span
                className="absolute right-2 top-1/2 -translate-y-1/2 mono text-[9px] font-semibold text-white/70"
              >
                {row.maxAct.toFixed(2)}
              </span>
            </div>

            {/* Top concept label (appears on hover) */}
            <span
              className="hidden group-hover:flex items-center gap-1 text-[9px] flex-shrink-0 max-w-[80px] truncate"
              style={{ color: modelColor.text, opacity: 0.85 }}
              title={row.topLabel}
            >
              <Zap size={8} />
              {row.topLabel}
            </span>
          </div>
        );
      })}
    </div>
  );
}
