import { useMemo, useState } from 'react';

/**
 * TextAttribution
 * Renders the input prompt as inline token spans, each highlighted in the
 * model's accent colour with opacity proportional to its peak SAE activation.
 *
 * Props:
 *   modelData  — full model data object
 *   modelColor — colour tokens from constants.js
 */
export default function TextAttribution({ modelData, modelColor }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  const tokenData = useMemo(() => {
    const firings = modelData?.report_2_per_token?.token_level_firings ?? [];
    if (!firings.length) return [];

    // Per-token peak activation
    const peaks = firings.map((t) => t.top_50_features?.[0]?.activation ?? 0);
    const maxPeak = Math.max(...peaks, 1e-9);

    return firings.map((t, i) => ({
      token: t.token_string,
      index: t.token_index,
      peak: peaks[i],
      norm: peaks[i] / maxPeak,
      topLabel: t.top_50_features?.[0]?.concept_label ?? '—',
      topFeatId: t.top_50_features?.[0]?.feature_id ?? null,
    }));
  }, [modelData]);

  if (!tokenData.length) {
    return <p className="text-white/20 text-sm">No token data</p>;
  }

  // Hex to rgba helper
  const hexToRgba = (hex, alpha) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  return (
    <div>
      {/* Token row */}
      <p
        className="text-[13px] leading-loose font-medium tracking-wide flex flex-wrap gap-y-1"
        aria-label="Token attribution map"
      >
        {tokenData.map((t) => {
          const isHovered = hoveredIdx === t.index;
          const alpha = 0.08 + t.norm * 0.72;
          const borderAlpha = 0.15 + t.norm * 0.65;

          return (
            <span
              key={t.index}
              className="relative inline-flex items-center px-1 py-0.5 rounded-md cursor-default mx-[1px]"
              style={{
                background: hexToRgba(modelColor.accent, isHovered ? Math.min(alpha + 0.2, 0.9) : alpha),
                borderBottom: `2px solid ${hexToRgba(modelColor.accent, isHovered ? 1 : borderAlpha)}`,
                color: t.norm > 0.55 ? 'white' : 'rgba(255,255,255,0.75)',
                fontWeight: t.norm > 0.7 ? '600' : '400',
                transition: 'background 0.15s ease, color 0.15s ease',
                boxShadow: isHovered ? `0 0 12px ${hexToRgba(modelColor.accent, 0.4)}` : 'none',
              }}
              onMouseEnter={() => setHoveredIdx(t.index)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {t.token}

              {/* Hover tooltip */}
              {isHovered && (
                <span
                  className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-20
                             flex flex-col items-center gap-0.5
                             px-2.5 py-1.5 rounded-lg text-[10px] whitespace-nowrap shadow-xl"
                  style={{
                    background: '#12141aee',
                    border: `1px solid ${hexToRgba(modelColor.accent, 0.45)}`,
                    boxShadow: `0 4px 20px ${hexToRgba(modelColor.accent, 0.2)}`,
                  }}
                >
                  <span className="mono font-bold text-white">
                    &quot;{t.token}&quot; · act {t.peak.toFixed(4)}
                  </span>
                  <span style={{ color: modelColor.text }}>
                    #{t.topFeatId} {t.topLabel}
                  </span>
                  {/* Activation heat bar */}
                  <div
                    className="w-full h-1 rounded-full mt-0.5"
                    style={{ background: 'rgba(255,255,255,0.1)' }}
                  >
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${t.norm * 100}%`,
                        background: modelColor.gradientBar,
                      }}
                    />
                  </div>
                </span>
              )}
            </span>
          );
        })}
      </p>

      {/* Intensity scale legend */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-[10px] text-white/25">SAE activation intensity:</span>
        <div
          className="flex-1 h-1.5 rounded-full max-w-[120px]"
          style={{
            background: `linear-gradient(90deg, ${hexToRgba(modelColor.accent, 0.08)}, ${modelColor.accent})`,
          }}
        />
        <span className="text-[10px] text-white/25">Low → High</span>
      </div>
    </div>
  );
}
