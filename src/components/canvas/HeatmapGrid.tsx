import React from 'react';

type Palette = 'shapley' | 'delta';

interface HeatmapGridProps {
  values: number[];
  palette?: Palette;
  size?: number;
  columns?: number;
  rows?: number;
  gap?: boolean;
}

function cellColor(value: number, palette: Palette): string {
  const abs = Math.min(Math.abs(value), 1);
  const alpha = 0.15 + abs * 0.85;
  if (palette === 'delta') {
    return value >= 0
      ? `rgba(34, 197, 94, ${alpha})`
      : `rgba(239, 68, 68, ${alpha})`;
  }
  return value >= 0
    ? `rgba(168, 85, 247, ${alpha})`
    : `rgba(20, 184, 166, ${alpha})`;
}

function gridColumns(count: number, columns?: number): string {
  if (columns) return `repeat(${columns}, 1fr)`;
  if (count === 48) return 'repeat(8, 1fr)';
  if (count <= 4) return `repeat(${count}, 1fr)`;
  return 'repeat(4, 1fr)';
}

function gridRows(count: number, rows?: number): string {
  if (rows) return `repeat(${rows}, 1fr)`;
  if (count === 48) return 'repeat(6, 1fr)';
  if (count <= 4) return '1fr';
  return 'repeat(4, 1fr)';
}

export const HeatmapGrid: React.FC<HeatmapGridProps> = ({
  values,
  palette = 'shapley',
  size = 80,
  columns,
  rows,
  gap,
}) => (
  <div
    className="overflow-hidden rounded-lg border border-default bg-surface-secondary"
    style={{
      width: size,
      height: size,
      display: 'grid',
      gridTemplateColumns: gridColumns(values.length, columns),
      gridTemplateRows: gridRows(values.length, rows),
      ...(gap ? { gap: '2px', padding: '2px' } : {}),
    }}
  >
    {values.map((v, idx) => (
      <div
        key={idx}
        className={gap ? 'rounded-sm flex items-center justify-center' : ''}
        style={{ backgroundColor: cellColor(v, palette) }}
        title={gap ? `N${idx}: ${v.toFixed(2)}` : undefined}
      >
        {gap && (
          <span className="text-[6px] text-default/70 dark:text-white/70 font-mono">
            {idx}
          </span>
        )}
      </div>
    ))}
  </div>
);
