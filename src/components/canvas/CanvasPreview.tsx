import React from 'react';
import { HeatmapGrid } from './HeatmapGrid';

const PREVIEW_SIZE = 80;

interface CanvasPreviewProps {
  previewRef: React.RefObject<HTMLCanvasElement | null>;
  previewHeatmap?: number[];
  neuronHeatmap?: number[];
  beforeHeatmap?: number[];
  afterHeatmap?: number[];
}

const styles = {
  container: 'flex flex-col items-center gap-3',
  row: 'flex max-w-full flex-wrap items-start justify-center gap-2 sm:gap-4',
  cell: 'flex shrink-0 flex-col items-center gap-1',
  label: 'text-xs text-muted uppercase tracking-wider sm:text-sm',
  previewWrap: 'rounded-lg overflow-hidden border border-default bg-white',
  legend: 'text-center text-sm text-muted sm:text-xs',
  legendRow:
    'mt-1 flex flex-nowrap items-center justify-center gap-1.5 text-xs sm:gap-4 sm:text-inherit',
  legendItem: 'flex shrink-0 items-center gap-0.5 whitespace-nowrap sm:gap-1',
  legendDot: 'inline-block h-2.5 w-2.5 shrink-0 rounded sm:h-3 sm:w-3',
  divider: 'shrink-0 text-muted',
  hint: 'whitespace-nowrap text-center text-xs text-muted sm:text-sm',
} as const;

const LABEL_NAMES: Record<number, string> = {
  48: 'Channels (48)',
  4: 'Channels (4)',
  3: 'Channels (3)',
  2: 'Channels (2)',
};

function neuronLabel(count: number): string {
  return LABEL_NAMES[count] ?? 'Neurons (16)';
}

export const CanvasPreview: React.FC<CanvasPreviewProps> = ({
  previewRef,
  previewHeatmap,
  neuronHeatmap,
  beforeHeatmap,
  afterHeatmap,
}) => {
  const isTrainingMode = beforeHeatmap && beforeHeatmap.length === 64;

  return (
    <div className={styles.container}>
      {!isTrainingMode && (
        <p className={styles.hint}>
          Draw digit 0-9 · fill canvas for best results
        </p>
      )}
      <div className={styles.row}>
        <div className={styles.cell}>
          <span className={styles.label}>Input</span>
          <div className={styles.previewWrap}>
            <canvas
              ref={previewRef}
              width={8}
              height={8}
              className="block"
              style={{
                width: PREVIEW_SIZE,
                height: PREVIEW_SIZE,
                imageRendering: 'pixelated',
              }}
            />
          </div>
        </div>
        {isTrainingMode ? (
          <>
            <div className={styles.cell}>
              <span className={styles.label}>Before</span>
              <HeatmapGrid
                values={beforeHeatmap}
                palette="shapley"
                size={PREVIEW_SIZE}
                columns={8}
                rows={8}
              />
            </div>
            {afterHeatmap && afterHeatmap.length === 64 && (
              <>
                <div className={styles.cell}>
                  <span className={styles.label}>After</span>
                  <HeatmapGrid
                    values={afterHeatmap}
                    palette="shapley"
                    size={PREVIEW_SIZE}
                    columns={8}
                    rows={8}
                  />
                </div>
                <div className={styles.cell}>
                  <span className={styles.label}>Delta</span>
                  <HeatmapGrid
                    values={beforeHeatmap.map(
                      (b, i) => (afterHeatmap[i] ?? 0) - b
                    )}
                    palette="delta"
                    size={PREVIEW_SIZE}
                    columns={8}
                    rows={8}
                  />
                </div>
              </>
            )}
          </>
        ) : (
          <>
            {previewHeatmap && previewHeatmap.length === 64 && (
              <div className={styles.cell}>
                <span className={styles.label}>Input Attribution</span>
                <HeatmapGrid
                  values={previewHeatmap}
                  size={PREVIEW_SIZE}
                  columns={8}
                  rows={8}
                />
              </div>
            )}
            {neuronHeatmap && neuronHeatmap.length > 0 && (
              <div className={styles.cell}>
                <span className={styles.label}>
                  {neuronLabel(neuronHeatmap.length)}
                </span>
                <HeatmapGrid values={neuronHeatmap} size={PREVIEW_SIZE} gap />
              </div>
            )}
          </>
        )}
      </div>
      {isTrainingMode ? (
        <div className={styles.legend}>
          <p>Training impact visualization</p>
          <div className={styles.legendRow}>
            <div className={styles.legendItem}>
              <span
                className={styles.legendDot}
                style={{ backgroundColor: 'rgba(168, 85, 247, 0.85)' }}
              />
              <span>
                <span className="sm:hidden">+</span>
                <span className="hidden sm:inline">+contrib</span>
              </span>
            </div>
            <div className={styles.legendItem}>
              <span
                className={styles.legendDot}
                style={{ backgroundColor: 'rgba(20, 184, 166, 0.85)' }}
              />
              <span>
                <span className="sm:hidden">−</span>
                <span className="hidden sm:inline">-contrib</span>
              </span>
            </div>
            {afterHeatmap && (
              <>
                <span className={styles.divider}>|</span>
                <div className={styles.legendItem}>
                  <span
                    className={styles.legendDot}
                    style={{ backgroundColor: 'rgba(34, 197, 94, 0.85)' }}
                  />
                  <span>
                    <span className="sm:hidden">↑</span>
                    <span className="hidden sm:inline">increased</span>
                  </span>
                </div>
                <div className={styles.legendItem}>
                  <span
                    className={styles.legendDot}
                    style={{ backgroundColor: 'rgba(239, 68, 68, 0.85)' }}
                  />
                  <span>
                    <span className="sm:hidden">↓</span>
                    <span className="hidden sm:inline">decreased</span>
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      ) : (
        (previewHeatmap || neuronHeatmap) && (
          <div className="mt-1 flex flex-nowrap items-center justify-center gap-2 text-center text-xs text-muted sm:gap-3 sm:text-sm">
            <div className={styles.legendItem}>
              <span
                className={styles.legendDot}
                style={{ backgroundColor: 'rgba(168, 85, 247, 0.85)' }}
              />
              <span>positive</span>
            </div>
            <div className={styles.legendItem}>
              <span
                className={styles.legendDot}
                style={{ backgroundColor: 'rgba(20, 184, 166, 0.85)' }}
              />
              <span>negative</span>
            </div>
          </div>
        )
      )}
    </div>
  );
};
