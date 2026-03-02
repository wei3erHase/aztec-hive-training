import React from 'react';
import { motion } from 'motion/react';
import { cn } from '../../utils';
import type { PredictionResult } from '../../hooks';
import type { ProofImageData } from '../../utils/canvas_processor';

interface PredictionMatrixProps {
  prediction: PredictionResult | null;
  selectedLabel: number | null;
  isExplaining: boolean;
  currentImage: ProofImageData | null;
  onLabelSelect: (label: number) => void;
}

const styles = {
  panel: 'panel rounded-2xl p-4 sm:p-5',
  header: 'mb-3 flex items-center justify-between gap-2',
  title: 'text-base font-semibold text-default sm:text-lg',
  meta: 'text-right',
  metaLabel: 'mono-label text-muted',
  metaTime: 'mt-1 text-xs text-muted sm:text-[11px]',
  subtitle: 'mb-4 text-base text-muted sm:text-sm',
  grid: 'grid grid-cols-5 gap-2',
  cellBase:
    'relative flex aspect-square flex-col items-center justify-center overflow-hidden rounded-xl border font-mono transition-all',
  cellSelectedPredicted:
    'border-emerald-600/80 bg-emerald-100 ring-1 ring-[#7a9820]/45 dark:border-emerald-500/80 dark:bg-emerald-500/22 dark:ring-[var(--accent-primary)]/45',
  cellSelected:
    'border-emerald-600/80 bg-emerald-100 dark:border-emerald-500/80 dark:bg-emerald-500/16',
  cellPredicted:
    'border-[#6f8d10]/60 bg-[#dbe8a7] dark:border-[var(--accent-primary)]/55 dark:bg-[var(--accent-primary)]/12',
  cellDefault:
    'border-slate-300 bg-white dark:border-default dark:bg-interactive',
  cellDisabled: 'cursor-not-allowed opacity-70',
  badgeBest:
    'absolute left-1 top-1 hidden rounded bg-[#dbe8a7] px-1.5 py-0.5 text-[10px] text-[#3f5800] dark:bg-[#d4ff28]/20 dark:text-[#d4ff28] sm:inline sm:text-[9px]',
  badgeTarget:
    'absolute right-1 top-1 hidden rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300 sm:inline sm:text-[9px]',
  digitPredicted:
    'relative z-10 text-lg sm:text-xl text-[#3f5800] dark:text-[#d4ff28]',
  digitDefault: 'relative z-10 text-lg sm:text-xl text-default',
  probPredicted:
    'relative z-10 mt-1 text-xs sm:text-[11px] text-[#5e7300] dark:text-[#c8ef55]',
  probDefault:
    'relative z-10 mt-1 text-xs sm:text-[11px] text-slate-600 dark:text-muted',
  fillPredicted:
    'absolute bottom-0 left-0 right-0 bg-[#b9d463]/40 dark:bg-[#d4ff28]/30',
  fillDefault:
    'absolute bottom-0 left-0 right-0 bg-slate-300/45 dark:bg-[var(--accent-primary)]/12',
} as const;

export const PredictionMatrix: React.FC<PredictionMatrixProps> = ({
  prediction,
  selectedLabel,
  isExplaining,
  currentImage,
  onLabelSelect,
}) => {
  const disabled = !currentImage || isExplaining;

  return (
    <div className={styles.panel} data-testid="predictions-panel">
      <div className={styles.header}>
        <h3 className={styles.title}>Prediction Matrix</h3>
        <div className={styles.meta}>
          <span className={styles.metaLabel}>0-9</span>
          {prediction && (
            <p className={styles.metaTime}>
              Inference: {prediction.processingTime.toFixed(2)}ms
            </p>
          )}
        </div>
      </div>
      <p className={styles.subtitle}>
        Predicted digit stays highlighted. Click any digit to choose the
        training target.
      </p>
      <div className={styles.grid}>
        {Array.from({ length: 10 }).map((_, i) => {
          const isPredicted = prediction?.digit === i;
          const isSelected = selectedLabel === i;
          const prob = prediction?.probabilities
            ? (prediction.probabilities[i] ?? 0)
            : isPredicted
              ? (prediction?.confidence ?? 0)
              : 0;

          return (
            <motion.button
              key={i}
              data-testid={`prediction-cell-${i}`}
              onClick={() => onLabelSelect(i)}
              disabled={disabled}
              className={cn(
                styles.cellBase,
                isSelected && isPredicted
                  ? styles.cellSelectedPredicted
                  : isSelected
                    ? styles.cellSelected
                    : isPredicted
                      ? styles.cellPredicted
                      : styles.cellDefault,
                disabled && styles.cellDisabled
              )}
              whileHover={!disabled ? { scale: 1.03 } : {}}
              whileTap={!disabled ? { scale: 0.97 } : {}}
            >
              {isPredicted && <span className={styles.badgeBest}>best</span>}
              {isSelected && <span className={styles.badgeTarget}>target</span>}
              <span
                className={
                  isPredicted ? styles.digitPredicted : styles.digitDefault
                }
              >
                {i}
              </span>
              {prediction && (
                <span
                  className={
                    isPredicted ? styles.probPredicted : styles.probDefault
                  }
                >
                  {(prob * 100).toFixed(1)}%
                </span>
              )}
              <div
                className={
                  isPredicted ? styles.fillPredicted : styles.fillDefault
                }
                style={{ height: `${prob * 100}%` }}
              />
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};
