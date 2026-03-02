import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  SL_WEIGHTS,
  SL_BIASES,
  MLP_WEIGHTS,
  MLP_BIASES,
  CNN_WEIGHTS,
  CNN_BIASES,
} from '../../config/pretrainedWeights';

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  root: 'rounded-2xl border border-default bg-surface overflow-hidden',
  tabBar: 'flex border-b border-default bg-surface-secondary',
  tab: 'flex-1 py-3 px-2 text-center transition-all duration-200 border-b-2',
  tabActive: 'border-[var(--accent-primary)] bg-surface',
  tabInactive: 'border-transparent hover:border-default hover:bg-interactive',
  tabLabel: 'block text-sm font-medium',
  tabLabelActive: 'text-default',
  tabLabelInactive: 'text-muted',
  tabSub: 'block text-[10px] font-mono mt-0.5',
  tabSubActive: 'text-accent',
  tabSubInactive: 'text-muted',
  body: 'p-4 sm:p-6',
  legend: 'flex items-center gap-3 mb-5',
  legendLabel: 'text-[10px] text-muted font-mono',
  legendGradient: 'flex rounded overflow-hidden',
  legendCell: 'h-2',
  section: 'mb-6',
  sectionTitle: 'flex items-center gap-2 mb-2',
  sectionLabel: 'text-[11px] font-semibold text-muted uppercase tracking-wider',
  sectionShape:
    'text-[10px] font-mono text-muted bg-interactive px-2 py-0.5 rounded',
  sectionCaption: 'text-[10px] text-muted mt-1.5',
  row: 'flex flex-wrap gap-8 items-start',
  filters: 'flex gap-4',
  filterWrap: 'flex flex-col gap-1',
  filterLabel: 'text-[9px] text-muted font-mono',
  biasCol: 'shrink-0',
  biasTitle: 'text-[11px] font-semibold text-muted mb-2',
  biasRow: 'flex items-center gap-1.5 mb-1',
  biasKey: 'font-mono text-right text-muted',
  biasTrack: 'rounded overflow-hidden bg-interactive',
  biasFill: 'h-full rounded',
  biasVal: 'font-mono text-muted',
} as const;

// ── Color mapping ─────────────────────────────────────────────────────────────
// Maps signed weight values to a blue (negative) → dark (zero) → amber (positive) gradient.

function weightColor(value: number, absMax: number): string {
  const t = absMax > 0 ? Math.max(-1, Math.min(1, value / absMax)) : 0;
  if (t < 0) {
    const a = -t;
    // dark → blue-500: rgb(59,130,246)
    return `rgb(${Math.round(17 + 42 * a)},${Math.round(17 + 113 * a)},${Math.round(24 + 222 * a)})`;
  } else if (t > 0) {
    // dark → amber-400: rgb(251,191,36)
    return `rgb(${Math.round(17 + 234 * t)},${Math.round(17 + 174 * t)},${Math.round(24 + 12 * t)})`;
  }
  return 'rgb(17,17,24)';
}

function absMax(arr: number[]): number {
  return arr.reduce((m, v) => Math.max(m, Math.abs(v)), 1);
}

// ── Weight grid ───────────────────────────────────────────────────────────────

interface GridProps {
  data: number[];
  rows: number;
  cols: number;
  px?: number;
}

function WeightGrid({ data, rows, cols, px = 5 }: GridProps) {
  const max = absMax(data);
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${cols}, ${px}px)`,
        gap: '1px',
        lineHeight: 0,
        flexShrink: 0,
      }}
    >
      {data.slice(0, rows * cols).map((v, i) => (
        <div
          key={i}
          style={{
            width: px,
            height: px,
            backgroundColor: weightColor(v, max),
          }}
        />
      ))}
    </div>
  );
}

// ── Bias bar chart ────────────────────────────────────────────────────────────

interface BiasBarProps {
  biases: number[];
  labels: string[];
  barWidth?: number;
}

function BiasBar({ biases, labels, barWidth = 72 }: BiasBarProps) {
  const max = absMax(biases);
  return (
    <div className="space-y-1">
      {biases.map((v, i) => (
        <div key={i} className={styles.biasRow}>
          <span
            className={styles.biasKey}
            style={{ width: 18, fontSize: 9, flexShrink: 0 }}
          >
            {labels[i]}
          </span>
          <div
            className={styles.biasTrack}
            style={{ width: barWidth, height: 7, flexShrink: 0 }}
          >
            <div
              className={styles.biasFill}
              style={{
                width: `${(Math.abs(v) / max) * 100}%`,
                height: '100%',
                backgroundColor: weightColor(v, max),
              }}
            />
          </div>
          <span className={styles.biasVal} style={{ fontSize: 9, width: 32 }}>
            {v}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Color legend ──────────────────────────────────────────────────────────────

function ColorLegend() {
  const STEPS = 28;
  return (
    <div className={styles.legend}>
      <span className={styles.legendLabel}>−256</span>
      <div className={styles.legendGradient}>
        {Array.from({ length: STEPS }, (_, i) => {
          const t = (i / (STEPS - 1)) * 2 - 1;
          return (
            <div
              key={i}
              className={styles.legendCell}
              style={{
                width: 10,
                backgroundColor: weightColor(Math.round(t * 255), 255),
              }}
            />
          );
        })}
      </div>
      <span className={styles.legendLabel}>+255</span>
    </div>
  );
}

// ── Tab definitions ───────────────────────────────────────────────────────────

type ArchId = 'singleLayer' | 'mlp' | 'cnn';

const TABS: { id: ArchId; label: string; sub: string }[] = [
  { id: 'singleLayer', label: 'Simple Linear', sub: '640w + 10b' },
  { id: 'mlp', label: 'MLP', sub: '1184w + 26b' },
  { id: 'cnn', label: 'CNN + GAP', sub: '78w + 13b' },
];

const DIGIT_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
const HIDDEN_LABELS = Array.from({ length: 16 }, (_, i) => `h${i}`);
const CONV_LABELS = ['c0', 'c1', 'c2'];
const GAP_LABELS = ['g0', 'g1', 'g2'];

// CNN: extract filter f (out_c = f) from interleaved conv weight array.
// weight_idx = (ky * K + kx) * C_IN * C_OUT + in_c * C_OUT + out_c
// With C_IN=1, C_OUT=3: weight_idx = kernel_pos * 3 + f
function extractConvFilter(
  weights: number[],
  f: number,
  kernelSize = 16
): number[] {
  return Array.from({ length: kernelSize }, (_, i) => weights[i * 3 + f]);
}

// ── Main component ────────────────────────────────────────────────────────────

export function WeightsViewer() {
  const [active, setActive] = useState<ArchId>('singleLayer');

  return (
    <div className={styles.root}>
      {/* Tab bar */}
      <div className={styles.tabBar}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            className={`${styles.tab} ${active === tab.id ? styles.tabActive : styles.tabInactive}`}
          >
            <span
              className={`${styles.tabLabel} ${active === tab.id ? styles.tabLabelActive : styles.tabLabelInactive}`}
            >
              {tab.label}
            </span>
            <span
              className={`${styles.tabSub} ${active === tab.id ? styles.tabSubActive : styles.tabSubInactive}`}
            >
              {tab.sub}
            </span>
          </button>
        ))}
      </div>

      {/* Body */}
      <div className={styles.body}>
        <ColorLegend />

        <AnimatePresence mode="wait">
          {active === 'singleLayer' && (
            <motion.div
              key="sl"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
            >
              <div className={styles.row}>
                <div>
                  <div className={styles.section}>
                    <div className={styles.sectionTitle}>
                      <span className={styles.sectionLabel}>Weights</span>
                      <span className={styles.sectionShape}>10 × 64</span>
                    </div>
                    <WeightGrid data={SL_WEIGHTS} rows={10} cols={64} px={5} />
                    <p className={styles.sectionCaption}>
                      rows: digit classes 0–9 · cols: 64 input pixels (8×8 grid)
                    </p>
                  </div>
                </div>
                <div className={styles.biasCol}>
                  <div className={styles.biasTitle}>
                    Biases{' '}
                    <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                      ×10
                    </span>
                  </div>
                  <BiasBar biases={SL_BIASES} labels={DIGIT_LABELS} />
                </div>
              </div>
            </motion.div>
          )}

          {active === 'mlp' && (
            <motion.div
              key="mlp"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
            >
              <div className={styles.row}>
                <div className="space-y-6">
                  <div className={styles.section}>
                    <div className={styles.sectionTitle}>
                      <span className={styles.sectionLabel}>
                        Layer 1 Weights
                      </span>
                      <span className={styles.sectionShape}>16 × 64</span>
                    </div>
                    <WeightGrid
                      data={MLP_WEIGHTS.slice(0, 1024)}
                      rows={16}
                      cols={64}
                      px={5}
                    />
                    <p className={styles.sectionCaption}>
                      rows: 16 hidden neurons · cols: 64 input pixels
                    </p>
                  </div>
                  <div className={styles.section}>
                    <div className={styles.sectionTitle}>
                      <span className={styles.sectionLabel}>
                        Layer 2 Weights
                      </span>
                      <span className={styles.sectionShape}>10 × 16</span>
                    </div>
                    <WeightGrid
                      data={MLP_WEIGHTS.slice(1024)}
                      rows={10}
                      cols={16}
                      px={5}
                    />
                    <p className={styles.sectionCaption}>
                      rows: digit classes 0–9 · cols: 16 hidden neurons
                    </p>
                  </div>
                </div>
                <div className={styles.biasCol}>
                  <div className={styles.biasTitle}>
                    Hidden Biases{' '}
                    <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                      ×16
                    </span>
                  </div>
                  <BiasBar
                    biases={MLP_BIASES.slice(0, 16)}
                    labels={HIDDEN_LABELS}
                  />
                  <div className={`${styles.biasTitle} mt-4`}>
                    Output Biases{' '}
                    <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                      ×10
                    </span>
                  </div>
                  <BiasBar
                    biases={MLP_BIASES.slice(16)}
                    labels={DIGIT_LABELS}
                  />
                </div>
              </div>
            </motion.div>
          )}

          {active === 'cnn' && (
            <motion.div
              key="cnn"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
            >
              <div className={styles.row}>
                <div className="space-y-6">
                  <div className={styles.section}>
                    <div className={styles.sectionTitle}>
                      <span className={styles.sectionLabel}>Conv Filters</span>
                      <span className={styles.sectionShape}>3 × 4 × 4</span>
                    </div>
                    <div className={styles.filters}>
                      {[0, 1, 2].map((f) => (
                        <div key={f} className={styles.filterWrap}>
                          <span className={styles.filterLabel}>filter {f}</span>
                          <WeightGrid
                            data={extractConvFilter(CNN_WEIGHTS, f)}
                            rows={4}
                            cols={4}
                            px={10}
                          />
                        </div>
                      ))}
                    </div>
                    <p className={styles.sectionCaption}>
                      3 learnable 4×4 kernels scanning the 8×8 input
                    </p>
                  </div>
                  <div className={styles.section}>
                    <div className={styles.sectionTitle}>
                      <span className={styles.sectionLabel}>FC Weights</span>
                      <span className={styles.sectionShape}>3 × 10</span>
                    </div>
                    <WeightGrid
                      data={CNN_WEIGHTS.slice(48)}
                      rows={3}
                      cols={10}
                      px={10}
                    />
                    <p className={styles.sectionCaption}>
                      rows: 3 GAP features · cols: digit classes 0–9
                    </p>
                  </div>
                </div>
                <div className={styles.biasCol}>
                  <div className={styles.biasTitle}>
                    Conv Biases{' '}
                    <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                      ×3
                    </span>
                  </div>
                  <BiasBar
                    biases={CNN_BIASES.slice(0, 3)}
                    labels={CONV_LABELS}
                    barWidth={80}
                  />
                  <div className={`${styles.biasTitle} mt-4`}>
                    FC Biases{' '}
                    <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                      ×10
                    </span>
                  </div>
                  <BiasBar
                    biases={CNN_BIASES.slice(3)}
                    labels={DIGIT_LABELS}
                    barWidth={80}
                  />
                  <div className={`${styles.biasTitle} mt-4`}>GAP features</div>
                  <p className={`${styles.sectionCaption} mt-0`}>
                    {GAP_LABELS.join(' · ')} — one averaged
                    <br />
                    activation per conv filter
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
