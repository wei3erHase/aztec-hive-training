/**
 * Input Shapley Value Visualization
 * Displays Shapley values for 64 input pixels as an 8x8 heatmap
 */

import React from 'react';
import { motion } from 'motion/react';

interface InputShapleyVisualizationProps {
  shapleyValues: number[]; // 64 values
  className?: string;
}

export function InputShapleyVisualization({
  shapleyValues,
  className = '',
}: InputShapleyVisualizationProps) {
  // Normalize values for visualization
  const maxAbs = Math.max(
    ...shapleyValues.map((v) => Math.abs(v)),
    0.0001 // Avoid division by zero
  );

  // Convert to 8x8 grid
  const grid: number[][] = [];
  for (let y = 0; y < 8; y++) {
    const row: number[] = [];
    for (let x = 0; x < 8; x++) {
      row.push(shapleyValues[y * 8 + x]);
    }
    grid.push(row);
  }

  // Get color for a value (red for negative, blue for positive)
  const getColor = (value: number) => {
    const normalized = value / maxAbs; // Range: -1 to 1
    const intensity = Math.abs(normalized);

    if (normalized < 0) {
      // Negative: red scale
      const r = Math.min(255, 100 + intensity * 155);
      const g = Math.max(0, 100 - intensity * 100);
      const b = Math.max(0, 100 - intensity * 100);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Positive: blue scale
      const r = Math.max(0, 100 - intensity * 100);
      const g = Math.max(0, 100 - intensity * 100);
      const b = Math.min(255, 100 + intensity * 155);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  return (
    <div className={className}>
      <h4 className="text-sm font-semibold text-gray-300 mb-3">
        Input Feature Contributions (8×8)
      </h4>
      <div className="grid grid-cols-8 gap-1 p-2 bg-gray-900/50 rounded-lg">
        {grid.map((row, y) =>
          row.map((value, x) => {
            const normalized = value / maxAbs;
            const intensity = Math.abs(normalized);

            return (
              <motion.div
                key={`${y}-${x}`}
                className="aspect-square rounded border border-gray-700 flex items-center justify-center relative overflow-hidden"
                style={{
                  backgroundColor: getColor(value),
                  opacity: 0.3 + intensity * 0.7,
                }}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.3 + intensity * 0.7 }}
                transition={{ delay: (y * 8 + x) * 0.01 }}
                title={`Pixel (${x}, ${y}): ${value.toFixed(6)}`}
              >
                {/* Value indicator */}
                {intensity > 0.1 && (
                  <span
                    className={`text-[6px] font-mono ${
                      normalized < 0 ? 'text-red-200' : 'text-blue-200'
                    }`}
                  >
                    {normalized > 0 ? '+' : ''}
                    {normalized.toFixed(1)}
                  </span>
                )}
              </motion.div>
            );
          })
        )}
      </div>
      <div className="flex items-center justify-between mt-2 text-xs text-gray-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500/50 rounded" />
          <span>Negative contribution</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500/50 rounded" />
          <span>Positive contribution</span>
        </div>
      </div>
    </div>
  );
}
