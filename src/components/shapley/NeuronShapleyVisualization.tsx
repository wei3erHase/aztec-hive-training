/**
 * Neuron Shapley Value Visualization
 * Displays Shapley values for hidden neurons (MLP) or conv channels (CNN) as a bar chart.
 * Single Layer: pass empty array - component is hidden.
 */

import React from 'react';
import { motion } from 'motion/react';

interface NeuronShapleyVisualizationProps {
  shapleyValues: number[]; // 16 (MLP) or 2 (CNN) or [] (Single Layer - hidden)
  label?: string;
  className?: string;
}

export function NeuronShapleyVisualization({
  shapleyValues,
  label = 'Hidden Neurons (16)',
  className = '',
}: NeuronShapleyVisualizationProps) {
  if (shapleyValues.length === 0) {
    return null;
  }

  const maxAbs = Math.max(...shapleyValues.map((v) => Math.abs(v)), 0.0001);

  return (
    <div className={className}>
      <h4 className="text-sm font-semibold text-gray-300 mb-3">{label}</h4>
      <div className="grid grid-cols-4 gap-2">
        {shapleyValues.map((value, index) => {
          const normalized = value / maxAbs; // Range: -1 to 1
          const intensity = Math.abs(normalized);
          const isPositive = normalized >= 0;

          return (
            <motion.div
              key={index}
              className="flex flex-col items-center gap-1"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              {/* Bar */}
              <div className="w-full h-24 bg-gray-800 rounded relative overflow-hidden">
                <motion.div
                  className={`absolute bottom-0 left-0 right-0 ${
                    isPositive ? 'bg-blue-500' : 'bg-red-500'
                  }`}
                  initial={{ height: 0 }}
                  animate={{
                    height: `${intensity * 100}%`,
                  }}
                  transition={{ delay: index * 0.05, duration: 0.5 }}
                  style={{
                    opacity: 0.5 + intensity * 0.5,
                  }}
                />
                {/* Zero line */}
                <div className="absolute top-1/2 left-0 right-0 h-px bg-gray-600" />
                {/* Value label */}
                {intensity > 0.05 && (
                  <div
                    className={`absolute ${
                      isPositive ? 'bottom-1' : 'top-1'
                    } left-0 right-0 text-center text-[8px] font-mono ${
                      isPositive ? 'text-blue-200' : 'text-red-200'
                    }`}
                  >
                    {isPositive ? '+' : ''}
                    {normalized.toFixed(2)}
                  </div>
                )}
              </div>
              {/* Neuron index */}
              <span className="text-xs text-gray-500 font-mono">N{index}</span>
              {/* Value */}
              <span className="text-[10px] text-gray-400 font-mono">
                {value.toFixed(4)}
              </span>
            </motion.div>
          );
        })}
      </div>
      <div className="flex items-center justify-between mt-3 text-xs text-gray-400">
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
