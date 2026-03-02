import React from 'react';
import { Badge } from '../ui';
import type { PredictionResult } from '../../hooks';

interface PredictionStatusProps {
  prediction: PredictionResult | null;
  selectedLabel: number | null;
  explanationProcessingTime: number | null;
  predictError: string | null;
  explanationError: string | null;
}

export const PredictionStatus: React.FC<PredictionStatusProps> = ({
  prediction,
  selectedLabel,
  explanationProcessingTime,
  predictError,
  explanationError,
}) => (
  <div
    className="panel rounded-2xl p-4 sm:p-5"
    data-testid="prediction-status-card"
  >
    <p className="mono-label mb-3 text-gray-500">Status</p>

    <div className="flex flex-wrap items-center gap-2">
      {predictError ? (
        <Badge variant="error">{predictError}</Badge>
      ) : prediction ? (
        <>
          <Badge variant="success">Predicted: {prediction.digit}</Badge>
          <Badge variant="default" className="font-mono">
            {(prediction.confidence * 100).toFixed(1)}%
          </Badge>
        </>
      ) : (
        <Badge variant="default">Draw a digit to get a prediction</Badge>
      )}

      {selectedLabel !== null && (
        <Badge variant="signal">Target: {selectedLabel}</Badge>
      )}
    </div>

    {explanationProcessingTime !== null && (
      <p className="mt-2 text-xs text-gray-500 font-mono">
        Explanation: {explanationProcessingTime.toFixed(2)}ms
      </p>
    )}

    {explanationError && (
      <Badge variant="warning" className="mt-2">
        {explanationError}
      </Badge>
    )}
  </div>
);
