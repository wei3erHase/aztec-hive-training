import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Button, Badge } from '../ui';
import type { TrainingState } from '../../hooks';
import type { ProofImageData } from '../../utils/canvas_processor';

type TrainStatus =
  | 'idle'
  | 'labeling'
  | 'training'
  | 'submitting'
  | 'success'
  | 'error';

interface TrainingControlsProps {
  currentImage: ProofImageData | null;
  selectedLabel: number | null;
  trainStatus: TrainStatus;
  statusMessage: string;
  isSubmittingOnChain: boolean;
  trainingState: TrainingState;
  onSubmit: () => void;
}

export const TrainingControls: React.FC<TrainingControlsProps> = ({
  currentImage,
  selectedLabel,
  trainStatus,
  statusMessage,
  isSubmittingOnChain,
  trainingState,
  onSubmit,
}) => {
  const isSubmitting = isSubmittingOnChain || trainStatus === 'submitting';
  const canSubmit = !!currentImage && selectedLabel !== null && !isSubmitting;
  const statusVariant =
    trainStatus === 'success'
      ? 'success'
      : trainStatus === 'error'
        ? 'error'
        : 'default';

  return (
    <>
      <Button
        variant="signal"
        size="lg"
        className="w-full"
        onClick={onSubmit}
        data-testid="train-submit-button"
        disabled={!canSubmit}
        isLoading={isSubmitting}
      >
        {isSubmitting ? 'Submitting Training…' : 'Train On-Chain'}
      </Button>

      {trainingState.isTraining && (
        <div className="panel rounded-2xl p-4" data-testid="training-progress">
          <div className="mb-2 flex justify-between text-base sm:text-sm">
            <span className="text-gray-400">Training Progress</span>
            <span className="text-accent font-mono">
              Epoch {trainingState.currentEpoch}/{trainingState.totalEpochs}
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-gray-700">
            <motion.div
              className="h-full bg-[#d4ff28]"
              initial={{ width: 0 }}
              animate={{ width: `${trainingState.progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="mt-2 text-sm text-gray-500 font-mono sm:text-xs">
            Loss: {trainingState.currentLoss.toFixed(4)} | Accuracy:{' '}
            {(trainingState.accuracy * 100).toFixed(1)}%
          </p>
        </div>
      )}

      <AnimatePresence>
        {statusMessage && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            data-testid="training-status-message"
            className="flex items-start gap-2"
          >
            <Badge variant={statusVariant} className="shrink-0 mt-0.5">
              {trainStatus === 'success'
                ? '✓'
                : trainStatus === 'error'
                  ? '✕'
                  : '…'}
            </Badge>
            <span className="text-sm text-gray-300">{statusMessage}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
