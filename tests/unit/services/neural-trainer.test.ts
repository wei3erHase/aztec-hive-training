import { describe, it, expect } from 'vitest';
import {
  NeuralTrainer,
  DEFAULT_TRAINING_CONFIG,
  type NetworkWeights,
  type TrainingExample,
} from '@/services/core/neural_trainer';

function makeUniformImage(value = 0.5): number[] {
  return Array(784).fill(value);
}

function makeZeroWeights(): NetworkWeights {
  return {
    weights: Array.from({ length: 784 }, () => Array(10).fill(0)),
    biases: Array(10).fill(0),
  };
}

describe('NeuralTrainer', () => {
  describe('predict', () => {
    it('returns an integer in [0, 9]', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const result = trainer.predict(makeUniformImage());
      expect(Number.isInteger(result)).toBe(true);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThanOrEqual(9);
    });
  });

  describe('training invariant', () => {
    it('one step from uniform weights drives prediction toward the trained label', () => {
      // With all-zero weights, softmax gives uniform [0.1 … 0.1].
      // After one gradient step with LR=0.5 on label=3 and a non-zero image,
      // logit[3] rises sharply and the model should immediately predict 3.
      const trainer = new NeuralTrainer({
        learningRate: 0.5,
        batchSize: 1,
        epochs: 1,
        regularization: 0,
      });
      trainer.setWeights(makeZeroWeights());

      const image = makeUniformImage(0.5);
      const label = 3;
      trainer.trainBatch([{ image, label }]);

      expect(trainer.predict(image)).toBe(label);
    });

    it('loss decreases over several training steps on the same example', () => {
      const trainer = new NeuralTrainer({
        learningRate: 0.1,
        batchSize: 1,
        epochs: 1,
        regularization: 0,
      });
      trainer.setWeights(makeZeroWeights());

      const example: TrainingExample = {
        image: makeUniformImage(0.3),
        label: 7,
      };
      const firstLoss = trainer.trainBatch([example]);

      let lastLoss = firstLoss;
      for (let step = 0; step < 5; step++) {
        lastLoss = trainer.trainBatch([example]);
      }

      expect(lastLoss).toBeLessThan(firstLoss);
    });

    it('trainBatch returns a positive finite loss', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const loss = trainer.trainBatch([
        { image: makeUniformImage(), label: 0 },
      ]);
      expect(loss).toBeGreaterThan(0);
      expect(Number.isFinite(loss)).toBe(true);
    });
  });

  describe('getWeightsAsFields', () => {
    it('returns exactly 7840 weight field strings (784 inputs × 10 classes)', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const { weights } = trainer.getWeightsAsFields();
      expect(weights).toHaveLength(784 * 10);
    });

    it('returns exactly 10 bias field strings', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const { biases } = trainer.getWeightsAsFields();
      expect(biases).toHaveLength(10);
    });

    it('all field strings are parseable integers', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const { weights, biases } = trainer.getWeightsAsFields();
      for (const field of [...weights, ...biases]) {
        expect(Number.isInteger(Number(field))).toBe(true);
      }
    });

    it('zero weights produce all-zero field strings', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      trainer.setWeights(makeZeroWeights());
      const { weights, biases } = trainer.getWeightsAsFields();
      expect(weights.every((w) => w === '0')).toBe(true);
      expect(biases.every((b) => b === '0')).toBe(true);
    });
  });

  describe('getWeights / setWeights round-trip', () => {
    it('setWeights then getWeights returns identical values', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const known = makeZeroWeights();
      trainer.setWeights(known);
      const result = trainer.getWeights();
      expect(result.biases).toEqual(known.biases);
      expect(result.weights[0]).toEqual(known.weights[0]);
      expect(result.weights[783]).toEqual(known.weights[783]);
    });

    it('setWeights stores a copy — mutating the original does not affect the trainer', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const known = makeZeroWeights();
      trainer.setWeights(known);
      known.weights[0][0] = 999;
      expect(trainer.getWeights().weights[0][0]).toBe(0);
    });
  });

  describe('evaluate', () => {
    it('returns a value in [0, 1]', () => {
      const trainer = new NeuralTrainer(DEFAULT_TRAINING_CONFIG);
      const data: TrainingExample[] = [
        { image: makeUniformImage(0.5), label: 0 },
        { image: makeUniformImage(0.3), label: 1 },
        { image: makeUniformImage(0.1), label: 2 },
      ];
      const accuracy = trainer.evaluate(data);
      expect(accuracy).toBeGreaterThanOrEqual(0);
      expect(accuracy).toBeLessThanOrEqual(1);
    });

    it('returns 1.0 when the model predicts every example correctly', () => {
      const trainer = new NeuralTrainer({
        learningRate: 0.5,
        batchSize: 1,
        epochs: 1,
        regularization: 0,
      });
      trainer.setWeights(makeZeroWeights());
      const image = makeUniformImage(0.5);
      const label = 3;
      // Drive the model to predict 3 for this image
      for (let i = 0; i < 5; i++) {
        trainer.trainBatch([{ image, label }]);
      }
      expect(trainer.evaluate([{ image, label }])).toBe(1);
    });
  });

  describe('DEFAULT_TRAINING_CONFIG', () => {
    it('has a positive learningRate', () => {
      expect(DEFAULT_TRAINING_CONFIG.learningRate).toBeGreaterThan(0);
    });

    it('has a positive batchSize', () => {
      expect(DEFAULT_TRAINING_CONFIG.batchSize).toBeGreaterThan(0);
    });

    it('has a positive epoch count', () => {
      expect(DEFAULT_TRAINING_CONFIG.epochs).toBeGreaterThan(0);
    });

    it('has a non-negative regularization coefficient', () => {
      expect(DEFAULT_TRAINING_CONFIG.regularization).toBeGreaterThanOrEqual(0);
    });
  });
});
