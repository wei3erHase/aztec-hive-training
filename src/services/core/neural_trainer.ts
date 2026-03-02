/**
 * Neural Network Training Logic
 * Implements gradient descent and weight optimization for the neural network
 */

export interface TrainingExample {
  image: number[];
  label: number;
}

export interface NetworkWeights {
  weights: number[][];
  biases: number[];
}

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  regularization: number;
}

export class NeuralTrainer {
  private config: TrainingConfig;
  private weights: number[][] = [];
  private biases: number[] = [];

  constructor(config: TrainingConfig) {
    this.config = config;
    this.initializeWeights();
  }

  private initializeWeights(): void {
    const inputSize = 784;
    const outputSize = 10;
    const variance = 2.0 / (inputSize + outputSize);
    const stdDev = Math.sqrt(variance);

    this.weights = Array(inputSize)
      .fill(0)
      .map(() =>
        Array(outputSize)
          .fill(0)
          .map(() => this.randomNormal(0, stdDev))
      );
    this.biases = Array(outputSize).fill(0);
  }

  private randomNormal(mean: number, stdDev: number): number {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * stdDev + mean;
  }

  private forwardPass(image: number[]): {
    outputs: number[];
    activations: number[];
  } {
    const outputs = Array(10).fill(0);
    for (let j = 0; j < 10; j++) {
      let sum = this.biases[j];
      for (let i = 0; i < 784; i++) {
        sum += image[i] * this.weights[i][j];
      }
      outputs[j] = sum;
    }
    const activations = this.softmax(outputs);
    return { outputs, activations };
  }

  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((x) => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((sum, x) => sum + x, 0);
    return expLogits.map((x) => x / sumExp);
  }

  private computeLoss(predictions: number[], label: number): number {
    const epsilon = 1e-15;
    const clippedPred = Math.max(
      epsilon,
      Math.min(1 - epsilon, predictions[label])
    );
    return -Math.log(clippedPred);
  }

  private backwardPass(
    image: number[],
    predictions: number[],
    label: number
  ): { weightGradients: number[][]; biasGradients: number[] } {
    const weightGradients = Array(784)
      .fill(0)
      .map(() => Array(10).fill(0));
    const biasGradients = Array(10).fill(0);
    const outputGradients = [...predictions];
    outputGradients[label] -= 1;

    for (let i = 0; i < 784; i++) {
      for (let j = 0; j < 10; j++) {
        weightGradients[i][j] = image[i] * outputGradients[j];
      }
    }
    for (let j = 0; j < 10; j++) {
      biasGradients[j] = outputGradients[j];
    }
    return { weightGradients, biasGradients };
  }

  private updateWeights(
    weightGradients: number[][],
    biasGradients: number[],
    batchSize: number
  ): void {
    const lr = this.config.learningRate;
    const reg = this.config.regularization;

    for (let i = 0; i < 784; i++) {
      for (let j = 0; j < 10; j++) {
        const gradient = weightGradients[i][j] / batchSize;
        const regularizationTerm = reg * this.weights[i][j];
        this.weights[i][j] -= lr * (gradient + regularizationTerm);
      }
    }
    for (let j = 0; j < 10; j++) {
      this.biases[j] -= lr * (biasGradients[j] / batchSize);
    }
  }

  public trainBatch(examples: TrainingExample[]): number {
    const batchSize = examples.length;
    let totalLoss = 0;
    const accWeightGradients = Array(784)
      .fill(0)
      .map(() => Array(10).fill(0));
    const accBiasGradients = Array(10).fill(0);

    for (const example of examples) {
      const { activations } = this.forwardPass(example.image);
      totalLoss += this.computeLoss(activations, example.label);
      const { weightGradients, biasGradients } = this.backwardPass(
        example.image,
        activations,
        example.label
      );
      for (let i = 0; i < 784; i++) {
        for (let j = 0; j < 10; j++) {
          accWeightGradients[i][j] += weightGradients[i][j];
        }
      }
      for (let j = 0; j < 10; j++) {
        accBiasGradients[j] += biasGradients[j];
      }
    }
    this.updateWeights(accWeightGradients, accBiasGradients, batchSize);
    return totalLoss / batchSize;
  }

  public train(trainingData: TrainingExample[]): {
    losses: number[];
    accuracy: number;
  } {
    const losses: number[] = [];
    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
      let epochLoss = 0;
      let batchCount = 0;

      for (let i = 0; i < shuffled.length; i += this.config.batchSize) {
        const batch = shuffled.slice(i, i + this.config.batchSize);
        epochLoss += this.trainBatch(batch);
        batchCount++;
      }
      losses.push(epochLoss / batchCount);
    }
    return { losses, accuracy: this.evaluate(trainingData) };
  }

  public evaluate(testData: TrainingExample[]): number {
    let correct = 0;
    for (const example of testData) {
      if (this.predict(example.image) === example.label) correct++;
    }
    return correct / testData.length;
  }

  public predict(image: number[]): number {
    const { activations } = this.forwardPass(image);
    return activations.indexOf(Math.max(...activations));
  }

  public getWeights(): NetworkWeights {
    return {
      weights: this.weights.map((row) => [...row]),
      biases: [...this.biases],
    };
  }

  public setWeights(weights: NetworkWeights): void {
    this.weights = weights.weights.map((row) => [...row]);
    this.biases = [...weights.biases];
  }

  public getWeightsAsFields(): { weights: string[]; biases: string[] } {
    const flatWeights: string[] = [];
    for (let i = 0; i < 784; i++) {
      for (let j = 0; j < 10; j++) {
        flatWeights.push(Math.round(this.weights[i][j] * 1000000).toString());
      }
    }
    const scaledBiases = this.biases.map((b) =>
      Math.round(b * 1000000).toString()
    );
    return { weights: flatWeights, biases: scaledBiases };
  }
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learningRate: 0.05,
  batchSize: 32,
  epochs: 10,
  regularization: 0.001,
};
