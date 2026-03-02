// 2-layer MLP: 64 inputs -> 16 hidden (ReLU) -> 10 outputs.
// Forward pass, backpropagation, and weight update for this architecture.

use crate::packing;
use crate::{relu, SCALING_FACTOR, signed_fixed_multiply};
use crate::softmax;

pub global INPUT_SIZE: u32 = 64;
pub global HIDDEN_SIZE: u32 = 16;
pub global OUTPUT_SIZE: u32 = 10;
pub global TOTAL_WEIGHTS: u32 = 1184; // 64*16 + 16*10
pub global TOTAL_BIASES: u32 = 26; // 16 + 10

pub global WEIGHTS_PER_FIELD: u32 = packing::MAX_VALUES_PER_FIELD; // 28
pub global BIASES_PER_FIELD: u32 = 26; // all 26 biases fit in one Field
pub global PACKED_WEIGHT_FIELDS: u32 = 43; // ceil(1184 / 28)
pub global PACKED_BIAS_FIELDS: u32 = 1;

// Layer boundaries for weight indexing
pub global L1_WEIGHTS_START: u32 = 0;
pub global L1_WEIGHTS_END: u32 = 1024; // 64 * 16
pub global L2_WEIGHTS_START: u32 = 1024;
pub global L2_WEIGHTS_END: u32 = 1184; // 1024 + 16 * 10
pub global L1_BIASES_START: u32 = 0;
pub global L1_BIASES_END: u32 = 16;
pub global L2_BIASES_START: u32 = 16;
pub global L2_BIASES_END: u32 = 26;

pub struct TrainingExample {
    pub image: [Field; INPUT_SIZE],
    pub label: Field,
}

pub struct NetworkWeights {
    pub weights: [Field; TOTAL_WEIGHTS],
    pub biases: [Field; TOTAL_BIASES],
}

struct Gradients {
    weight_gradients: [Field; TOTAL_WEIGHTS],
    bias_gradients: [Field; TOTAL_BIASES],
}

/// Forward pass: inputs (64) -> hidden (16, ReLU) -> outputs (10).
pub fn compute_forward_pass(
    inputs: [Field; INPUT_SIZE],
    weights: NetworkWeights,
) -> [Field; OUTPUT_SIZE] {
    let mut hidden: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
    for j in 0..HIDDEN_SIZE {
        let mut sum = weights.biases[j];
        for i in 0..INPUT_SIZE {
            sum = sum + signed_fixed_multiply(inputs[i], weights.weights[i * HIDDEN_SIZE + j]);
        }
        hidden[j] = relu(sum);
    }

    let mut outputs: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];
    for j in 0..OUTPUT_SIZE {
        let mut sum = weights.biases[HIDDEN_SIZE + j]; // L2 bias offset by hidden layer size
        for i in 0..HIDDEN_SIZE {
            sum = sum
                + signed_fixed_multiply(
                    hidden[i],
                    weights.weights[L2_WEIGHTS_START + i * OUTPUT_SIZE + j],
                );
        }
        outputs[j] = sum;
    }

    outputs
}

fn apply_softmax(logits: [Field; OUTPUT_SIZE]) -> [Field; OUTPUT_SIZE] {
    softmax::softmax(logits)
}

/// Backpropagation for one training example. Returns weight and bias gradients.
fn compute_gradients(
    inputs: [Field; INPUT_SIZE],
    weights: NetworkWeights,
    probabilities: [Field; OUTPUT_SIZE],
    label: Field,
) -> Gradients {
    let label_index: u32 = label as u32;
    assert(label_index < OUTPUT_SIZE, "Label out of range");
    assert(label == (label_index as Field), "Label must be an integer");

    let mut weight_gradients: [Field; TOTAL_WEIGHTS] = [0; TOTAL_WEIGHTS];
    let mut bias_gradients: [Field; TOTAL_BIASES] = [0; TOTAL_BIASES];

    // ---- Output layer: softmax cross-entropy gradient ----
    // Gradient sign is inverted (1-p for correct, -p for others) so callers ADD instead of subtract.
    let mut output_gradients: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];
    for i in 0..OUTPUT_SIZE {
        if i == label_index {
            output_gradients[i] = SCALING_FACTOR - probabilities[i];
        } else {
            output_gradients[i] = 0 - probabilities[i];
        }
    }

    // ---- Recompute hidden activations (needed for L1 backprop) ----
    let mut hidden_pre_relu: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
    let mut hidden_activated: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
    for j in 0..HIDDEN_SIZE {
        let mut sum = weights.biases[j];
        for i in 0..INPUT_SIZE {
            sum = sum + signed_fixed_multiply(inputs[i], weights.weights[i * HIDDEN_SIZE + j]);
        }
        hidden_pre_relu[j] = sum;
        hidden_activated[j] = relu(sum);
    }

    // ---- L2 gradients: dL/dW2[i,j] = hidden_activated[i] * output_grad[j] ----
    for i in 0..HIDDEN_SIZE {
        for j in 0..OUTPUT_SIZE {
            weight_gradients[L2_WEIGHTS_START + i * OUTPUT_SIZE + j] =
                signed_fixed_multiply(hidden_activated[i], output_gradients[j]);
        }
    }
    for j in 0..OUTPUT_SIZE {
        bias_gradients[HIDDEN_SIZE + j] = output_gradients[j];
    }

    // ---- Backprop through ReLU: hidden_grad[i] = relu'(h_pre[i]) * sum_j(W2[i,j] * out_grad[j]) ----
    let mut hidden_gradients: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
    for i in 0..HIDDEN_SIZE {
        // MAX_POSITIVE_PRODUCT threshold: neg_u64 < threshold means value was negative
        let relu_active = (0 - hidden_pre_relu[i]) as u64 >= 10_000_000_000_000;
        if relu_active {
            let mut grad_sum: Field = 0;
            for j in 0..OUTPUT_SIZE {
                grad_sum = grad_sum
                    + signed_fixed_multiply(
                        weights.weights[L2_WEIGHTS_START + i * OUTPUT_SIZE + j],
                        output_gradients[j],
                    );
            }
            hidden_gradients[i] = grad_sum;
        }
    }

    // ---- L1 gradients: dL/dW1[i,j] = inputs[i] * hidden_grad[j] ----
    for i in 0..INPUT_SIZE {
        for j in 0..HIDDEN_SIZE {
            weight_gradients[i * HIDDEN_SIZE + j] =
                signed_fixed_multiply(inputs[i], hidden_gradients[j]);
        }
    }
    for j in 0..HIDDEN_SIZE {
        bias_gradients[j] = hidden_gradients[j];
    }

    Gradients { weight_gradients, bias_gradients }
}

/// Apply one gradient descent step with L2 regularization and learning rate scaling.
fn update_weights(
    old_weights: NetworkWeights,
    gradients: Gradients,
    batch_size: Field,
    learning_rate: Field,
    regularization: Field,
) -> NetworkWeights {
    let mut new_weights: [Field; TOTAL_WEIGHTS] = [0; TOTAL_WEIGHTS];
    let mut new_biases: [Field; TOTAL_BIASES] = [0; TOTAL_BIASES];

    for i in 0..TOTAL_WEIGHTS {
        let gradient = if batch_size == 1 {
            gradients.weight_gradients[i]
        } else {
            gradients.weight_gradients[i] / batch_size
        };
        // L2 term pulls weights toward zero; subtracted before scaling by LR.
        let reg_term = signed_fixed_multiply(regularization, old_weights.weights[i]);
        let total_gradient = gradient - reg_term;
        // Gradient sign is inverted (1-p not p-1), so we add the LR term.
        new_weights[i] =
            old_weights.weights[i] + signed_fixed_multiply(learning_rate, total_gradient);
    }

    for i in 0..TOTAL_BIASES {
        let gradient = if batch_size == 1 {
            gradients.bias_gradients[i]
        } else {
            gradients.bias_gradients[i] / batch_size
        };
        new_biases[i] = old_weights.biases[i] + signed_fixed_multiply(learning_rate, gradient);
    }

    NetworkWeights { weights: new_weights, biases: new_biases }
}

/// Compute updated weights for a single training example (batch_size = 1).
pub fn compute_new_weights(
    example: TrainingExample,
    old_weights: NetworkWeights,
    learning_rate: Field,
    regularization: Field,
) -> NetworkWeights {
    let outputs = compute_forward_pass(example.image, old_weights);
    let probabilities = apply_softmax(outputs);
    let gradients = compute_gradients(example.image, old_weights, probabilities, example.label);
    update_weights(old_weights, gradients, 1, learning_rate, regularization)
}

/// Circuit-level training step verifier: asserts that new_weights equals the
/// result of running one gradient descent step on old_weights with the given example.
pub fn verify_training_step(
    example: TrainingExample,
    old_weights: NetworkWeights,
    new_weights: NetworkWeights,
    learning_rate: Field,
    regularization: Field,
) {
    let computed = compute_new_weights(example, old_weights, learning_rate, regularization);

    for i in 0..TOTAL_WEIGHTS {
        assert(
            computed.weights[i] == new_weights.weights[i],
            "Training step verification failed on weights",
        );
    }
    for i in 0..TOTAL_BIASES {
        assert(
            computed.biases[i] == new_weights.biases[i],
            "Training step verification failed on biases",
        );
    }
}
