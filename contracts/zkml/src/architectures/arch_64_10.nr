// Single-layer linear network: 64 inputs -> 10 outputs (no hidden layer, no activation).

use crate::nn_trainer_template::{self, NNTrainerTemplate};
use crate::packing;
use crate::{SCALING_FACTOR, signed_fixed_multiply};
use crate::softmax;

pub global INPUT_SIZE: u32 = 64;
pub global OUTPUT_SIZE: u32 = 10;
pub global TRAINING_ARCHITECTURE_ID: u32 = 0;
pub global TRAINING_LEARNING_RATE: Field = 5000;
pub global TRAINING_REGULARIZATION: Field = 1000;
pub global TOTAL_WEIGHTS: u32 = 640; // 64 * 10
pub global TOTAL_BIASES: u32 = 10;

pub global WEIGHTS_PER_FIELD: u32 = packing::MAX_VALUES_PER_FIELD; // 28
pub global BIASES_PER_FIELD: u32 = 10; // all 10 biases fit in one Field
pub global PACKED_WEIGHT_FIELDS: u32 = 23; // ceil(640 / 28)
pub global PACKED_BIAS_FIELDS: u32 = 1;

pub struct TrainingExample {
    pub image: [Field; INPUT_SIZE],
    pub label: Field,
}

pub struct NetworkWeights {
    pub weights: [Field; TOTAL_WEIGHTS],
    pub biases: [Field; TOTAL_BIASES],
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
) -> ([Field; TOTAL_WEIGHTS], [Field; TOTAL_BIASES]) {
    let label_index: u32 = label as u32;
    assert(label_index < OUTPUT_SIZE, "Label out of range");
    assert(label == (label_index as Field), "Label must be an integer");

    let mut weight_gradients: [Field; TOTAL_WEIGHTS] = [0; TOTAL_WEIGHTS];
    let mut bias_gradients: [Field; TOTAL_BIASES] = [0; TOTAL_BIASES];

    // Gradient sign is inverted (1-p / -p) so callers ADD the LR term.
    let mut output_gradients: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];
    for i in 0..OUTPUT_SIZE {
        if i == label_index {
            output_gradients[i] = SCALING_FACTOR - probabilities[i];
        } else {
            output_gradients[i] = 0 - probabilities[i];
        }
    }

    // Single layer: weight_grad[i,j] = input[i] * output_grad[j]
    for i in 0..INPUT_SIZE {
        for j in 0..OUTPUT_SIZE {
            let weight_idx = i * OUTPUT_SIZE + j;
            weight_gradients[weight_idx] = signed_fixed_multiply(inputs[i], output_gradients[j]);
        }
    }
    for j in 0..OUTPUT_SIZE {
        bias_gradients[j] = output_gradients[j];
    }

    (weight_gradients, bias_gradients)
}

/// Apply one gradient descent step with L2 regularization and learning rate scaling.
fn update_weights(
    old_weights: NetworkWeights,
    weight_gradients: [Field; TOTAL_WEIGHTS],
    bias_gradients: [Field; TOTAL_BIASES],
    learning_rate: Field,
    regularization: Field,
) -> NetworkWeights {
    let mut new_weights: [Field; TOTAL_WEIGHTS] = [0; TOTAL_WEIGHTS];
    let mut new_biases: [Field; TOTAL_BIASES] = [0; TOTAL_BIASES];

    for i in 0..TOTAL_WEIGHTS {
        let reg_term = signed_fixed_multiply(regularization, old_weights.weights[i]);
        let total_gradient = weight_gradients[i] - reg_term;
        new_weights[i] =
            old_weights.weights[i] + signed_fixed_multiply(learning_rate, total_gradient);
    }

    for i in 0..TOTAL_BIASES {
        new_biases[i] =
            old_weights.biases[i] + signed_fixed_multiply(learning_rate, bias_gradients[i]);
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
    let (weight_gradients, bias_gradients) =
        compute_gradients(example.image, old_weights, probabilities, example.label);
    update_weights(
        old_weights,
        weight_gradients,
        bias_gradients,
        learning_rate,
        regularization,
    )
}

/// Forward pass: linear transform inputs (64) -> outputs (10), no activation.
pub fn compute_forward_pass(
    inputs: [Field; INPUT_SIZE],
    weights: NetworkWeights,
) -> [Field; OUTPUT_SIZE] {
    let mut outputs: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];

    for j in 0..OUTPUT_SIZE {
        let mut sum = weights.biases[j];
        for i in 0..INPUT_SIZE {
            sum = sum + signed_fixed_multiply(inputs[i], weights.weights[i * OUTPUT_SIZE + j]);
        }
        outputs[j] = sum;
    }

    outputs
}

pub fn compute_forward_pass_trainer(
    input_pixels: [Field; NNTrainerTemplate::MAX_TRAINER_INPUT_SIZE],
    network_weights: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS],
    network_biases: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES],
) -> [Field; NNTrainerTemplate::MAX_TRAINER_OUTPUT_SIZE] {
    let arch_weights: [Field; TOTAL_WEIGHTS] = NNTrainerTemplate::nn_trainer_copy_prefix::<TOTAL_WEIGHTS, NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS>(
        network_weights,
    );
    let arch_biases: [Field; TOTAL_BIASES] = NNTrainerTemplate::nn_trainer_copy_prefix::<TOTAL_BIASES, NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES>(
        network_biases,
    );
    let arch_network = NetworkWeights { weights: arch_weights, biases: arch_biases };
    let arch_outputs = compute_forward_pass(input_pixels, arch_network);
    NNTrainerTemplate::nn_trainer_copy_prefix::<NNTrainerTemplate::MAX_TRAINER_OUTPUT_SIZE, OUTPUT_SIZE>(
        arch_outputs,
    )
}

pub fn compute_new_weights_trainer(
    input_pixels: [Field; NNTrainerTemplate::MAX_TRAINER_INPUT_SIZE],
    label: Field,
    old_network_weights: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS],
    old_network_biases: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES],
    learning_rate: Field,
    regularization: Field,
    ) -> ([Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS], [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES]) {
    let arch_weights: [Field; TOTAL_WEIGHTS] = NNTrainerTemplate::nn_trainer_copy_prefix::<TOTAL_WEIGHTS, NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS>(
        old_network_weights,
    );
    let arch_biases: [Field; TOTAL_BIASES] = NNTrainerTemplate::nn_trainer_copy_prefix::<TOTAL_BIASES, NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES>(
        old_network_biases,
    );
    let old_arch_network = NetworkWeights { weights: arch_weights, biases: arch_biases };
    let old_example = TrainingExample { image: input_pixels, label };

    let new_network =
        compute_new_weights(old_example, old_arch_network, learning_rate, regularization);
    (
        NNTrainerTemplate::nn_trainer_copy_prefix::<NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS, TOTAL_WEIGHTS>(
            new_network.weights,
        ),
        NNTrainerTemplate::nn_trainer_copy_prefix::<NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES, TOTAL_BIASES>(
            new_network.biases,
        ),
    )
}
