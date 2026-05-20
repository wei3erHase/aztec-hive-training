// CNN + Global Average Pooling architecture.
// Pipeline: Conv(4x4, 3 channels) -> ReLU -> GAP -> FC(3->10).
// Input: 8x8 grayscale image (64 pixels). 4x4 kernel chosen for accuracy vs.
// gas trade-off; yields CONV_OUT=75 with lower AVM cost than a 3x3 variant.

use crate::conv;
use crate::nn_trainer_template::{self, NNTrainerTemplate};
use crate::packing;
use crate::pooling;
use crate::{relu, SCALING_FACTOR, signed_fixed_multiply};
use crate::softmax;

pub global INPUT_SIZE: u32 = 64;
pub global OUTPUT_SIZE: u32 = 10;
pub global TRAINING_ARCHITECTURE_ID: u32 = 2;
pub global TRAINING_LEARNING_RATE: Field = 50000;
pub global TRAINING_REGULARIZATION: Field = 1000;

// Conv layer dimensions: 8x8x1 -> 5x5x3 (kernel 4x4, padding 0, stride 1).
pub global CONV_IN_H: u32 = 8;
pub global CONV_IN_W: u32 = 8;
pub global CONV_C_IN: u32 = 1;
pub global CONV_C_OUT: u32 = 3;
pub global CONV_K: u32 = 4;
pub global CONV_P: u32 = 0;
pub global CONV_S: u32 = 1;
pub global CONV_OUT_H: u32 = 5; // (8 - 4) / 1 + 1
pub global CONV_OUT_W: u32 = 5;
pub global CONV_OUT_SIZE: u32 = 75; // 5 * 5 * 3

pub global GAP_SIZE: u32 = 3; // one value per output channel after GAP

// Weights: conv(4*4*1*3=48) + fc(3*10=30) = 78
pub global TOTAL_WEIGHTS: u32 = 78;
pub global CONV_WEIGHTS: u32 = 48;
pub global FC_WEIGHTS: u32 = 30;

// Biases: conv(3) + fc(10) = 13
pub global TOTAL_BIASES: u32 = 13;
pub global CONV_BIASES: u32 = 3;
pub global FC_BIASES: u32 = 10;

pub global WEIGHTS_PER_FIELD: u32 = packing::MAX_VALUES_PER_FIELD; // 28
pub global BIASES_PER_FIELD: u32 = 28; // all 13 biases fit in one Field
pub global PACKED_WEIGHT_FIELDS: u32 = 3; // ceil(78 / 28)
pub global PACKED_BIAS_FIELDS: u32 = 1;

pub struct TrainingExample {
    pub image: [Field; INPUT_SIZE],
    pub label: Field,
}

pub struct NetworkWeights {
    pub weights: [Field; TOTAL_WEIGHTS],
    pub biases: [Field; TOTAL_BIASES],
}

/// Forward pass: Conv(8x8x1->5x5x3) -> ReLU -> GAP(->3) -> FC(3->10).
pub fn compute_forward_pass(
    inputs: [Field; INPUT_SIZE],
    weights: NetworkWeights,
) -> [Field; OUTPUT_SIZE] {
    let input_3d: [Field; CONV_IN_H * CONV_IN_W * CONV_C_IN] = inputs;

    let mut conv_weights: [Field; CONV_WEIGHTS] = [0; CONV_WEIGHTS];
    for i in 0..CONV_WEIGHTS {
        conv_weights[i] = weights.weights[i];
    }
    let mut conv_biases: [Field; CONV_BIASES] = [0; CONV_BIASES];
    for i in 0..CONV_BIASES {
        conv_biases[i] = weights.biases[i];
    }

    let conv_out: [Field; CONV_OUT_SIZE] = conv::conv2d::<CONV_IN_H, CONV_IN_W, CONV_C_IN, CONV_C_OUT, CONV_K, CONV_P, CONV_S, CONV_OUT_H, CONV_OUT_W>(
        input_3d,
        conv_weights,
        conv_biases,
    );

    let mut conv_relu: [Field; CONV_OUT_SIZE] = [0; CONV_OUT_SIZE];
    for i in 0..CONV_OUT_SIZE {
        conv_relu[i] = relu(conv_out[i]);
    }

    let gap_out: [Field; GAP_SIZE] =
        pooling::global_avg_pool::<CONV_OUT_H, CONV_OUT_W, CONV_C_OUT>(conv_relu);

    let mut outputs: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];
    for j in 0..OUTPUT_SIZE {
        let mut sum = weights.biases[CONV_BIASES + j];
        for i in 0..GAP_SIZE {
            sum = sum
                + signed_fixed_multiply(
                    gap_out[i],
                    weights.weights[CONV_WEIGHTS + i * OUTPUT_SIZE + j],
                );
        }
        outputs[j] = sum;
    }

    outputs
}

fn apply_softmax(logits: [Field; OUTPUT_SIZE]) -> [Field; OUTPUT_SIZE] {
    softmax::softmax(logits)
}

/// Compute updated weights for a single training example (full forward + backward pass).
pub fn compute_new_weights(
    example: TrainingExample,
    old_weights: NetworkWeights,
    learning_rate: Field,
    regularization: Field,
) -> NetworkWeights {
    let input_3d: [Field; CONV_IN_H * CONV_IN_W * CONV_C_IN] = example.image;

    let mut conv_weights: [Field; CONV_WEIGHTS] = [0; CONV_WEIGHTS];
    for i in 0..CONV_WEIGHTS {
        conv_weights[i] = old_weights.weights[i];
    }
    let mut conv_biases: [Field; CONV_BIASES] = [0; CONV_BIASES];
    for i in 0..CONV_BIASES {
        conv_biases[i] = old_weights.biases[i];
    }

    let conv_out: [Field; CONV_OUT_SIZE] = conv::conv2d::<CONV_IN_H, CONV_IN_W, CONV_C_IN, CONV_C_OUT, CONV_K, CONV_P, CONV_S, CONV_OUT_H, CONV_OUT_W>(
        input_3d,
        conv_weights,
        conv_biases,
    );

    let mut conv_relu: [Field; CONV_OUT_SIZE] = [0; CONV_OUT_SIZE];
    for i in 0..CONV_OUT_SIZE {
        conv_relu[i] = relu(conv_out[i]);
    }

    let gap_out: [Field; GAP_SIZE] =
        pooling::global_avg_pool::<CONV_OUT_H, CONV_OUT_W, CONV_C_OUT>(conv_relu);

    let mut outputs: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];
    for j in 0..OUTPUT_SIZE {
        let mut sum = old_weights.biases[CONV_BIASES + j];
        for i in 0..GAP_SIZE {
            let weight_idx = CONV_WEIGHTS + i * OUTPUT_SIZE + j;
            let weight = old_weights.weights[weight_idx];
            let product = signed_fixed_multiply(gap_out[i], weight);
            sum = sum + product;
        }
        outputs[j] = sum;
    }

    let probabilities = apply_softmax(outputs);

    let label_index: u32 = example.label as u32;
    assert(label_index < OUTPUT_SIZE, "Label out of range");
    assert(example.label == (label_index as Field), "Label must be an integer");

    let mut output_gradients: [Field; OUTPUT_SIZE] = [0; OUTPUT_SIZE];
    for i in 0..OUTPUT_SIZE {
        if i == label_index {
            output_gradients[i] = SCALING_FACTOR - probabilities[i];
        } else {
            output_gradients[i] = 0 - probabilities[i];
        }
    }

    // FC backward: gap_grad[c] = sum_j(output_grad[j] * fc_weight[c,j])
    let mut gap_grad: [Field; GAP_SIZE] = [0; GAP_SIZE];
    for c in 0..GAP_SIZE {
        let mut sum: Field = 0;
        for j in 0..OUTPUT_SIZE {
            let weight_idx = CONV_WEIGHTS + c * OUTPUT_SIZE + j;
            let w = old_weights.weights[weight_idx];
            sum = sum + signed_fixed_multiply(output_gradients[j], w);
        }
        gap_grad[c] = sum;
    }

    let conv_out_grad: [Field; CONV_OUT_SIZE] =
        pooling::gap_backward_relu_masked::<CONV_OUT_H, CONV_OUT_W, CONV_C_OUT>(conv_out, gap_grad);

    let (_input_grad, conv_weight_grad, conv_bias_grad) = conv::conv2d_backward::<CONV_IN_H, CONV_IN_W, CONV_C_IN, CONV_C_OUT, CONV_K, CONV_P, CONV_S, CONV_OUT_H, CONV_OUT_W>(
        input_3d,
        conv_out_grad,
        conv_weights,
    );

    // FC weight gradients: fc_weight_grad[c,j] = gap_out[c] * output_grad[j].
    let mut new_weights: [Field; TOTAL_WEIGHTS] = [0; TOTAL_WEIGHTS];
    for i in 0..CONV_WEIGHTS {
        let grad = conv_weight_grad[i];
        let reg_term = signed_fixed_multiply(regularization, old_weights.weights[i]);
        let total_grad = grad - reg_term;
        let lr_grad = signed_fixed_multiply(learning_rate, total_grad);
        new_weights[i] = old_weights.weights[i] + lr_grad;
    }
    for c in 0..GAP_SIZE {
        for j in 0..OUTPUT_SIZE {
            let weight_idx = CONV_WEIGHTS + c * OUTPUT_SIZE + j;
            let grad = signed_fixed_multiply(gap_out[c], output_gradients[j]);
            let reg_term = signed_fixed_multiply(regularization, old_weights.weights[weight_idx]);
            let total_grad = grad - reg_term;
            let lr_grad = signed_fixed_multiply(learning_rate, total_grad);
            new_weights[weight_idx] = old_weights.weights[weight_idx] + lr_grad;
        }
    }

    let mut new_biases: [Field; TOTAL_BIASES] = [0; TOTAL_BIASES];
    for i in 0..CONV_BIASES {
        let lr_grad = signed_fixed_multiply(learning_rate, conv_bias_grad[i]);
        new_biases[i] = old_weights.biases[i] + lr_grad;
    }
    for j in 0..OUTPUT_SIZE {
        let lr_grad = signed_fixed_multiply(learning_rate, output_gradients[j]);
        new_biases[CONV_BIASES + j] = old_weights.biases[CONV_BIASES + j] + lr_grad;
    }

    NetworkWeights { weights: new_weights, biases: new_biases }
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
    let old_weights: [Field; TOTAL_WEIGHTS] = NNTrainerTemplate::nn_trainer_copy_prefix::<TOTAL_WEIGHTS, NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS>(
        old_network_weights,
    );
    let old_biases: [Field; TOTAL_BIASES] = NNTrainerTemplate::nn_trainer_copy_prefix::<TOTAL_BIASES, NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES>(
        old_network_biases,
    );

    let old_arch_network = NetworkWeights { weights: old_weights, biases: old_biases };

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
