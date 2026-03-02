// Architecture: CNN+GAP (Conv 4x4x3, ReLU, GAP, FC 3->10)
// 78 weights, 13 biases

use crate::architectures::arch_cnn_gap::{
    self, CONV_BIASES, CONV_C_IN, CONV_C_OUT, CONV_IN_H, CONV_IN_W, CONV_K, CONV_OUT_H,
    CONV_OUT_SIZE, CONV_OUT_W, CONV_P, CONV_S, CONV_WEIGHTS, GAP_SIZE, OUTPUT_SIZE,
};
use crate::{conv, pooling, relu, softmax};
use crate::test::utils;

global TOTAL_WEIGHTS: u32 = 78;
global TOTAL_BIASES: u32 = 13;
global WEIGHTS_PER_FIELD: u32 = 28;
global PACKED_WEIGHT_FIELDS: u32 = 3;
global PACKED_BIAS_FIELDS: u32 = 1;

global LEARNING_RATE: Field = 50000; // 0.05: higher for CNN (GAP attenuates gradients)
global REGULARIZATION: Field = 1000;

// ============================================================================
// Forward Pass Tests
// ============================================================================

#[test]
unconstrained fn test_cnn_gap_forward_pass_basic() {
    let input = utils::create_small_test_input();
    let weights = create_test_weights_cnn();
    let biases = create_test_biases_cnn();

    let outputs = compute_forward_pass_pure_cnn(input, weights, biases);

    // argmax must be a valid class index
    let max_idx = softmax::argmax(outputs);
    assert((max_idx as Field).lt(10), "CNN forward pass must produce a valid argmax in [0, 9]");

    // With non-zero inputs, at least one output must be non-zero
    let mut all_zero = true;
    for i in 0..10 {
        if outputs[i] != 0 {
            all_zero = false;
        }
    }
    assert(
        !all_zero,
        "CNN forward pass with non-zero input must produce at least one non-zero output",
    );
}

#[test]
unconstrained fn test_cnn_gap_forward_pass_zero_input() {
    let input: [Field; 64] = [0; 64];
    let weights = create_test_weights_cnn();
    // FC biases: biases[3..12] = [3000, 4000, ..., 12000]
    let biases = create_test_biases_cnn();

    let outputs = compute_forward_pass_pure_cnn(input, weights, biases);

    // With zero input, conv output is driven purely by conv biases -> GAP -> FC.
    // The FC layer still receives non-zero GAP activations (from conv biases),
    // so outputs should not all be zero.
    let mut all_zero = true;
    for i in 0..10 {
        if outputs[i] != 0 {
            all_zero = false;
        }
    }
    assert(
        !all_zero,
        "CNN with zero pixel input but non-zero biases must still produce non-zero logits",
    );
    // argmax must always be valid
    let max_idx = softmax::argmax(outputs);
    assert((max_idx as Field).lt(10), "argmax must be a valid class index with zero input");
}

// ============================================================================
// Training Tests
// ============================================================================

#[test]
unconstrained fn test_cnn_gap_compute_new_weights_updates_target_bias() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[0] = 100000;
    input_pixels[10] = 200000;

    let label: Field = 3;
    let weights = create_trained_weights_cnn();
    let biases = create_trained_biases_cnn();
    let old_network = arch_cnn_gap::NetworkWeights { weights, biases };

    let new_network = arch_cnn_gap::compute_new_weights(
        arch_cnn_gap::TrainingExample { image: input_pixels, label },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );

    // FC biases are stored at biases[CONV_BIASES + label] (indices 3..12).
    // The FC bias update only requires output_gradients (not gap_out), so it is
    // always non-zero after one gradient step.
    assert(
        new_network.biases[CONV_BIASES + label as u32]
            != old_network.biases[CONV_BIASES + label as u32],
        "Training step must update the FC bias for the target class",
    );
    // Verify at least one of the non-target FC biases also changed
    // (output_gradients[j] = -P(j) for j != label, which is always non-zero).
    let mut any_other_bias_changed = false;
    for j in 0..OUTPUT_SIZE {
        if j != (label as u32) {
            if new_network.biases[CONV_BIASES + j] != old_network.biases[CONV_BIASES + j] {
                any_other_bias_changed = true;
            }
        }
    }
    assert(
        any_other_bias_changed,
        "Training step must update FC biases for non-target classes too",
    );
}

// INVARIANT: Training input X with label L must increase P(L|X).
// Tests the invariant for digit "2" trained to be 2
#[test]
unconstrained fn test_invariant_train_2_increases_p2() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[10] = 1000;
    input_pixels[20] = 1500;
    input_pixels[30] = 800;

    let label: Field = 2;
    let weights = create_trained_weights_cnn();
    let biases = create_trained_biases_cnn();
    let old_network = arch_cnn_gap::NetworkWeights { weights, biases };

    let probs_before = softmax::softmax(arch_cnn_gap::compute_forward_pass(
        input_pixels,
        old_network,
    ));
    let new_network = arch_cnn_gap::compute_new_weights(
        arch_cnn_gap::TrainingExample { image: input_pixels, label },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );
    let probs_after = softmax::softmax(arch_cnn_gap::compute_forward_pass(input_pixels, new_network));

    assert(
        (probs_after[label as u32] as u64) > (probs_before[label as u32] as u64),
        "INVARIANT violated: training 2->2 must increase P(2|input)",
    );
}

// ============================================================================
// get_activations flow (conv -> relu -> gap)
// ============================================================================

#[test]
unconstrained fn test_get_activations_flow_conv_and_gap_shapes() {
    let input = utils::create_small_test_input();
    let weights = create_test_weights_cnn();
    let biases = create_test_biases_cnn();

    let mut conv_weights: [Field; CONV_WEIGHTS] = [0; CONV_WEIGHTS];
    for i in 0..CONV_WEIGHTS {
        conv_weights[i] = weights[i];
    }
    let mut conv_biases: [Field; CONV_BIASES] = [0; CONV_BIASES];
    for i in 0..CONV_BIASES {
        conv_biases[i] = biases[i];
    }

    let conv_out: [Field; CONV_OUT_SIZE] = conv::conv2d::<CONV_IN_H, CONV_IN_W, CONV_C_IN, CONV_C_OUT, CONV_K, CONV_P, CONV_S, CONV_OUT_H, CONV_OUT_W>(
        input,
        conv_weights,
        conv_biases,
    );

    let mut conv_relu: [Field; CONV_OUT_SIZE] = [0; CONV_OUT_SIZE];
    for i in 0..CONV_OUT_SIZE {
        conv_relu[i] = relu(conv_out[i]);
    }

    let gap_out: [Field; GAP_SIZE] =
        pooling::global_avg_pool::<CONV_OUT_H, CONV_OUT_W, CONV_C_OUT>(conv_relu);

    assert(conv_relu.len() == 75, "conv_relu must have 75 elements (5x5x3)");
    assert(gap_out.len() == 3, "gap_out must have 3 elements");
}

// ============================================================================
// Helper functions
// ============================================================================

unconstrained fn create_test_weights_cnn() -> [Field; 78] {
    let mut weights: [Field; 78] = [0; 78];
    for i in 0..78 {
        weights[i] = (i as Field) * 50;
    }
    weights
}

unconstrained fn create_test_biases_cnn() -> [Field; 13] {
    let mut biases: [Field; 13] = [0; 13];
    for i in 0..13 {
        biases[i] = (i as Field) * 1000;
    }
    biases
}

unconstrained fn create_trained_weights_cnn() -> [Field; 78] {
    let mut weights: [Field; 78] = [0; 78];
    for i in 0..78 {
        weights[i] = (i % 10) as Field;
    }
    weights
}

unconstrained fn create_trained_biases_cnn() -> [Field; 13] {
    let mut biases: [Field; 13] = [0; 13];
    for i in 0..13 {
        biases[i] = (i % 5) as Field;
    }
    biases
}

unconstrained fn compute_forward_pass_pure_cnn(
    input: [Field; 64],
    weights: [Field; 78],
    biases: [Field; 13],
) -> [Field; 10] {
    let network_weights = arch_cnn_gap::NetworkWeights { weights, biases };
    arch_cnn_gap::compute_forward_pass(input, network_weights)
}
