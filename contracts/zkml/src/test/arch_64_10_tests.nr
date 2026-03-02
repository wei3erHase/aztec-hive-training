// Architecture: single-layer (64 -> 10)
// Linear layer only, no hidden layer, no ReLU

use crate::{architectures::arch_64_10, softmax};
use crate::test::utils;

global TOTAL_WEIGHTS: u32 = 640;
global WEIGHTS_PER_FIELD: u32 = 28;
global PACKED_WEIGHT_FIELDS: u32 = 23;

global LEARNING_RATE: Field = 5000; // 0.005 gentler update
global REGULARIZATION: Field = 1000;

// ============================================================================
// Forward Pass Tests
// ============================================================================

#[test]
unconstrained fn test_single_layer_forward_pass_basic() {
    let input = utils::create_small_test_input();
    let weights = create_test_weights_single();
    let biases = create_test_biases_single();

    let outputs = compute_forward_pass_pure_single(input, weights, biases);

    // argmax must be a valid class index
    let max_idx = softmax::argmax(outputs);
    assert((max_idx as Field).lt(10), "forward pass must produce a valid argmax in [0, 9]");

    // With non-zero inputs and weights, the outputs must not all equal the biases
    // (i.e. the weighted-input contribution is non-trivial)
    let mut all_equal_bias = true;
    for i in 0..10 {
        if outputs[i] != biases[i] {
            all_equal_bias = false;
        }
    }
    assert(!all_equal_bias, "Non-zero inputs should produce outputs that differ from biases alone");
}

#[test]
unconstrained fn test_single_layer_forward_pass_zero_input() {
    let input: [Field; 64] = [0; 64];
    let weights = create_test_weights_single();
    // biases[j] = j * 10_000 (see create_test_biases_single)
    let biases = create_test_biases_single();

    let outputs = compute_forward_pass_pure_single(input, weights, biases);

    // With zero input the weighted sum is 0 for every class, so output[j] == bias[j]
    for i in 0..10 {
        assert(outputs[i] == biases[i], "Zero input: output must equal bias for every class");
    }
    // Specifically: bias[0] = 0, bias[9] = 90_000
    assert(outputs[0] == 0, "Class 0 output should be 0 (bias[0] = 0)");
    assert(outputs[9] == 90000, "Class 9 output should be 90000 (bias[9] = 9 * 10000)");
    // argmax of [0, 10000, ..., 90000] is 9
    assert(softmax::argmax(outputs) == 9, "argmax of bias-only outputs should be class 9");
}

#[test]
unconstrained fn test_single_layer_forward_pass_fixed_point() {
    let mut input: [Field; 64] = [0; 64];
    input[0] = 1_000_000;

    let mut weights: [Field; 640] = [0; 640];
    weights[0] = 1_000_000;

    let biases: [Field; 10] = [0; 10];

    let outputs = compute_forward_pass_pure_single(input, weights, biases);
    let first_output = outputs[0];
    let expected: Field = 1_000_000;
    let diff = if expected.lt(first_output) {
        first_output - expected
    } else {
        expected - first_output
    };
    assert(diff.lt(1001), "Fixed-point multiplication should work");
}

// ============================================================================
// Training Tests
// ============================================================================

#[test]
unconstrained fn test_single_layer_compute_new_weights_no_revert() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[0] = 100000;
    input_pixels[10] = 200000;

    let label: Field = 3;
    let weights = create_trained_weights_single();
    let biases = create_trained_biases_single();

    let training_example = arch_64_10::TrainingExample { image: input_pixels, label };
    let old_network = arch_64_10::NetworkWeights { weights, biases };

    let _new_weights = arch_64_10::compute_new_weights(
        training_example,
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );

    assert(true, "compute_new_weights completed without reverting");
}

// INVARIANT: Training input X with label L must increase P(L|X).
#[test]
unconstrained fn test_invariant_train_2_increases_p2() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[10] = 200000;
    input_pixels[20] = 150000;

    let label: Field = 2;
    let weights = create_trained_weights_single();
    let biases = create_trained_biases_single();
    let old_network = arch_64_10::NetworkWeights { weights, biases };

    let probs_before =
        softmax::softmax(arch_64_10::compute_forward_pass(input_pixels, old_network));
    let new_network = arch_64_10::compute_new_weights(
        arch_64_10::TrainingExample { image: input_pixels, label },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );
    let probs_after = softmax::softmax(arch_64_10::compute_forward_pass(input_pixels, new_network));

    assert(
        (probs_after[label as u32] as u64) > (probs_before[label as u32] as u64),
        "INVARIANT violated: training 2->2 must increase P(2|input)",
    );
}

// ============================================================================
// Helper functions
// ============================================================================

unconstrained fn create_test_weights_single() -> [Field; 640] {
    let mut weights: [Field; 640] = [0; 640];
    for i in 0..640 {
        weights[i] = (i as Field) * 100;
    }
    weights
}

unconstrained fn create_test_biases_single() -> [Field; 10] {
    let mut biases: [Field; 10] = [0; 10];
    for i in 0..10 {
        biases[i] = (i as Field) * 10000;
    }
    biases
}

unconstrained fn create_trained_weights_single() -> [Field; 640] {
    let mut weights: [Field; 640] = [0; 640];
    for i in 0..640 {
        weights[i] = (i as Field) * 10;
    }
    weights
}

unconstrained fn create_trained_biases_single() -> [Field; 10] {
    let mut biases: [Field; 10] = [0; 10];
    for i in 0..10 {
        biases[i] = (i as Field) * 100;
    }
    biases
}

unconstrained fn compute_forward_pass_pure_single(
    input: [Field; 64],
    weights: [Field; 640],
    biases: [Field; 10],
) -> [Field; 10] {
    let network_weights = arch_64_10::NetworkWeights { weights, biases };
    arch_64_10::compute_forward_pass(input, network_weights)
}
