use crate::{architectures::arch_64_16_10, packing, softmax};
use crate::test::utils;

// Architecture: 2-layer (64 -> 16 -> 10)
global TOTAL_WEIGHTS: u32 = 1184; // 64*16 + 16*10
global WEIGHTS_PER_FIELD: u32 = 28; // 9-bit packing
global PACKED_WEIGHT_FIELDS: u32 = 43; // ceil(1184 / 28)

// Training parameters (implementation-specific)
global LEARNING_RATE: Field = 5000; // 0.005 gentler update
global REGULARIZATION: Field = 1000; // 0.001 in fixed-point (0.001 * 1000000)

// ============================================================================
// Forward Pass Tests
// ============================================================================

#[test]
unconstrained fn test_pure_forward_pass_basic() {
    let input = utils::create_small_test_input();
    let weights = utils::create_test_weights();
    let biases = utils::create_test_biases();

    let outputs = utils::compute_forward_pass_pure(input, weights, biases);

    // argmax must be a valid class index
    let max_idx = softmax::argmax(outputs);
    assert((max_idx as Field).lt(10), "forward pass must produce a valid argmax in [0, 9]");

    // With non-zero inputs and weights, at least one output must be non-zero
    let mut all_zero = true;
    for i in 0..10 {
        if outputs[i] != 0 {
            all_zero = false;
        }
    }
    assert(!all_zero, "Non-zero inputs and weights should produce at least one non-zero output");
}

#[test]
unconstrained fn test_pure_forward_pass_with_zero_input() {
    let input: [Field; 64] = [0; 64];
    let weights = utils::create_test_weights();
    let biases = utils::create_test_biases();

    let outputs = utils::compute_forward_pass_pure(input, weights, biases);

    // With zero input the 2-layer computation is:
    //   hidden[j] = ReLU(bias1[j]) - all positive since bias1[j] = j * 10_000
    //   logit[k]  = f(hidden, W2, bias2)
    // So at least the hidden layer is non-trivially non-zero, and that propagates.
    // Verify argmax is valid and outputs are not all identical (non-trivial propagation).
    let max_idx = softmax::argmax(outputs);
    assert((max_idx as Field).lt(10), "argmax must be a valid class index even with zero input");

    // Outputs should not all be equal (the 2-layer network produces differentiated logits)
    let mut all_equal = true;
    for i in 1..10 {
        if outputs[i] != outputs[0] {
            all_equal = false;
        }
    }
    assert(
        !all_equal,
        "2-layer forward pass should produce differentiated logits with non-zero biases",
    );
}

#[test]
unconstrained fn test_pure_forward_pass_fixed_point_scaling() {
    // Test that fixed-point multiplication is working correctly
    // For 2-layer network: Layer 1 output contributes to Layer 2 input
    let mut input: [Field; 64] = [0; 64];
    input[0] = 1_000_000; // SCALING_FACTOR

    // 2-layer network: 1184 weights (64*16 + 16*10), 26 biases (16 + 10)
    let mut weights: [Field; 1184] = [0; 1184];
    // Set first weight in Layer 1 (input 0 -> hidden 0)
    weights[0] = 1_000_000; // SCALING_FACTOR
    // Set first weight in Layer 2 (hidden 0 -> output 0)
    weights[1024] = 1_000_000; // SCALING_FACTOR

    let biases: [Field; 26] = [0; 26];

    let outputs = utils::compute_forward_pass_pure(input, weights, biases);

    // With ReLU activation on hidden layer, the signal should propagate
    // First output should be approximately SCALING_FACTOR (1.0 * 1.0 * 1.0 after ReLU)
    let first_output = outputs[0];
    let expected: Field = 1_000_000;
    let diff = if expected.lt(first_output) {
        first_output - expected
    } else {
        expected - first_output
    };
    assert(diff.lt(1001), "Fixed-point multiplication should work correctly");
}

// ============================================================================
// Training Tests
// ============================================================================

#[test]
unconstrained fn test_pure_compute_new_weights_updates_target_bias() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[0] = 100000;
    input_pixels[10] = 200000;
    input_pixels[20] = 150000;

    let label: Field = 3;
    let weights = utils::create_trained_weights();
    let biases = utils::create_trained_biases();
    let old_network = arch_64_16_10::NetworkWeights { weights, biases };

    let new_network = arch_64_16_10::compute_new_weights(
        arch_64_16_10::TrainingExample { image: input_pixels, label },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );

    // After one gradient step the bias for the target class must change.
    // Output biases are stored at biases[HIDDEN_SIZE + label] (indices 16..25).
    // (cross-entropy gradient pushes bias[label] toward 1 - P(label))
    assert(
        new_network.biases[16 + label as u32] != old_network.biases[16 + label as u32],
        "Training step must update the output bias for the target class",
    );
    // L2 weights (indices 1024..1183) are directly tied to output gradients and must change.
    let mut any_weight_changed = false;
    for i in 0..10 {
        if new_network.weights[1024 + i] != old_network.weights[1024 + i] {
            any_weight_changed = true;
        }
    }
    assert(any_weight_changed, "Training step must update at least some L2 weights");
}

#[test]
unconstrained fn test_pure_compute_new_weights_different_labels_produce_different_updates() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[0] = 100000;

    let weights = utils::create_trained_weights();
    let biases = utils::create_trained_biases();
    let old_network = arch_64_16_10::NetworkWeights { weights, biases };

    let new_network0 = arch_64_16_10::compute_new_weights(
        arch_64_16_10::TrainingExample { image: input_pixels, label: 0 },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );
    let new_network5 = arch_64_16_10::compute_new_weights(
        arch_64_16_10::TrainingExample { image: input_pixels, label: 5 },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );
    let new_network9 = arch_64_16_10::compute_new_weights(
        arch_64_16_10::TrainingExample { image: input_pixels, label: 9 },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );

    // Output biases are at biases[16..25] (HIDDEN_SIZE + output_class_index).
    // Training on different labels must yield different output-bias updates for those classes.
    assert(
        new_network0.biases[16] != new_network5.biases[16],
        "output bias[0]: training on label=0 vs label=5 must produce different updates",
    );
    assert(
        new_network0.biases[21] != new_network5.biases[21],
        "output bias[5]: training on label=0 vs label=5 must produce different updates",
    );
    assert(
        new_network5.biases[25] != new_network9.biases[25],
        "output bias[9]: training on label=5 vs label=9 must produce different updates",
    );
}

// INVARIANT: Training input X with label L must increase P(L|X).
// Tests the invariant for label 2 (digit "2" trained to be 2)
#[test]
unconstrained fn test_invariant_train_increases_correct_class_probability() {
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[10] = 200000;
    input_pixels[20] = 150000;
    input_pixels[30] = 180000;

    let label: Field = 2; // digit "2"
    let weights = utils::create_trained_weights();
    let biases = utils::create_trained_biases();
    let old_network = arch_64_16_10::NetworkWeights { weights, biases };

    let probs_before = softmax::softmax(arch_64_16_10::compute_forward_pass(
        input_pixels,
        old_network,
    ));
    let new_network = arch_64_16_10::compute_new_weights(
        arch_64_16_10::TrainingExample { image: input_pixels, label },
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );
    let probs_after = softmax::softmax(arch_64_16_10::compute_forward_pass(
        input_pixels,
        new_network,
    ));

    assert(
        (probs_after[label as u32] as u64) > (probs_before[label as u32] as u64),
        "INVARIANT violated: training 2->2 must increase P(2|input)",
    );
}

#[test]
unconstrained fn test_training_decreases_incorrect_label_probabilities() {
    // Create a test input
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[10] = 200000;
    input_pixels[20] = 150000;

    // Use label 2 (digit 2)
    let label: Field = 2;

    // Use trained weights from utils
    let weights = utils::create_trained_weights();
    let biases = utils::create_trained_biases();
    let old_network = arch_64_16_10::NetworkWeights { weights, biases };

    // Make prediction BEFORE training
    let logits_before = arch_64_16_10::compute_forward_pass(input_pixels, old_network);
    let probabilities_before = softmax::softmax(logits_before);

    // Train on this input with correct label
    let training_example = arch_64_16_10::TrainingExample { image: input_pixels, label };
    let new_network = arch_64_16_10::compute_new_weights(
        training_example,
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );

    // Make prediction AFTER training
    let logits_after = arch_64_16_10::compute_forward_pass(input_pixels, new_network);
    let probabilities_after = softmax::softmax(logits_after);

    // Verify that probabilities for incorrect labels DECREASED (or stayed same)
    for i in 0..10 {
        if i != (label as u32) {
            let prob_before = probabilities_before[i];
            let prob_after = probabilities_after[i];
            // prob_after should be <= prob_before (decreased or stayed same)
            // Since probabilities are always positive, we can use u64 comparison
            let prob_before_u64 = prob_before as u64;
            let prob_after_u64 = prob_after as u64;
            assert(
                prob_after_u64 <= prob_before_u64,
                "Training should decrease probability for incorrect labels",
            );
        }
    }
}

#[test]
unconstrained fn test_submit_training_input_hash_computation() {
    // Create test input
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[0] = 100000;
    input_pixels[10] = 200000;

    // Use hardcoded trained weights
    let current_weights = utils::create_trained_weights();
    let current_biases = utils::create_trained_biases();

    // Simulate submit_training_input hash computation logic
    // This matches the logic in submit_training_input: pack weights/biases and hash them
    let mut current_weights_hash: Field = 0;

    // Pack weights into Fields and hash them (matching submit_training_input)
    for packed_idx in 0..PACKED_WEIGHT_FIELDS {
        let mut weight_batch: [Field; 28] = [0; 28];
        let start_idx = packed_idx * WEIGHTS_PER_FIELD;

        // Collect 28 weights for this packed Field
        for i in 0..WEIGHTS_PER_FIELD {
            let global_idx = start_idx + i;
            if global_idx < TOTAL_WEIGHTS {
                weight_batch[i] = current_weights[global_idx];
            }
        }

        // Pack and add to hash
        let packed = packing::pack(weight_batch);
        current_weights_hash = current_weights_hash + packed;
    }

    // Pack biases and add to hash
    let packed_biases = packing::pack(current_biases);
    current_weights_hash = current_weights_hash + packed_biases;

    // If we get here, the hash computation didn't revert - test passes
    assert(current_weights_hash != 0, "Hash should be non-zero");
}

#[test]
unconstrained fn test_submit_training_input_full_flow() {
    // Create test input
    let mut input_pixels: [Field; 64] = [0; 64];
    input_pixels[0] = 100000;
    input_pixels[10] = 200000;
    let label: Field = 5;

    // Use hardcoded trained weights
    let current_weights = utils::create_trained_weights();
    let current_biases = utils::create_trained_biases();

    // Step 1: Compute new weights (matching submit_training_input)
    let training_example = arch_64_16_10::TrainingExample { image: input_pixels, label };
    let old_network =
        arch_64_16_10::NetworkWeights { weights: current_weights, biases: current_biases };
    let new_network = arch_64_16_10::compute_new_weights(
        training_example,
        old_network,
        LEARNING_RATE,
        REGULARIZATION,
    );

    // Output biases are at biases[16..25] (HIDDEN_SIZE + output_class_index).
    // Training must update the output bias for the target label.
    assert(
        new_network.biases[16 + label as u32] != old_network.biases[16 + label as u32],
        "Training step must update the output bias for the target label",
    );

    // L2 weights (indices 1024..1183) are directly tied to output gradients and must change.
    let mut any_weight_changed = false;
    for i in 0..10 {
        if new_network.weights[1024 + i] != old_network.weights[1024 + i] {
            any_weight_changed = true;
        }
    }
    assert(any_weight_changed, "Training step must update at least some L2 weights");

    // Step 2: Verify packed hash from new weights is non-zero (matches submit_training_input commit)
    let mut new_weights_hash: Field = 0;
    for packed_idx in 0..PACKED_WEIGHT_FIELDS {
        let mut weight_batch: [Field; 28] = [0; 28];
        let start_idx = packed_idx * WEIGHTS_PER_FIELD;
        for i in 0..WEIGHTS_PER_FIELD {
            let global_idx = start_idx + i;
            if global_idx < TOTAL_WEIGHTS {
                weight_batch[i] = new_network.weights[global_idx];
            }
        }
        let packed = packing::pack(weight_batch);
        new_weights_hash = new_weights_hash + packed;
    }
    let packed_biases = packing::pack(new_network.biases);
    new_weights_hash = new_weights_hash + packed_biases;

    assert(new_weights_hash != 0, "New weights hash must be non-zero");
}
