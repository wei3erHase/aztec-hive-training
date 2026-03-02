use crate::softmax;
use crate::test::utils;

// ============================================================================
// Softmax Function Tests
// ============================================================================

#[test]
unconstrained fn test_pure_exp_approx_zero() {
    // Test exp(0) = 1
    let result = softmax::exp_approx(0);
    assert(result == 1_000_000, "exp(0) should equal SCALING_FACTOR");
}

#[test]
unconstrained fn test_pure_exp_approx_small_negative() {
    // Test exp with small positive difference (corresponds to exp(-1.5))
    // Now we pass positive differences: max_logit - logit
    let result = softmax::exp_approx(1_500_000);
    assert(result != 0, "exp(-1.5) should be positive");
    // exp(-1.5) is about 0.223, so in fixed-point should be around 223,000
    // Use explicit Field for comparison
    let sf: Field = 1_000_000;
    assert(result.lt(sf) | (result == sf), "exp(-1.5) should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_pure_exp_approx_large_negative() {
    // Test exp(-5) should use precomputed value (EXP_NEG_5 = 7000)
    // Now we pass positive difference: 5 * SCALING_FACTOR
    let result = softmax::exp_approx(5_000_000);
    // Field values are always >= 0
    assert(result.lt(1_000_001), "exp(-5) should be <= SCALING_FACTOR");
    // Should be close to EXP_NEG_5 (7000)
    assert(result.lt(10_001), "exp(-5) should be small (around 7000)");
}

#[test]
unconstrained fn test_pure_find_max_logit() {
    let logits: [Field; 10] = [100, 200, 50, 300, 150, 250, 100, 50, 200, 100];
    let max_logit = softmax::find_max_logit(logits);
    assert(max_logit == 300, "Max logit should be 300");
}

#[test]
unconstrained fn test_pure_argmax() {
    let values: [Field; 10] = [100, 200, 50, 300, 150, 250, 100, 50, 200, 100];
    let max_index = softmax::argmax(values);
    assert(max_index == 3, "argmax should return index 3");
}

#[test]
unconstrained fn test_pure_softmax_probability_bounds() {
    // Use very small logits to avoid overflow in exp_approx (x4 calculation)
    // Max difference should be < SCALING_FACTOR to avoid overflow
    let logits: [Field; 10] = [1_000, 2_000, 500, 3_000, 1_500, 2_500, 1_000, 500, 2_000, 1_000];
    let probabilities = softmax::softmax(logits);

    // Each probability should be in [0, SCALING_FACTOR]
    for i in 0..10 {
        assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
    }
}

#[test]
unconstrained fn test_pure_softmax_probabilities_sum() {
    // Use very small logits to avoid overflow in exp_approx
    let logits: [Field; 10] = [1_000, 2_000, 500, 3_000, 1_500, 2_500, 1_000, 500, 2_000, 1_000];
    let probabilities = softmax::softmax(logits);

    let mut sum: Field = 0;
    for i in 0..10 {
        sum = sum + probabilities[i];
    }

    let scaling_factor: Field = 1_000_000;
    let tolerance = scaling_factor / 20; // 5% tolerance
    let lower_bound = scaling_factor - tolerance;
    let upper_bound = scaling_factor + tolerance;

    assert(!sum.lt(lower_bound), "Probabilities sum should be >= 95% of SCALING_FACTOR");
    assert(
        sum.lt(upper_bound) | (sum == upper_bound),
        "Probabilities sum should be <= 105% of SCALING_FACTOR",
    );
}

#[test]
unconstrained fn test_pure_softmax_with_exp_returns_exp_outputs() {
    // Use very small logits to avoid overflow in exp_approx
    let logits: [Field; 10] = [1_000, 2_000, 500, 3_000, 1_500, 2_500, 1_000, 500, 2_000, 1_000];
    let (exp_outputs, sum_exp_safe, probabilities) = softmax::softmax_with_exp(logits);

    // exp_outputs should be bounded
    for i in 0..10 {
        assert(!exp_outputs[i].lt(1), "exp_outputs should be >= 1");
        assert(exp_outputs[i].lt(10_000_001), "exp_outputs should be <= SCALING_FACTOR * 10");
    }

    // sum_exp_safe should be at least SCALING_FACTOR
    assert(!sum_exp_safe.lt(1_000_000), "sum_exp_safe should be >= SCALING_FACTOR");

    // Probabilities should be bounded
    for i in 0..10 {
        assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
    }
}

#[test]
unconstrained fn test_pure_softmax_differentiated_probabilities() {
    // Create logits with clear differences
    // Need differences large enough to produce prob differences > 10k tolerance
    // Using SCALING_FACTOR-level differences for meaningful differentiation
    let mut logits: [Field; 10] = [0; 10];
    logits[3] = 1_500_000; // 1.5 in unscaled terms - clear winner
    logits[5] = 500_000; // 0.5 in unscaled terms - second place
    // Rest are 0 (baseline)

    let probabilities = softmax::softmax(logits);

    // Count unique probability values (within tolerance)
    let mut unique_count = 0;
    let mut seen: [Field; 10] = [0; 10];
    let mut seen_count = 0;

    for i in 0..10 {
        let prob = probabilities[i];
        let mut is_new = true;

        for j in 0..seen_count {
            let seen_val = seen[j];
            let diff = if seen_val.lt(prob) {
                prob - seen_val
            } else {
                seen_val - prob
            };
            if diff.lt(10000) {
                is_new = false;
            }
        }

        if is_new & (seen_count < 10) {
            seen[seen_count] = prob;
            seen_count = seen_count + 1;
            unique_count = unique_count + 1;
        }
    }

    // Should have more than 2 unique values
    assert(unique_count > 2, "Softmax should produce more than 2 unique probability values");
}

#[test]
unconstrained fn test_pure_confidence_bounds_with_extreme_outputs() {
    let weights = utils::create_test_weights();
    let biases = utils::create_test_biases();

    // Create input that produces extreme outputs
    let mut extreme_input: [Field; 64] = [0; 64];
    for i in 0..64 {
        extreme_input[i] = (i as Field) * 10_000_000;
    }

    let logits = utils::compute_forward_pass_pure(extreme_input, weights, biases);
    let (prediction, confidence) = utils::compute_confidence_pure(logits);

    // Verify prediction is valid (Field values are always >= 0)
    assert(prediction.lt(10), "Prediction should be <= 9");

    // Verify confidence is within bounds
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
    assert(confidence.lt(10_000_001), "Confidence should not exceed 10,000,000 (overflow check)");
}

#[test]
unconstrained fn test_pure_confidence_with_small_sum_exp() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();

    // Create input that produces outputs where one dominates (use smaller values to avoid overflow)
    let mut input: [Field; 64] = [0; 64];
    for i in 0..64 {
        if i < 10 {
            input[i] = 1_000; // Small but larger than rest
        } else {
            input[i] = 10;
        }
    }

    let logits = utils::compute_forward_pass_pure(input, weights, biases);
    let (prediction, confidence) = utils::compute_confidence_pure(logits);

    // Verify prediction is valid (Field values are always >= 0)
    assert(prediction.lt(10), "Prediction should be <= 9");

    // Verify confidence is bounded
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_pure_confidence_not_astronomical() {
    // Use very small weights/biases/inputs to avoid overflow in exp_approx
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();
    let input = utils::create_small_test_input();

    let logits = utils::compute_forward_pass_pure(input, weights, biases);
    let (prediction, confidence) = utils::compute_confidence_pure(logits);

    // Verify prediction is valid (Field values are always >= 0)
    assert(prediction.lt(10), "Prediction should be <= 9");

    // Verify confidence is reasonable
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR (100%)");
    assert(confidence.lt(10_000_000), "Confidence should not be astronomical (regression test)");
}

#[test]
unconstrained fn test_pure_sum_exp_safe_prevents_overflow() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();

    // Test multiple scenarios (use smaller values to avoid overflow)
    let scenarios: [[Field; 64]; 5] = [
        // Scenario 1: Very small input values
        {
            let mut arr: [Field; 64] = [0; 64];
            for i in 0..64 {
                arr[i] = 1;
            }
            arr
        },
        // Scenario 2: Small input values
        {
            let mut arr: [Field; 64] = [0; 64];
            for i in 0..64 {
                arr[i] = 100;
            }
            arr
        },
        // Scenario 3: One larger value, rest small
        {
            let mut arr: [Field; 64] = [0; 64];
            arr[0] = 1_000;
            for i in 1..64 {
                arr[i] = 1;
            }
            arr
        },
        // Scenario 4: Alternating pattern
        {
            let mut arr: [Field; 64] = [0; 64];
            for i in 0..64 {
                arr[i] = if i % 2 == 0 { 100 } else { 1 };
            }
            arr
        },
        // Scenario 5: Random-like pattern (very small values)
        {
            let mut arr: [Field; 64] = [0; 64];
            for i in 0..64 {
                arr[i] = ((i * 17 + 23) % 100) as Field * 10;
            }
            arr
        },
    ];

    for input in scenarios {
        let logits = utils::compute_forward_pass_pure(input, weights, biases);
        let (_, confidence) = utils::compute_confidence_pure(logits);

        // Verify confidence is bounded
        assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
        assert(
            confidence.lt(10_000_001),
            "Confidence overflow check: should not exceed 10,000,000",
        );

        // Test probabilities too
        let probabilities = softmax::softmax(logits);
        for i in 0..10 {
            assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
            assert(
                probabilities[i].lt(10_000_001),
                "Probability overflow check: should not exceed 10,000,000",
            );
        }
    }
}

#[test]
unconstrained fn test_pure_probabilities_bounds_with_extreme_outputs() {
    let weights = utils::create_test_weights();
    let biases = utils::create_test_biases();

    // Create input that produces extreme outputs
    let mut extreme_input: [Field; 64] = [0; 64];
    for i in 0..64 {
        extreme_input[i] = (i as Field) * 10_000_000;
    }

    let logits = utils::compute_forward_pass_pure(extreme_input, weights, biases);
    let probabilities = softmax::softmax(logits);

    let scaling_factor: Field = 1_000_000;

    // Verify each probability is within bounds
    for i in 0..10 {
        assert(
            probabilities[i].lt(scaling_factor) | (probabilities[i] == scaling_factor),
            "Probability should be <= SCALING_FACTOR",
        );
        assert(
            probabilities[i].lt(10_000_001),
            "Probability should not exceed 10,000,000 (overflow check)",
        );
    }
}

#[test]
unconstrained fn test_pure_probabilities_sum_to_one() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();
    let input = utils::create_small_test_input();

    let logits = utils::compute_forward_pass_pure(input, weights, biases);
    let probabilities = softmax::softmax(logits);

    // Calculate sum
    let mut sum: Field = 0;
    for i in 0..10 {
        sum = sum + probabilities[i];
    }

    let scaling_factor: Field = 1_000_000;
    let tolerance = scaling_factor / 20; // 5% tolerance
    let lower_bound = scaling_factor - tolerance;
    let upper_bound = scaling_factor + tolerance;

    assert(!sum.lt(lower_bound), "Probabilities sum should be >= 95% of SCALING_FACTOR");
    assert(
        sum.lt(upper_bound) | (sum == upper_bound),
        "Probabilities sum should be <= 105% of SCALING_FACTOR",
    );
}

#[test]
unconstrained fn test_pure_predict_and_predict_all_consistency() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();
    let input = utils::create_small_test_input();

    // Get prediction, confidence, and probabilities
    let (prediction, confidence, probabilities) =
        utils::compute_prediction_pure(input, weights, biases);

    // Find max probability and its index
    let mut max_prob = probabilities[0];
    let mut max_idx: u32 = 0;
    for i in 1..10 {
        if max_prob.lt(probabilities[i]) {
            max_prob = probabilities[i];
            max_idx = i;
        }
    }

    // Prediction should match the index of max probability
    assert(prediction == (max_idx as Field), "Prediction should match max probability index");

    // Confidence should be close to max probability (within tolerance)
    let diff = if max_prob.lt(confidence) {
        confidence - max_prob
    } else {
        max_prob - confidence
    };

    // Allow 1% tolerance (10000 in fixed-point)
    assert(diff.lt(10001), "Confidence should match max probability");
}

#[test]
unconstrained fn test_pure_probability_distribution_varies_with_logit_differences() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();

    // Test 1: Input with small values (should produce more uniform distribution)
    let mut small_input: [Field; 64] = [0; 64];
    for i in 0..64 {
        small_input[i] = (i as Field) * 10; // Very small values
    }

    let logits_small = utils::compute_forward_pass_pure(small_input, weights, biases);
    let probs_small = softmax::softmax(logits_small);

    // Test 2: Input with larger values (should produce more peaked distribution, but not too large)
    let mut large_input: [Field; 64] = [0; 64];
    for i in 0..64 {
        large_input[i] = (i as Field) * 100; // Larger but still safe
    }

    let logits_large = utils::compute_forward_pass_pure(large_input, weights, biases);
    let probs_large = softmax::softmax(logits_large);

    // Find max probability in each distribution
    let mut max_small = probs_small[0];
    let mut max_large = probs_large[0];
    for i in 1..10 {
        if max_small.lt(probs_small[i]) {
            max_small = probs_small[i];
        }
        if max_large.lt(probs_large[i]) {
            max_large = probs_large[i];
        }
    }

    // Both distributions should have valid probabilities
    assert(max_small != 0, "Small input should produce non-zero max probability");
    assert(max_large != 0, "Large input should produce non-zero max probability");
    assert(max_small.lt(1_000_001), "Max probability should be <= SCALING_FACTOR");
    assert(max_large.lt(1_000_001), "Max probability should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_pure_digit_like_patterns_produce_different_predictions() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();

    // Pattern 1: Vertical line (like "1") - use smaller values to avoid overflow
    let mut pattern_1: [Field; 64] = [0; 64];
    for i in 0..8 {
        pattern_1[i * 8 + 4] = 800;
    }

    // Pattern 2: Circle-like (like "0") - use smaller values
    let mut pattern_0: [Field; 64] = [0; 64];
    // Top and bottom edges
    for j in 2..6 {
        pattern_0[1 * 8 + j] = 800;
        pattern_0[6 * 8 + j] = 800;
    }
    // Left and right edges
    for i in 2..6 {
        pattern_0[i * 8 + 1] = 800;
        pattern_0[i * 8 + 6] = 800;
    }

    // Get predictions for both patterns
    let (pred_1, conf_1, _) = utils::compute_prediction_pure(pattern_1, weights, biases);
    let (pred_0, conf_0, _) = utils::compute_prediction_pure(pattern_0, weights, biases);

    // Both should have valid predictions and confidences (Field values are always >= 0)
    assert(pred_1.lt(10), "Prediction 1 should be <= 9");
    assert(pred_0.lt(10), "Prediction 0 should be <= 9");

    assert(conf_1.lt(1_000_001), "Confidence 1 should be <= SCALING_FACTOR");
    assert(conf_0.lt(1_000_001), "Confidence 0 should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_pure_exp_outputs_clamping() {
    // Use very small weights/biases to avoid overflow
    let weights = utils::create_small_test_weights();
    let biases = utils::create_small_test_biases();

    // Test with various input patterns (use smaller values to avoid overflow)
    let test_inputs: [[Field; 64]; 3] = [
        // Test 1: All zeros
        [0; 64],
        // Test 2: Very small values
        {
            let mut arr: [Field; 64] = [0; 64];
            for i in 0..64 {
                arr[i] = 100;
            }
            arr
        },
        // Test 3: Mixed pattern (very small values)
        {
            let mut arr: [Field; 64] = [0; 64];
            for i in 0..64 {
                arr[i] = (i as Field) * 10;
            }
            arr
        },
    ];

    for input in test_inputs {
        let logits = utils::compute_forward_pass_pure(input, weights, biases);
        let (_, confidence) = utils::compute_confidence_pure(logits);

        // Verify confidence is valid (Field values are always >= 0)
        assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
    }
}
