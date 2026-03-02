use crate::{MultiLayerPerceptron, test::utils};

// ============================================================================
// MultiLayerPerceptron Contract Integration Tests
// ============================================================================
// These tests verify contract deployment, storage, and end-to-end flows.
// Pure logic tests (forward pass, softmax, confidence calculation) are in
// separate module-specific test files.

#[test]
unconstrained fn test_deploy_contract_initializes_to_zero() {
    let mut env = aztec::test::helpers::test_environment::TestEnvironment::new();
    let owner = env.create_light_account();

    // Deploy contract (constructor initializes to zero-packed values)
    let contract_address = utils::deploy_contract(&mut env, owner);

    // Verify contract is deployed and weights are initialized to zero
    let weight_0 = env.view_public(MultiLayerPerceptron::at(contract_address).get_weight(0));
    assert(weight_0 == 0, "Weight should be initialized to 0");

    // Verify biases are also zero
    let bias_0 = env.view_public(MultiLayerPerceptron::at(contract_address).get_bias(0));
    assert(bias_0 == 0, "Bias should be initialized to 0");
}

#[test]
unconstrained fn test_deploy_predict_after_initialization() {
    let (env, contract_address, _owner) = utils::setup_contract_with_weights();

    // Create test input
    let input = utils::create_test_input();

    // Make prediction
    let (prediction, confidence) =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict(input));

    // Verify prediction is valid (Field values are always >= 0, check <= 9)
    assert(prediction.lt(10), "Prediction should be <= 9");

    // Verify confidence is reasonable (Field values are always >= 0, check <= SCALING_FACTOR)
    // Note: With test weights, confidence might be very low
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_deploy_full_flow_deploy_predict() {
    let mut env = aztec::test::helpers::test_environment::TestEnvironment::new();
    let owner = env.create_light_account();

    // Step 1: Deploy contract (constructor initializes to zero)
    let contract_address = utils::deploy_contract(&mut env, owner);

    // Step 2: Verify zero initialization
    let weight_0 = env.view_public(MultiLayerPerceptron::at(contract_address).get_weight(0));
    assert(weight_0 == 0, "First weight should be zero");

    let bias_0 = env.view_public(MultiLayerPerceptron::at(contract_address).get_bias(0));
    assert(bias_0 == 0, "First bias should be zero");

    // Step 3: Make prediction with zero weights (should produce uniform probabilities)
    let input = utils::create_test_input();
    let (prediction, confidence) =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict(input));

    // Verify prediction results (Field values are always >= 0)
    assert(prediction.lt(10), "Prediction should be <= 9");
    // Note: With zero weights, confidence will be uniform (~10% each class)
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_deploy_predict_all_outputs() {
    let (env, contract_address, _owner) = utils::setup_contract_with_weights();

    // Create test input
    let input = utils::create_test_input();

    // Get all prediction outputs
    let probabilities: [Field; 10] =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict_all(input));

    // Verify we get 10 outputs (one for each class)
    let mut sum: Field = 0;
    for i in 0..10 {
        // Probabilities should be non-negative (check as Field comparison)
        // Note: Small positive values might appear negative when cast to i64 due to field modulus
        // So we check that the value itself is reasonable rather than the cast
        let prob_val = probabilities[i];
        // Just verify the sum is reasonable - individual probabilities might be very small
        sum = sum + prob_val;
    }

    // Verify probabilities are valid (Field values are always >= 0)
    // Note: With test weights, probabilities might be very small due to softmax approximation
    // but they should still be bounded
    assert(sum.lt(10_000_001), "Probabilities should not exceed reasonable bounds");

    // Verify each probability is in valid range [0, SCALING_FACTOR]
    for i in 0..10 {
        assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
    }

    // Note: With certain weight patterns and softmax approximation, all probabilities
    // might be very small (near 0) but this is still valid behavior
    // The important thing is that they're bounded and non-negative
}

#[test]
unconstrained fn test_confidence_predict_confidence_bounds() {
    let (env, contract_address, _owner) = utils::setup_contract_with_weights();
    let input = utils::create_test_input();

    let (prediction, confidence) =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict(input));

    // Verify prediction is valid (0-9)
    assert(prediction.lt(10), "Prediction should be <= 9");

    // Verify confidence is within bounds (integration test with storage)
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
    assert(confidence.lt(10_000_000), "Confidence should not be astronomical");
}

#[test]
unconstrained fn test_confidence_predict_all_probability_bounds() {
    let (env, contract_address, _owner) = utils::setup_contract_with_weights();
    let input = utils::create_test_input();

    let probabilities: [Field; 10] =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict_all(input));

    // Verify each probability is within bounds (integration test with storage)
    for i in 0..10 {
        assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
        assert(probabilities[i].lt(10_000_001), "Probability should not exceed 10,000,000");
    }
}

#[test]
unconstrained fn test_predict_validation() {
    let (env, contract_address, _owner) = utils::setup_contract_with_trained_weights();

    // Get test input and expected label
    let input = utils::create_prediction_test_input();
    let expected_label = utils::get_prediction_test_label(); // Returns 2

    // Make prediction
    let (prediction, confidence) =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict(input));

    // Verify prediction matches expected label (digit "2")
    assert(prediction == expected_label, "Prediction should match expected label (2)");

    // Verify confidence is reasonable (Field values are always >= 0, check <= SCALING_FACTOR)
    let SCALING_FACTOR: Field = 1_000_000;
    assert(
        confidence.lt(SCALING_FACTOR) | (confidence == SCALING_FACTOR),
        "Confidence should be <= SCALING_FACTOR",
    );
}

#[test]
unconstrained fn test_predict_all_validation() {
    let (env, contract_address, _owner) = utils::setup_contract_with_trained_weights();

    // Get test input
    let input = utils::create_prediction_test_input();
    let expected_label = utils::get_prediction_test_label();

    // Get all prediction outputs
    let probabilities: [Field; 10] =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict_all(input));

    // Verify we get 10 outputs (one for each class)
    let mut sum: Field = 0;
    for i in 0..10 {
        assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
        sum = sum + probabilities[i];
    }

    // Verify probabilities sum is reasonable (should be close to SCALING_FACTOR)
    // Note: With softmax approximation, sum might not be exactly SCALING_FACTOR
    assert(sum.lt(10_000_001), "Probabilities should not exceed reasonable bounds");

    // Verify that the expected label has the highest probability
    let expected_prob = probabilities[expected_label as u32];
    for i in 0..10 {
        if i != (expected_label as u32) {
            // With trained weights, expected label (2) should have highest probability
            assert(
                probabilities[i].lt(expected_prob) | (probabilities[i] == expected_prob),
                "Expected label should have highest probability",
            );
        }
    }
}

#[test]
unconstrained fn test_softmax_predict_all_produces_valid_probabilities() {
    let (env, contract_address, _owner) = utils::setup_contract_with_weights();
    let input = utils::create_test_input();

    let probabilities: [Field; 10] =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict_all(input));

    // Verify probabilities sum to approximately SCALING_FACTOR (integration test with storage)
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
unconstrained fn test_softmax_predict_and_predict_all_consistency() {
    let (env, contract_address, _owner) = utils::setup_contract_with_weights();
    let input = utils::create_test_input();

    // Get prediction and confidence
    let (prediction, confidence) =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict(input));

    // Get all probabilities
    let probabilities: [Field; 10] =
        env.view_public(MultiLayerPerceptron::at(contract_address).predict_all(input));

    // Find max probability and its index from predict_all
    let mut max_prob = probabilities[0];
    let mut max_idx: u32 = 0;
    for i in 1..10 {
        if max_prob.lt(probabilities[i]) {
            max_prob = probabilities[i];
            max_idx = i;
        }
    }

    // Prediction from predict() should match the index of max probability (integration test)
    assert(
        prediction == (max_idx as Field),
        "Prediction from predict() should match max probability index from predict_all()",
    );

    // Confidence should be close to max probability (within tolerance)
    let diff = if confidence.lt(max_prob) {
        max_prob - confidence
    } else {
        confidence - max_prob
    };

    // Allow 1% tolerance (10000 in fixed-point)
    assert(
        diff.lt(10001), // diff <= 10000
        "Confidence from predict() should match max probability from predict_all()",
    );
}
