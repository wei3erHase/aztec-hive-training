use crate::{SingleLayer, test::utils};

// ============================================================================
// SingleLayer Contract Integration Tests
// ============================================================================
// These tests verify contract deployment, storage, and end-to-end flows.
// Pure logic tests (forward pass, training) are in zkml arch_64_10_tests.

#[test]
unconstrained fn test_deploy_contract_initializes_to_zero() {
    let mut env = aztec::test::helpers::test_environment::TestEnvironment::new();
    let owner = env.create_light_account();

    let contract_address = utils::deploy_contract(&mut env, owner);

    let weight_0 = env.view_public(SingleLayer::at(contract_address).get_weight(0));
    assert(weight_0 == 0, "Weight should be initialized to 0");

    let bias_0 = env.view_public(SingleLayer::at(contract_address).get_bias(0));
    assert(bias_0 == 0, "Bias should be initialized to 0");
}

#[test]
unconstrained fn test_deploy_predict_after_initialization() {
    let (env, contract_address, _owner) = utils::setup_contract();

    let input = utils::create_test_input();

    let (prediction, confidence) =
        env.view_public(SingleLayer::at(contract_address).predict(input));

    assert(prediction.lt(10), "Prediction should be <= 9");
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_deploy_full_flow_deploy_predict() {
    let mut env = aztec::test::helpers::test_environment::TestEnvironment::new();
    let owner = env.create_light_account();

    let contract_address = utils::deploy_contract(&mut env, owner);

    let weight_0 = env.view_public(SingleLayer::at(contract_address).get_weight(0));
    assert(weight_0 == 0, "First weight should be zero");

    let bias_0 = env.view_public(SingleLayer::at(contract_address).get_bias(0));
    assert(bias_0 == 0, "First bias should be zero");

    let input = utils::create_test_input();
    let (prediction, confidence) =
        env.view_public(SingleLayer::at(contract_address).predict(input));

    assert(prediction.lt(10), "Prediction should be <= 9");
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_predict_all_outputs() {
    let (env, contract_address, _owner) = utils::setup_contract();

    let input = utils::create_test_input();

    let probabilities: [Field; 10] =
        env.view_public(SingleLayer::at(contract_address).predict_all(input));

    let mut sum: Field = 0;
    for i in 0..10 {
        sum = sum + probabilities[i];
        assert(probabilities[i].lt(1_000_001), "Probability should be <= SCALING_FACTOR");
    }

    assert(sum.lt(10_000_001), "Probabilities should not exceed reasonable bounds");
}

#[test]
unconstrained fn test_predict_and_predict_all_consistency() {
    let (env, contract_address, _owner) = utils::setup_contract();
    let input = utils::create_test_input();

    let (prediction, confidence) =
        env.view_public(SingleLayer::at(contract_address).predict(input));

    let probabilities: [Field; 10] =
        env.view_public(SingleLayer::at(contract_address).predict_all(input));

    let mut max_prob = probabilities[0];
    let mut max_idx: u32 = 0;
    for i in 1..10 {
        if max_prob.lt(probabilities[i]) {
            max_prob = probabilities[i];
            max_idx = i;
        }
    }

    assert(
        prediction == (max_idx as Field),
        "Prediction from predict() should match max probability index from predict_all()",
    );

    let diff = if confidence.lt(max_prob) {
        max_prob - confidence
    } else {
        confidence - max_prob
    };
    assert(diff.lt(10001), "Confidence should match max probability within tolerance");
}

// ============================================================================
// Train Step Tests
// ============================================================================
// Validates that contract is ready for training: correct packed structure,
// and that pretrained weights produce valid predictions.

#[test]
unconstrained fn test_train_step_packed_structure() {
    let (env, contract_address, _owner) = utils::setup_contract();

    let packed_weights =
        env.view_public(SingleLayer::at(contract_address).get_all_packed_weights());
    let packed_biases = env.view_public(SingleLayer::at(contract_address).get_packed_biases());

    assert(
        packed_weights.len() == 23,
        "PACKED_WEIGHT_FIELDS should be 23 for submit_training_input",
    );
    assert(packed_biases.len() == 1, "PACKED_BIAS_FIELDS should be 1 for submit_training_input");
}

#[test]
unconstrained fn test_train_step_predict_with_trained_weights() {
    let (env, contract_address, _owner) = utils::setup_contract_with_trained_weights();

    let input = utils::create_test_input();

    let (prediction, confidence) =
        env.view_public(SingleLayer::at(contract_address).predict(input));

    assert(prediction.lt(10), "Prediction should be valid (0-9)");
    assert(confidence.lt(1_000_001), "Confidence should be <= SCALING_FACTOR");
}

#[test]
unconstrained fn test_train_step_weights_storage() {
    let (env, contract_address, _owner) = utils::setup_contract_with_trained_weights();

    let weight_0 = env.view_public(SingleLayer::at(contract_address).get_weight(0));
    let bias_0 = env.view_public(SingleLayer::at(contract_address).get_bias(0));

    let both_zero = (weight_0 == 0) & (bias_0 == 0);
    assert(!both_zero, "Trained weights should not both be zero (storage written correctly)");
}
