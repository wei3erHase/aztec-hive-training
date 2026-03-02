use crate::{CNNGAP, test::utils};

// ============================================================================
// CNNGAP Contract Integration Tests
// ============================================================================
// CNN: 3ch 4x4 for predict/train/explain. Pure logic in zkml arch_cnn_gap_tests.

#[test]
unconstrained fn test_deploy_contract_initializes_to_zero() {
    let mut env = aztec::test::helpers::test_environment::TestEnvironment::new();
    let owner = env.create_light_account();

    let contract_address = utils::deploy_contract(&mut env, owner);

    let weight_0 = env.view_public(CNNGAP::at(contract_address).get_weight(0));
    assert(weight_0 == 0, "Weight should be initialized to 0");

    let bias_0 = env.view_public(CNNGAP::at(contract_address).get_bias(0));
    assert(bias_0 == 0, "Bias should be initialized to 0");
}

#[test]
unconstrained fn test_deploy_storage_reads() {
    let (env, contract_address, _owner) = utils::setup_contract();

    let weight_0 = env.view_public(CNNGAP::at(contract_address).get_weight(0));
    assert(weight_0 == 0, "First weight should be zero");

    let bias_0 = env.view_public(CNNGAP::at(contract_address).get_bias(0));
    assert(bias_0 == 0, "First bias should be zero");

    let packed_weights = env.view_public(CNNGAP::at(contract_address).get_all_packed_weights());
    assert(packed_weights.len() == 3, "Should return 3 packed weight fields (3ch 4x4 CNN)");

    let packed_biases = env.view_public(CNNGAP::at(contract_address).get_packed_biases());
    assert(packed_biases.len() == 1, "Should return 1 packed bias field");
}

// predict/predict_all may exceed local network gas limit (6M). Test via JS: tests/app/architectures-flow.test.ts
// get_activations: Logic tested in zkml arch_cnn_gap_tests (test_get_activations_flow_conv_and_gap_shapes)
// Contract call tested in tests/app/architectures-flow.test.ts
