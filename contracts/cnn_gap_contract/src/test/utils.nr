use crate::CNNGAP;
use aztec::{protocol::address::AztecAddress, test::helpers::test_environment::TestEnvironment};

// Architecture: CNN+GAP (Conv 4x4x3, ReLU, GAP, FC 3->10)
// 78 weights, 13 biases

/// Setup function to deploy contract with empty constructor
pub unconstrained fn deploy_contract(
    env: &mut TestEnvironment,
    owner: AztecAddress,
) -> AztecAddress {
    let initializer_call_interface = CNNGAP::interface().constructor();
    let contract_address =
        env.deploy("CNNGAP").with_public_initializer(owner, initializer_call_interface);
    contract_address
}

/// Setup function that deploys contract with zero weights
pub unconstrained fn setup_contract() -> (&mut TestEnvironment, AztecAddress, AztecAddress) {
    let mut env = TestEnvironment::new();
    let owner = env.create_light_account();
    let contract_address = deploy_contract(&mut env, owner);
    (&mut env, contract_address, owner)
}
