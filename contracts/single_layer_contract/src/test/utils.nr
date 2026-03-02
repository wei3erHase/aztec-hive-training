use crate::SingleLayer;
use aztec::{protocol::address::AztecAddress, test::helpers::test_environment::TestEnvironment};
use zkml::architectures::arch_64_10::{PACKED_BIAS_FIELDS, PACKED_WEIGHT_FIELDS, WEIGHTS_PER_FIELD};
use zkml::packing;

// Architecture: single-layer (64 -> 10)
// 640 weights, 10 biases

/// Helper function to create test input pixels (64 pixels)
pub unconstrained fn create_test_input() -> [Field; 64] {
    let mut input: [Field; 64] = [0; 64];
    for i in 0..64 {
        input[i] = (i as Field) * 1_000;
    }
    input
}

/// Create quantized test weights for single layer (640 weights, range -256 to 255)
pub unconstrained fn create_trained_weights_single() -> [Field; 640] {
    let mut weights: [Field; 640] = [0; 640];
    for i in 0..640 {
        let i_u32: u32 = i;
        let cycle = i_u32 % 100;
        weights[i] = (cycle as Field) - 50;
    }
    weights
}

/// Create quantized test biases for single layer (10 biases, range -256 to 255)
pub unconstrained fn create_trained_biases_single() -> [Field; 10] {
    let mut biases: [Field; 10] = [0; 10];
    for i in 0..10 {
        biases[i] = (i as Field) * 20 - 100;
    }
    biases
}

/// Setup function to deploy contract with empty constructor
pub unconstrained fn deploy_contract(
    env: &mut TestEnvironment,
    owner: AztecAddress,
) -> AztecAddress {
    let initializer_call_interface = SingleLayer::interface().constructor();
    let contract_address =
        env.deploy("SingleLayer").with_public_initializer(owner, initializer_call_interface);
    contract_address
}

/// Setup function that deploys contract with zero weights
pub unconstrained fn setup_contract() -> (&mut TestEnvironment, AztecAddress, AztecAddress) {
    let mut env = TestEnvironment::new();
    let owner = env.create_light_account();
    let contract_address = deploy_contract(&mut env, owner);
    (&mut env, contract_address, owner)
}

/// Setup function that deploys contract with trained weights (for train-step validation)
pub unconstrained fn setup_contract_with_trained_weights() -> (&mut TestEnvironment, AztecAddress, AztecAddress) {
    let mut env = TestEnvironment::new();
    let owner = env.create_light_account();

    let weights = create_trained_weights_single();
    let biases = create_trained_biases_single();

    let mut packed_weights: [Field; PACKED_WEIGHT_FIELDS] = [0; PACKED_WEIGHT_FIELDS];
    for packed_idx in 0..PACKED_WEIGHT_FIELDS {
        let mut weight_batch: [Field; WEIGHTS_PER_FIELD] = [0; WEIGHTS_PER_FIELD];
        let start_idx = packed_idx * WEIGHTS_PER_FIELD;

        for i in 0..WEIGHTS_PER_FIELD {
            let global_idx = start_idx + i;
            if global_idx < 640 {
                weight_batch[i] = weights[global_idx];
            }
        }

        packed_weights[packed_idx] = packing::pack(weight_batch);
    }

    let packed_biases_array: [Field; PACKED_BIAS_FIELDS] = [packing::pack(biases)];

    let initializer_call_interface =
        SingleLayer::interface().constructor_pretrained(packed_weights, packed_biases_array);
    let contract_address =
        env.deploy("SingleLayer").with_public_initializer(owner, initializer_call_interface);

    (&mut env, contract_address, owner)
}
