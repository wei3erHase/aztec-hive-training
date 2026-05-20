use aztec::macros::{aztec, AztecConfig};
pub mod test;

/*
  by             .__________                 ___ ___
  __  _  __ ____ |__\_____  \  ___________  /   |   \_____    ______ ____
  \ \/ \/ // __ \|  | _(__  <_/ __ \_  __ \/    ~    \__  \  /  ___// __ \
   \     /\  ___/|  |/       \  ___/|  | \/\    Y    // __ \_\___ \\  ___/
    \/\_/  \___  >__/______  /\___  >__|    \___|_  /(____  /____  >\___  >
               \/          \/     \/              \/      \/     \/     \/
               */

#[aztec(
    AztecConfig::new()
        .compose("nn_trainer_template")
        .override_internal_template("nn_trainer_template", "_compute_forward_pass_impl")
        .override_internal_template("nn_trainer_template", "_compute_new_weights_impl"),
)]
pub contract SingleLayer {
    use aztec::{
        macros::{functions::{external, internal, view}, storage::storage},
        state_vars::{Map, PublicMutable},
    };

    use zkml::{architectures::arch_64_10 as architecture, nn_trainer_template::NNTrainerTemplate};

    pub use zkml::{hash, MAX_POSITIVE_PRODUCT, packing, softmax};

    pub global MAX_TRAINER_INPUT_SIZE: u32 = NNTrainerTemplate::MAX_TRAINER_INPUT_SIZE;
    pub global MAX_TRAINER_OUTPUT_SIZE: u32 = NNTrainerTemplate::MAX_TRAINER_OUTPUT_SIZE;
    pub global MAX_TRAINER_PACKED_WEIGHT_FIELDS: u32 =
        NNTrainerTemplate::MAX_TRAINER_PACKED_WEIGHT_FIELDS;
    pub global MAX_TRAINER_PACKED_BIAS_FIELDS: u32 =
        NNTrainerTemplate::MAX_TRAINER_PACKED_BIAS_FIELDS;
    pub global MAX_TRAINER_WEIGHTS_PER_FIELD: u32 =
        NNTrainerTemplate::MAX_TRAINER_WEIGHTS_PER_FIELD;
    pub global MAX_TRAINER_BIASES_PER_FIELD: u32 = NNTrainerTemplate::MAX_TRAINER_BIASES_PER_FIELD;
    pub global MAX_TRAINER_TOTAL_WEIGHTS: u32 = NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS;
    pub global MAX_TRAINER_TOTAL_BIASES: u32 = NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES;

    pub global TRAINING_LEARNING_RATE: Field = architecture::TRAINING_LEARNING_RATE;
    pub global TRAINING_REGULARIZATION: Field = architecture::TRAINING_REGULARIZATION;

    #[storage]
    struct Storage<Context> {
        weights: Map<Field, PublicMutable<Field, Context>, Context>,
        biases: Map<Field, PublicMutable<Field, Context>, Context>,
    }

    #[internal("public")]
    fn _compute_forward_pass_impl(
        input_pixels: [Field; NNTrainerTemplate::MAX_TRAINER_INPUT_SIZE],
        network_weights: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS],
        network_biases: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES],
    ) -> pub [Field; NNTrainerTemplate::MAX_TRAINER_OUTPUT_SIZE] {
        architecture::compute_forward_pass_trainer(input_pixels, network_weights, network_biases)
    }

    #[internal("public")]
    fn _compute_new_weights_impl(
        input_pixels: [Field; NNTrainerTemplate::MAX_TRAINER_INPUT_SIZE],
        label: Field,
        old_network_weights: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS],
        old_network_biases: [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES],
    ) -> pub ([Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS], [Field; NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES]) {
        architecture::compute_new_weights_trainer(
            input_pixels,
            label,
            old_network_weights,
            old_network_biases,
            TRAINING_LEARNING_RATE,
            TRAINING_REGULARIZATION,
        )
    }

    /// Compute gradient-based Shapley attributions for a target class.
    /// Returns (input_contribs[64], base_bias, logit): per-pixel contributions,
    /// the class bias, and the total pre-softmax logit for the requested class.
    #[external("public")]
    #[view]
    fn shapley_for_class(
        input_pixels: [Field; architecture::INPUT_SIZE],
        class_index: Field,
    ) -> pub ([Field; architecture::INPUT_SIZE], Field, Field) {
        let class_idx: u32 = class_index as u32;
        assert(class_idx < architecture::OUTPUT_SIZE, "class_index must be 0-9");
        assert(class_index == (class_idx as Field), "class_index must be an integer");

        let (network_weights, network_biases) = self.internal._read_network();

        let base_bias = network_biases[class_idx];

        // input_contrib[i] = input[i] * weight[i, class]  (gradient wrt input)
        let mut input_contribs: [Field; architecture::INPUT_SIZE] = [0; architecture::INPUT_SIZE];
        let mut logit_sum: Field = base_bias;

        for i in 0..architecture::INPUT_SIZE {
            let weight_idx = i * architecture::OUTPUT_SIZE + class_idx;
            let weight = network_weights[weight_idx];
            let contrib = zkml::signed_fixed_multiply(input_pixels[i], weight);
            input_contribs[i] = contrib;
            logit_sum = logit_sum + contrib;
        }

        (input_contribs, base_bias, logit_sum)
    }

}
