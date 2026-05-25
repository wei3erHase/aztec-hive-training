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
pub contract MultiLayerPerceptron {
    use aztec::{
        macros::{functions::{external, internal, view}, storage::storage},
        state_vars::{Map, PublicMutable},
    };

    use zkml::{
        architectures::arch_64_16_10 as architecture, nn_trainer_template::NNTrainerTemplate,
    };

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

    #[internal("private")]
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

    /// Compute Shapley attributions for 2-layer network
    #[external("public")]
    #[view]
    fn shapley_for_class(
        input_pixels: [Field; architecture::INPUT_SIZE],
        class_index: Field,
    ) -> pub ([Field; architecture::INPUT_SIZE], [Field; architecture::HIDDEN_SIZE], Field, Field) {
        let class_idx: u32 = class_index as u32;
        assert(class_idx < architecture::OUTPUT_SIZE, "class_index must be 0-9");
        assert(class_index == (class_idx as Field), "class_index must be an integer");

        let (network_weights, network_biases) = self.internal._read_network();

        let mut hidden_pre_relu: [Field; architecture::HIDDEN_SIZE] =
            [0; architecture::HIDDEN_SIZE];
        let mut hidden_activated: [Field; architecture::HIDDEN_SIZE] =
            [0; architecture::HIDDEN_SIZE];
        for j in 0..architecture::HIDDEN_SIZE {
            let mut sum = network_biases[j];
            for i in 0..architecture::INPUT_SIZE {
                let weight_idx = i * architecture::HIDDEN_SIZE + j;
                let weight = network_weights[weight_idx];
                let input_val = input_pixels[i];
                let product = zkml::signed_fixed_multiply(input_val, weight);
                sum = sum + product;
            }
            hidden_pre_relu[j] = sum;
            let is_positive = NNTrainerTemplate::nn_trainer_is_non_negative_field(sum);
            if is_positive {
                hidden_activated[j] = sum;
            } else {
                hidden_activated[j] = 0;
            }
        }

        let base_bias = network_biases[architecture::HIDDEN_SIZE + class_idx];

        let mut neuron_contribs: [Field; architecture::HIDDEN_SIZE] =
            [0; architecture::HIDDEN_SIZE];
        let mut sum_neuron_contrib: Field = 0;
        for i in 0..architecture::HIDDEN_SIZE {
            let weight_idx = architecture::INPUT_SIZE * architecture::HIDDEN_SIZE
                + i * architecture::OUTPUT_SIZE
                + class_idx;
            let weight = network_weights[weight_idx];
            let contrib = zkml::signed_fixed_multiply(hidden_activated[i], weight);
            neuron_contribs[i] = contrib;
            sum_neuron_contrib = sum_neuron_contrib + contrib;
        }

        let logit = base_bias + sum_neuron_contrib;

        let mut input_contribs: [Field; architecture::INPUT_SIZE] = [0; architecture::INPUT_SIZE];
        for i in 0..architecture::INPUT_SIZE {
            let mut gradient_sum: Field = 0;
            for j in 0..architecture::HIDDEN_SIZE {
                let relu_active =
                    NNTrainerTemplate::nn_trainer_is_non_negative_field(hidden_pre_relu[j]);

                if relu_active {
                    let weight_l1_idx = i * architecture::HIDDEN_SIZE + j;
                    let weight_l2_idx = architecture::INPUT_SIZE * architecture::HIDDEN_SIZE
                        + j * architecture::OUTPUT_SIZE
                        + class_idx;
                    let w1 = network_weights[weight_l1_idx];
                    let w2 = network_weights[weight_l2_idx];
                    let grad_contrib = zkml::signed_fixed_multiply(w1, w2);
                    gradient_sum = gradient_sum + grad_contrib;
                }
            }
            input_contribs[i] = zkml::signed_fixed_multiply(input_pixels[i], gradient_sum);
        }

        (input_contribs, neuron_contribs, base_bias, logit)
    }

}
