use aztec::macros::{aztec, AztecConfig};
pub mod test;

#[aztec(
    AztecConfig::new()
        .compose("nn_trainer_template")
        .override_internal_template("nn_trainer_template", "_compute_forward_pass_impl")
        .override_internal_template("nn_trainer_template", "_compute_new_weights_impl"),
)]
pub contract CNNGAP {
    use aztec::{
        macros::{functions::{external, internal, view}, storage::storage},
        state_vars::{Map, PublicMutable},
    };

    use zkml::{
        architectures::arch_cnn_gap as architecture, nn_trainer_template::NNTrainerTemplate, relu,
        signed_fixed_multiply,
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

    /// Compute gradient-based Shapley attributions for a target class.
    /// Returns (input_contribs[64], channel_contribs[GAP_SIZE], base_bias, logit).
    #[external("public")]
    #[view]
    fn shapley_for_class(
        input_pixels: [Field; architecture::INPUT_SIZE],
        class_index: Field,
    ) -> pub ([Field; architecture::INPUT_SIZE], [Field; architecture::GAP_SIZE], Field, Field) {
        let class_idx: u32 = class_index as u32;
        assert(class_idx < architecture::OUTPUT_SIZE, "class_index must be 0-9");
        assert(class_index == (class_idx as Field), "class_index must be an integer");

        let (network_weights, network_biases) = self.internal._read_network();

        let conv_weights: [Field; architecture::CONV_WEIGHTS] = NNTrainerTemplate::nn_trainer_copy_prefix::<architecture::CONV_WEIGHTS, NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS>(
            network_weights,
        );
        let conv_biases: [Field; architecture::CONV_BIASES] = NNTrainerTemplate::nn_trainer_copy_prefix::<architecture::CONV_BIASES, NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES>(
            network_biases,
        );
        let base_bias = network_biases[architecture::CONV_BIASES + class_idx];

        let input_3d: [Field; architecture::CONV_IN_H * architecture::CONV_IN_W * architecture::CONV_C_IN] =
            input_pixels;

        let conv_out: [Field; architecture::CONV_OUT_SIZE] = zkml::conv::conv2d::<architecture::CONV_IN_H, architecture::CONV_IN_W, architecture::CONV_C_IN, architecture::CONV_C_OUT, architecture::CONV_K, architecture::CONV_P, architecture::CONV_S, architecture::CONV_OUT_H, architecture::CONV_OUT_W>(
            input_3d,
            conv_weights,
            conv_biases,
        );

        let gap_out: [Field; architecture::GAP_SIZE] = zkml::pooling::global_avg_pool_relu::<architecture::CONV_OUT_H, architecture::CONV_OUT_W, architecture::CONV_C_OUT>(
            conv_out,
        );

        let mut channel_contribs: [Field; architecture::GAP_SIZE] = [0; architecture::GAP_SIZE];
        let mut logit_sum: Field = base_bias;
        let mut fc_grad: [Field; architecture::GAP_SIZE] = [0; architecture::GAP_SIZE];
        for c in 0..architecture::GAP_SIZE {
            let weight_idx = architecture::CONV_WEIGHTS + c * architecture::OUTPUT_SIZE + class_idx;
            let w = network_weights[weight_idx];
            let contrib = signed_fixed_multiply(gap_out[c], w);
            channel_contribs[c] = contrib;
            logit_sum = logit_sum + contrib;
            fc_grad[c] = w;
        }

        let relu_grad: [Field; architecture::CONV_OUT_SIZE] = zkml::pooling::gap_backward_relu_masked::<architecture::CONV_OUT_H, architecture::CONV_OUT_W, architecture::CONV_C_OUT>(
            conv_out,
            fc_grad,
        );

        let input_grad_3d = zkml::conv::conv2d_input_grad_only::<architecture::CONV_IN_H, architecture::CONV_IN_W, architecture::CONV_C_IN, architecture::CONV_C_OUT, architecture::CONV_K, architecture::CONV_P, architecture::CONV_S, architecture::CONV_OUT_H, architecture::CONV_OUT_W>(
            relu_grad,
            conv_weights,
        );

        let mut input_contribs: [Field; architecture::INPUT_SIZE] = [0; architecture::INPUT_SIZE];
        for i in 0..architecture::INPUT_SIZE {
            input_contribs[i] = signed_fixed_multiply(input_pixels[i], input_grad_3d[i]);
        }

        (input_contribs, channel_contribs, base_bias, logit_sum)
    }

    /// Return intermediate activations for visualization.
    /// conv_relu: 75 post-ReLU conv values (5x5x3); gap_out: 3 GAP channel averages.
    /// Both are in fixed-point; divide by SCALING_FACTOR (1_000_000) for real values.
    #[external("public")]
    #[view]
    fn get_activations(
        input_pixels: [Field; architecture::INPUT_SIZE],
    ) -> pub ([Field; architecture::CONV_OUT_SIZE], [Field; architecture::GAP_SIZE]) {
        let (network_weights, network_biases) = self.internal._read_network();

        let conv_weights: [Field; architecture::CONV_WEIGHTS] = NNTrainerTemplate::nn_trainer_copy_prefix::<architecture::CONV_WEIGHTS, NNTrainerTemplate::MAX_TRAINER_TOTAL_WEIGHTS>(
            network_weights,
        );
        let conv_biases: [Field; architecture::CONV_BIASES] = NNTrainerTemplate::nn_trainer_copy_prefix::<architecture::CONV_BIASES, NNTrainerTemplate::MAX_TRAINER_TOTAL_BIASES>(
            network_biases,
        );

        let input_3d: [Field; architecture::CONV_IN_H * architecture::CONV_IN_W * architecture::CONV_C_IN] =
            input_pixels;
        let conv_out: [Field; architecture::CONV_OUT_SIZE] = zkml::conv::conv2d::<architecture::CONV_IN_H, architecture::CONV_IN_W, architecture::CONV_C_IN, architecture::CONV_C_OUT, architecture::CONV_K, architecture::CONV_P, architecture::CONV_S, architecture::CONV_OUT_H, architecture::CONV_OUT_W>(
            input_3d,
            conv_weights,
            conv_biases,
        );

        let mut conv_relu: [Field; architecture::CONV_OUT_SIZE] = [0; architecture::CONV_OUT_SIZE];
        for i in 0..architecture::CONV_OUT_SIZE {
            conv_relu[i] = relu(conv_out[i]);
        }

        let gap_out: [Field; architecture::GAP_SIZE] = zkml::pooling::global_avg_pool::<architecture::CONV_OUT_H, architecture::CONV_OUT_W, architecture::CONV_C_OUT>(
            conv_relu,
        );

        (conv_relu, gap_out)
    }

}
