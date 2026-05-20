use aztec::macros::{aztec, contract_template};

#[contract_template("nn_trainer_template")]
#[aztec]
pub contract NNTrainerTemplate {
    use aztec::{
        macros::{
            functions::{external, initializer, internal, only_self, template_virtual, view},
            storage::storage,
        },
        state_vars::{Map, PublicMutable},
    };
    #[contract_library_method]
    pub fn nn_trainer_is_negative_field(val: Field) -> bool {
        let neg_val = 0 - val;
        let neg_as_u64 = neg_val as u64;
        neg_as_u64 < 10_000_000_000_000
    }

    #[contract_library_method]
    pub fn nn_trainer_is_non_negative_field(val: Field) -> bool {
        !nn_trainer_is_negative_field(val)
    }

    #[contract_library_method]
    pub fn nn_trainer_signed_field_gt(a: Field, b: Field) -> bool {
        let a_neg = nn_trainer_is_negative_field(a);
        let b_neg = nn_trainer_is_negative_field(b);

        if a_neg & !b_neg {
            false
        } else if !a_neg & b_neg {
            true
        } else if a_neg & b_neg {
            let abs_a = (0 - a) as u64;
            let abs_b = (0 - b) as u64;
            abs_a < abs_b
        } else {
            let a_u64 = a as u64;
            let b_u64 = b as u64;
            a_u64 > b_u64
        }
    }

    #[contract_library_method]
    pub fn nn_trainer_signed_diff_positive(a: Field, b: Field) -> Field {
        let a_neg = nn_trainer_is_negative_field(a);
        let b_neg = nn_trainer_is_negative_field(b);

        if !a_neg & !b_neg {
            let a_u64 = a as u64;
            let b_u64 = b as u64;
            if a_u64 >= b_u64 {
                (a_u64 - b_u64) as Field
            } else {
                0
            }
        } else if !a_neg & b_neg {
            let a_u64 = a as u64;
            let abs_b = (0 - b) as u64;
            (a_u64 + abs_b) as Field
        } else if a_neg & !b_neg {
            0
        } else {
            let abs_a = (0 - a) as u64;
            let abs_b = (0 - b) as u64;
            if abs_b >= abs_a {
                (abs_b - abs_a) as Field
            } else {
                0
            }
        }
    }

    #[contract_library_method]
    fn nn_trainer_pack(values: [Field; MAX_TRAINER_WEIGHTS_PER_FIELD]) -> Field {
        crate::packing::pack::<MAX_TRAINER_WEIGHTS_PER_FIELD>(values)
    }

    #[contract_library_method]
    fn nn_trainer_unpack(packed: Field) -> [Field; MAX_TRAINER_WEIGHTS_PER_FIELD] {
        crate::packing::unpack::<MAX_TRAINER_WEIGHTS_PER_FIELD>(packed)
    }

    #[contract_library_method]
    fn nn_trainer_unpack_all_scaled_weights(
        packed_values: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
    ) -> [Field; MAX_TRAINER_TOTAL_WEIGHTS] {
        crate::packing::unpack_all_scaled::<MAX_TRAINER_TOTAL_WEIGHTS, MAX_TRAINER_PACKED_WEIGHT_FIELDS, MAX_TRAINER_WEIGHTS_PER_FIELD>(
            packed_values,
        )
    }

    #[contract_library_method]
    fn nn_trainer_unpack_all_scaled_biases(
        packed_values: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) -> [Field; MAX_TRAINER_TOTAL_BIASES] {
        crate::packing::unpack_all_scaled::<MAX_TRAINER_TOTAL_BIASES, MAX_TRAINER_PACKED_BIAS_FIELDS, MAX_TRAINER_BIASES_PER_FIELD>(
            packed_values,
        )
    }

    #[contract_library_method]
    fn nn_trainer_pack_scaled_values_weights(
        scaled_values: [Field; MAX_TRAINER_TOTAL_WEIGHTS],
    ) -> [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS] {
        crate::packing::pack_scaled_values::<MAX_TRAINER_TOTAL_WEIGHTS, MAX_TRAINER_PACKED_WEIGHT_FIELDS, MAX_TRAINER_WEIGHTS_PER_FIELD>(
            scaled_values,
        )
    }

    #[contract_library_method]
    fn nn_trainer_pack_scaled_values_biases(
        scaled_values: [Field; MAX_TRAINER_TOTAL_BIASES],
    ) -> [Field; MAX_TRAINER_PACKED_BIAS_FIELDS] {
        crate::packing::pack_scaled_values::<MAX_TRAINER_TOTAL_BIASES, MAX_TRAINER_PACKED_BIAS_FIELDS, MAX_TRAINER_BIASES_PER_FIELD>(
            scaled_values,
        )
    }

    #[contract_library_method]
    pub fn nn_trainer_copy_prefix<let DEST_SIZE: u32, let SRC_SIZE: u32>(
        source: [Field; SRC_SIZE],
    ) -> [Field; DEST_SIZE] {
        let mut output: [Field; DEST_SIZE] = [0; DEST_SIZE];
        let count = if SRC_SIZE < DEST_SIZE {
            SRC_SIZE
        } else {
            DEST_SIZE
        };

        for i in 0..count {
            output[i] = source[i];
        }

        output
    }

    #[contract_library_method]
    fn nn_trainer_compute_secure_hash(
        packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) -> Field {
        crate::hash::compute_secure_hash(packed_weights, packed_biases)
    }

    #[contract_library_method]
    fn nn_trainer_argmax(values: [Field; MAX_TRAINER_OUTPUT_SIZE]) -> u32 {
        crate::softmax::argmax(values)
    }

    #[contract_library_method]
    fn nn_trainer_exp_approx(raw_diff: Field) -> Field {
        crate::softmax::exp_approx(raw_diff)
    }

    #[contract_library_method]
    fn nn_trainer_softmax_with_exp(
        logits: [Field; MAX_TRAINER_OUTPUT_SIZE],
    ) -> ([Field; MAX_TRAINER_OUTPUT_SIZE], Field, [Field; MAX_TRAINER_OUTPUT_SIZE]) {
        crate::softmax::softmax_with_exp(logits)
    }

    #[contract_library_method]
    fn nn_trainer_softmax(
        logits: [Field; MAX_TRAINER_OUTPUT_SIZE],
    ) -> [Field; MAX_TRAINER_OUTPUT_SIZE] {
        let (_exp_outputs, _sum_exp_safe, probabilities) = nn_trainer_softmax_with_exp(logits);
        probabilities
    }

    #[contract_library_method]
    fn nn_trainer_compute_confidence(
        exp_outputs: [Field; MAX_TRAINER_OUTPUT_SIZE],
        sum_exp_safe: Field,
        max_index: u32,
    ) -> Field {
        crate::softmax::compute_confidence(exp_outputs, sum_exp_safe, max_index)
    }

    // Keep one contract size for all packed trainers, while each host supplies
    // its own architecture-specific forward/new-weight implementation.
    pub global MAX_TRAINER_INPUT_SIZE: u32 = 64;
    pub global MAX_TRAINER_OUTPUT_SIZE: u32 = 10;
    pub global MAX_TRAINER_PACKED_WEIGHT_FIELDS: u32 = 43;
    pub global MAX_TRAINER_PACKED_BIAS_FIELDS: u32 = 1;
    pub global MAX_TRAINER_WEIGHTS_PER_FIELD: u32 = 28;
    pub global MAX_TRAINER_BIASES_PER_FIELD: u32 = 28;
    pub global MAX_TRAINER_TOTAL_WEIGHTS: u32 = 1184;
    pub global MAX_TRAINER_TOTAL_BIASES: u32 = 26;
    pub global TRAINING_LEARNING_RATE: Field = 5000;
    pub global TRAINING_REGULARIZATION: Field = 1000;
    #[storage]
    struct Storage<Context> {
        weights: Map<Field, PublicMutable<Field, Context>, Context>,
        biases: Map<Field, PublicMutable<Field, Context>, Context>,
    }

    #[internal("public")]
    fn _set_training_state(
        packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) {
        for i in 0..MAX_TRAINER_PACKED_WEIGHT_FIELDS {
            let map_index = (i + 1) as Field;
            self.storage.weights.at(map_index).write(packed_weights[i]);
        }

        for i in 0..MAX_TRAINER_PACKED_BIAS_FIELDS {
            let map_index = (i + 1) as Field;
            self.storage.biases.at(map_index).write(packed_biases[i]);
        }
    }

    #[internal("public")]
    fn _read_packed_weights() -> [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS] {
        let mut packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS] =
            [0; MAX_TRAINER_PACKED_WEIGHT_FIELDS];

        for i in 0..MAX_TRAINER_PACKED_WEIGHT_FIELDS {
            let map_index = (i + 1) as Field;
            packed_weights[i] = self.storage.weights.at(map_index).read();
        }

        packed_weights
    }

    #[internal("public")]
    fn _read_packed_biases() -> [Field; MAX_TRAINER_PACKED_BIAS_FIELDS] {
        let mut packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS] =
            [0; MAX_TRAINER_PACKED_BIAS_FIELDS];

        for i in 0..MAX_TRAINER_PACKED_BIAS_FIELDS {
            let map_index = (i + 1) as Field;
            packed_biases[i] = self.storage.biases.at(map_index).read();
        }

        packed_biases
    }

    // Shared pack/unpack for the template storage layout. Host contracts keep their
    // own architecture-specific compute logic and call these when needed.
    #[internal("public")]
    fn _read_network() -> ([Field; MAX_TRAINER_TOTAL_WEIGHTS], [Field; MAX_TRAINER_TOTAL_BIASES]) {
        let packed_weights = self.internal._read_packed_weights();
        let packed_biases = self.internal._read_packed_biases();

        let weights = nn_trainer_unpack_all_scaled_weights(packed_weights);
        let biases = nn_trainer_unpack_all_scaled_biases(packed_biases);

        (weights, biases)
    }

    #[internal("public")]
    fn _write_network(
        weights: [Field; MAX_TRAINER_TOTAL_WEIGHTS],
        biases: [Field; MAX_TRAINER_TOTAL_BIASES],
    ) {
        let packed_weights = nn_trainer_pack_scaled_values_weights(weights);
        let packed_biases = nn_trainer_pack_scaled_values_biases(biases);

        self.internal._set_training_state(packed_weights, packed_biases);
    }

    #[internal("public")]
    fn _initialize_empty_state() {
        let mut initial_packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS] =
            [0; MAX_TRAINER_PACKED_WEIGHT_FIELDS];
        let mut initial_packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS] =
            [0; MAX_TRAINER_PACKED_BIAS_FIELDS];

        for i in 0..MAX_TRAINER_PACKED_WEIGHT_FIELDS {
            let zeros: [Field; MAX_TRAINER_WEIGHTS_PER_FIELD] = [0; MAX_TRAINER_WEIGHTS_PER_FIELD];
            initial_packed_weights[i] = nn_trainer_pack(zeros);
        }

        for i in 0..MAX_TRAINER_PACKED_BIAS_FIELDS {
            let zeros: [Field; MAX_TRAINER_BIASES_PER_FIELD] = [0; MAX_TRAINER_BIASES_PER_FIELD];
            initial_packed_biases[i] = nn_trainer_pack(zeros);
        }

        self.internal._set_training_state(initial_packed_weights, initial_packed_biases);
    }

    #[internal("public")]
    fn _compute_forward_pass(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
        network_weights: [Field; MAX_TRAINER_TOTAL_WEIGHTS],
        network_biases: [Field; MAX_TRAINER_TOTAL_BIASES],
    ) -> [Field; MAX_TRAINER_OUTPUT_SIZE] {
        self.internal._compute_forward_pass_impl(input_pixels, network_weights, network_biases)
    }

    #[internal("public")]
    #[template_virtual]
    fn _compute_forward_pass_impl(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
        network_weights: [Field; MAX_TRAINER_TOTAL_WEIGHTS],
        network_biases: [Field; MAX_TRAINER_TOTAL_BIASES],
    ) -> pub [Field; MAX_TRAINER_OUTPUT_SIZE] {
        assert(false, "Host contract must override _compute_forward_pass_impl");
        [0; MAX_TRAINER_OUTPUT_SIZE]
    }

    #[internal("public")]
    fn _compute_new_weights(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
        label: Field,
        old_network_weights: [Field; MAX_TRAINER_TOTAL_WEIGHTS],
        old_network_biases: [Field; MAX_TRAINER_TOTAL_BIASES],
    ) -> ([Field; MAX_TRAINER_TOTAL_WEIGHTS], [Field; MAX_TRAINER_TOTAL_BIASES]) {
        self.internal._compute_new_weights_impl(
            input_pixels,
            label,
            old_network_weights,
            old_network_biases,
        )
    }

    #[internal("public")]
    #[template_virtual]
    fn _compute_new_weights_impl(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
        label: Field,
        old_network_weights: [Field; MAX_TRAINER_TOTAL_WEIGHTS],
        old_network_biases: [Field; MAX_TRAINER_TOTAL_BIASES],
    ) -> pub ([Field; MAX_TRAINER_TOTAL_WEIGHTS], [Field; MAX_TRAINER_TOTAL_BIASES]) {
        assert(false, "Host contract must override _compute_new_weights_impl");
        (old_network_weights, old_network_biases)
    }

    #[internal("public")]
    fn _constructor() {
        self.internal._initialize_empty_state();
    }

    #[internal("public")]
    fn _constructor_pretrained(
        current_packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        current_packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) {
        self.internal._set_training_state(current_packed_weights, current_packed_biases);
    }

    /// Shared submit entrypoint for training input across all trainer hosts.
    #[external("public")]
    fn submit_training_input(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
        label: Field,
        current_packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        current_packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) {
        let current_weights = nn_trainer_unpack_all_scaled_weights(current_packed_weights);
        let current_biases = nn_trainer_unpack_all_scaled_biases(current_packed_biases);

        let (new_weights, new_biases) = self.internal._compute_new_weights(
            input_pixels,
            label,
            current_weights,
            current_biases,
        );

        let new_packed_weights = nn_trainer_pack_scaled_values_weights(new_weights);
        let new_packed_biases = nn_trainer_pack_scaled_values_biases(new_biases);
        let current_weights_hash =
            nn_trainer_compute_secure_hash(current_packed_weights, current_packed_biases);

        self.internal._apply_training_update(
            current_weights_hash,
            new_packed_weights,
            new_packed_biases,
        );
    }

    #[internal("public")]
    fn _apply_training_update(
        expected_weights_hash: Field,
        new_packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        new_packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) {
        let current_packed_weights = self.internal._read_packed_weights();
        let current_packed_biases = self.internal._read_packed_biases();

        let actual_weights_hash =
            nn_trainer_compute_secure_hash(current_packed_weights, current_packed_biases);

        assert(
            expected_weights_hash == actual_weights_hash,
            "Weights changed since training was computed",
        );

        self.internal._set_training_state(new_packed_weights, new_packed_biases);
    }

    #[internal("public")]
    fn _predict(input_pixels: [Field; MAX_TRAINER_INPUT_SIZE]) -> (Field, Field) {
        let (network_weights, network_biases) = self.internal._read_network();
        let outputs =
            self.internal._compute_forward_pass(input_pixels, network_weights, network_biases);

        let (exp_outputs, sum_exp_safe, _) = nn_trainer_softmax_with_exp(outputs);
        let max_index = nn_trainer_argmax(outputs);
        let confidence = nn_trainer_compute_confidence(exp_outputs, sum_exp_safe, max_index);

        (max_index as Field, confidence)
    }

    #[internal("public")]
    fn _predict_all(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
    ) -> [Field; MAX_TRAINER_OUTPUT_SIZE] {
        let (network_weights, network_biases) = self.internal._read_network();
        let outputs =
            self.internal._compute_forward_pass(input_pixels, network_weights, network_biases);
        nn_trainer_softmax(outputs)
    }

    #[internal("public")]
    fn _get_weight(index: u32) -> pub Field {
        let packed_field_idx = index / MAX_TRAINER_WEIGHTS_PER_FIELD;
        let weight_in_packed = index % MAX_TRAINER_WEIGHTS_PER_FIELD;
        assert(index < MAX_TRAINER_TOTAL_WEIGHTS, "index out of bounds");
        let map_index = (packed_field_idx + 1) as Field;
        let packed = self.storage.weights.at(map_index).read();
        let unpacked: [Field; MAX_TRAINER_WEIGHTS_PER_FIELD] = nn_trainer_unpack(packed);
        unpacked[weight_in_packed] * 1953
    }

    #[internal("public")]
    fn _get_bias(index: u32) -> pub Field {
        let packed_field_idx = index / MAX_TRAINER_BIASES_PER_FIELD;
        let bias_in_packed = index % MAX_TRAINER_BIASES_PER_FIELD;
        assert(index < MAX_TRAINER_TOTAL_BIASES, "index out of bounds");
        let map_index = (packed_field_idx + 1) as Field;
        let packed = self.storage.biases.at(map_index).read();
        let unpacked: [Field; MAX_TRAINER_BIASES_PER_FIELD] = nn_trainer_unpack(packed);
        unpacked[bias_in_packed] * 1953
    }

    #[internal("public")]
    fn _get_all_packed_weights() -> [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS] {
        self.internal._read_packed_weights()
    }

    #[internal("public")]
    fn _get_packed_biases() -> [Field; MAX_TRAINER_PACKED_BIAS_FIELDS] {
        self.internal._read_packed_biases()
    }

    #[internal("public")]
    fn _get_all_biases() -> [Field; MAX_TRAINER_TOTAL_BIASES] {
        let (_weights, biases) = self.internal._read_network();
        biases
    }

    /// Initialize all weights and biases to zero (packed format).
    #[external("public")]
    #[initializer]
    fn constructor() {
        self.internal._constructor();
    }

    /// Initialize with pre-trained packed weights and biases.
    #[external("public")]
    #[initializer]
    fn constructor_pretrained(
        packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) {
        self.internal._constructor_pretrained(packed_weights, packed_biases);
    }

    /// Read current packed weights, verify hash, then write the updated packed values.
    #[external("public")]
    #[only_self]
    fn apply_training_update(
        expected_weights_hash: Field,
        new_packed_weights: [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS],
        new_packed_biases: [Field; MAX_TRAINER_PACKED_BIAS_FIELDS],
    ) {
        self.internal._apply_training_update(
            expected_weights_hash,
            new_packed_weights,
            new_packed_biases,
        );
    }

    /// Predict the digit class and confidence score for an input image.
    #[external("public")]
    #[view]
    fn predict(input_pixels: [Field; MAX_TRAINER_INPUT_SIZE]) -> pub (Field, Field) {
        self.internal._predict(input_pixels)
    }

    /// Return softmax probabilities for all output classes.
    #[external("public")]
    #[view]
    fn predict_all(
        input_pixels: [Field; MAX_TRAINER_INPUT_SIZE],
    ) -> pub [Field; MAX_TRAINER_OUTPUT_SIZE] {
        self.internal._predict_all(input_pixels)
    }

    /// Return a single weight by flat index (fixed-point scaled).
    #[external("public")]
    #[view]
    fn get_weight(index: u32) -> pub Field {
        self.internal._get_weight(index)
    }

    /// Return a single bias by index (fixed-point scaled).
    #[external("public")]
    #[view]
    fn get_bias(index: u32) -> pub Field {
        self.internal._get_bias(index)
    }

    /// Return all packed weight Fields in a single call.
    #[external("public")]
    #[view]
    fn get_all_packed_weights() -> pub [Field; MAX_TRAINER_PACKED_WEIGHT_FIELDS] {
        self.internal._get_all_packed_weights()
    }

    /// Return all packed bias Fields in a single call.
    #[external("public")]
    #[view]
    fn get_packed_biases() -> pub [Field; MAX_TRAINER_PACKED_BIAS_FIELDS] {
        self.internal._get_packed_biases()
    }

    /// Return all biases unpacked and fixed-point scaled in a single call.
    #[external("public")]
    #[view]
    fn get_all_biases() -> pub [Field; MAX_TRAINER_TOTAL_BIASES] {
        let (_weights, biases) = self.internal._read_network();
        biases
    }
}
