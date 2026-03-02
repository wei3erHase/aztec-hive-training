use aztec::macros::aztec;

pub mod test;

/*
  by             .__________                 ___ ___
  __  _  __ ____ |__\_____  \  ___________  /   |   \_____    ______ ____
  \ \/ \/ // __ \|  | _(__  <_/ __ \_  __ \/    ~    \__  \  /  ___// __ \
   \     /\  ___/|  |/       \  ___/|  | \/\    Y    // __ \_\___ \\  ___/
    \/\_/  \___  >__/______  /\___  >__|    \___|_  /(____  /____  >\___  >
               \/          \/     \/              \/      \/     \/     \/
               */

#[aztec]
pub contract SingleLayer {
    use aztec::{
        macros::{functions::{external, initializer, only_self, view}, storage::storage},
        state_vars::{Map, PublicMutable},
    };

    use zkml::{
        architectures::arch_64_10::{
            BIASES_PER_FIELD, compute_forward_pass, compute_new_weights, INPUT_SIZE, NetworkWeights,
            OUTPUT_SIZE, PACKED_BIAS_FIELDS, PACKED_WEIGHT_FIELDS, TOTAL_BIASES, TOTAL_WEIGHTS,
            TrainingExample, WEIGHTS_PER_FIELD,
        },
        hash::compute_secure_hash,
        packing::{pack, pack_scaled_values, unpack, unpack_all_scaled, WEIGHT_SCALE},
        softmax::{argmax, compute_confidence, softmax_with_exp},
    };

    // 0.005 - gentle update rate, avoids overshooting in fixed-point representation.
    global LEARNING_RATE: Field = 5000;
    // 0.001 in fixed-point (0.001 * 1_000_000).
    global REGULARIZATION: Field = 1000;

    #[storage]
    struct Storage<Context> {
        weights: Map<Field, PublicMutable<Field, Context>, Context>,
        biases: Map<Field, PublicMutable<Field, Context>, Context>,
    }

    /// Initialize all weights and biases to zero (packed format).
    #[external("public")]
    #[initializer]
    fn constructor() {
        for i in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (i + 1) as Field;
            self.storage.weights.at(map_index).write(pack([0; WEIGHTS_PER_FIELD]));
        }
        for i in 0..PACKED_BIAS_FIELDS {
            let bias_map_index = (i + 1) as Field;
            self.storage.biases.at(bias_map_index).write(pack([0; BIASES_PER_FIELD]));
        }
    }

    /// Initialize with pre-trained packed weights and biases.
    #[external("public")]
    #[initializer]
    fn constructor_pretrained(
        packed_weights: [Field; PACKED_WEIGHT_FIELDS],
        packed_biases: [Field; PACKED_BIAS_FIELDS],
    ) {
        for i in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (i + 1) as Field;
            self.storage.weights.at(map_index).write(packed_weights[i]);
        }
        for i in 0..PACKED_BIAS_FIELDS {
            let bias_map_index = (i + 1) as Field;
            self.storage.biases.at(bias_map_index).write(packed_biases[i]);
        }
    }

    /// Submit a training example privately. Unpacks weights, runs forward+backward,
    /// repacks, and enqueues apply_training_update with a hash commitment to prevent
    /// race conditions on the public state.
    #[external("private")]
    fn submit_training_input(
        input_pixels: [Field; INPUT_SIZE],
        label: Field,
        current_packed_weights: [Field; PACKED_WEIGHT_FIELDS],
        current_packed_biases: [Field; PACKED_BIAS_FIELDS],
    ) {
        let current_weights: [Field; TOTAL_WEIGHTS] = unpack_all_scaled::<TOTAL_WEIGHTS, PACKED_WEIGHT_FIELDS, WEIGHTS_PER_FIELD>(
            current_packed_weights,
        );
        let current_biases: [Field; TOTAL_BIASES] = unpack_all_scaled::<TOTAL_BIASES, PACKED_BIAS_FIELDS, BIASES_PER_FIELD>(
            current_packed_biases,
        );

        let training_example = TrainingExample { image: input_pixels, label };
        let old_network = NetworkWeights { weights: current_weights, biases: current_biases };
        let computed_new_weights =
            compute_new_weights(training_example, old_network, LEARNING_RATE, REGULARIZATION);

        let new_packed_weights = pack_scaled_values::<TOTAL_WEIGHTS, PACKED_WEIGHT_FIELDS, WEIGHTS_PER_FIELD>(
            computed_new_weights.weights,
        );
        let new_packed_biases = pack_scaled_values::<TOTAL_BIASES, PACKED_BIAS_FIELDS, BIASES_PER_FIELD>(
            computed_new_weights.biases,
        );

        let current_weights_hash = compute_secure_hash::<PACKED_WEIGHT_FIELDS, PACKED_BIAS_FIELDS>(
            current_packed_weights,
            current_packed_biases,
        );

        self.enqueue_self.apply_training_update(
            current_weights_hash,
            new_packed_weights,
            new_packed_biases,
        );
    }

    /// Apply a verified training update. Reads current weights, checks their hash
    /// against the commitment from the private call, then writes the new packed values.
    #[external("public")]
    #[only_self]
    fn apply_training_update(
        expected_weights_hash: Field,
        new_packed_weights: [Field; PACKED_WEIGHT_FIELDS],
        new_packed_biases: [Field; PACKED_BIAS_FIELDS],
    ) {
        let mut current_packed_weights: [Field; PACKED_WEIGHT_FIELDS] = [0; PACKED_WEIGHT_FIELDS];
        for packed_idx in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            current_packed_weights[packed_idx] = self.storage.weights.at(map_index).read();
        }

        let mut current_packed_biases: [Field; PACKED_BIAS_FIELDS] = [0; PACKED_BIAS_FIELDS];
        for packed_idx in 0..PACKED_BIAS_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            current_packed_biases[packed_idx] = self.storage.biases.at(map_index).read();
        }

        let actual_weights_hash = compute_secure_hash::<PACKED_WEIGHT_FIELDS, PACKED_BIAS_FIELDS>(
            current_packed_weights,
            current_packed_biases,
        );

        assert(
            expected_weights_hash == actual_weights_hash,
            "Weights changed since training was computed",
        );

        for packed_idx in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            self.storage.weights.at(map_index).write(new_packed_weights[packed_idx]);
        }

        for packed_idx in 0..PACKED_BIAS_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            self.storage.biases.at(map_index).write(new_packed_biases[packed_idx]);
        }
    }

    /// Predict the digit class and confidence score for an input image.
    /// Returns (predicted_class, confidence) where confidence is in [0, SCALING_FACTOR].
    #[external("public")]
    #[view]
    fn predict(input_pixels: [Field; INPUT_SIZE]) -> (Field, Field) {
        let mut packed_weights: [Field; PACKED_WEIGHT_FIELDS] = [0; PACKED_WEIGHT_FIELDS];
        for packed_idx in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_weights[packed_idx] = self.storage.weights.at(map_index).read();
        }

        let mut packed_biases: [Field; PACKED_BIAS_FIELDS] = [0; PACKED_BIAS_FIELDS];
        for packed_idx in 0..PACKED_BIAS_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_biases[packed_idx] = self.storage.biases.at(map_index).read();
        }

        let weights: [Field; TOTAL_WEIGHTS] = unpack_all_scaled::<TOTAL_WEIGHTS, PACKED_WEIGHT_FIELDS, WEIGHTS_PER_FIELD>(
            packed_weights,
        );
        let biases: [Field; TOTAL_BIASES] =
            unpack_all_scaled::<TOTAL_BIASES, PACKED_BIAS_FIELDS, BIASES_PER_FIELD>(packed_biases);

        let network = NetworkWeights { weights, biases };
        let outputs = compute_forward_pass(input_pixels, network);

        let (exp_outputs, sum_exp_safe, _probabilities) = softmax_with_exp(outputs);
        let max_index = argmax(outputs);
        let confidence = compute_confidence(exp_outputs, sum_exp_safe, max_index);

        (max_index as Field, confidence)
    }

    /// Return softmax probabilities for all 10 classes (fixed-point, sum ~= SCALING_FACTOR).
    #[external("public")]
    #[view]
    fn predict_all(input_pixels: [Field; INPUT_SIZE]) -> [Field; OUTPUT_SIZE] {
        let mut packed_weights: [Field; PACKED_WEIGHT_FIELDS] = [0; PACKED_WEIGHT_FIELDS];
        for packed_idx in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_weights[packed_idx] = self.storage.weights.at(map_index).read();
        }

        let mut packed_biases: [Field; PACKED_BIAS_FIELDS] = [0; PACKED_BIAS_FIELDS];
        for packed_idx in 0..PACKED_BIAS_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_biases[packed_idx] = self.storage.biases.at(map_index).read();
        }

        let weights: [Field; TOTAL_WEIGHTS] = unpack_all_scaled::<TOTAL_WEIGHTS, PACKED_WEIGHT_FIELDS, WEIGHTS_PER_FIELD>(
            packed_weights,
        );
        let biases: [Field; TOTAL_BIASES] =
            unpack_all_scaled::<TOTAL_BIASES, PACKED_BIAS_FIELDS, BIASES_PER_FIELD>(packed_biases);

        let network = NetworkWeights { weights, biases };
        let outputs = compute_forward_pass(input_pixels, network);

        zkml::softmax::softmax(outputs)
    }

    /// Return a single weight by flat index (fixed-point scaled).
    #[external("public")]
    #[view]
    fn get_weight(index: u32) -> Field {
        let packed_field_idx = index / WEIGHTS_PER_FIELD;
        let weight_in_packed = index % WEIGHTS_PER_FIELD;
        let map_index = (packed_field_idx + 1) as Field;
        let packed = self.storage.weights.at(map_index).read();
        let unpacked_weights: [Field; WEIGHTS_PER_FIELD] = unpack(packed);
        unpacked_weights[weight_in_packed] * WEIGHT_SCALE
    }

    /// Return a single bias by index (fixed-point scaled).
    #[external("public")]
    #[view]
    fn get_bias(index: u32) -> Field {
        let packed_field_idx = index / BIASES_PER_FIELD;
        let bias_in_packed = index % BIASES_PER_FIELD;
        let map_index = (packed_field_idx + 1) as Field;
        let packed = self.storage.biases.at(map_index).read();
        let unpacked_biases: [Field; BIASES_PER_FIELD] = unpack(packed);
        unpacked_biases[bias_in_packed] * WEIGHT_SCALE
    }

    /// Return all packed weight Fields in a single call.
    #[external("public")]
    #[view]
    fn get_all_packed_weights() -> [Field; PACKED_WEIGHT_FIELDS] {
        let mut packed_weights: [Field; PACKED_WEIGHT_FIELDS] = [0; PACKED_WEIGHT_FIELDS];
        for i in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (i + 1) as Field;
            packed_weights[i] = self.storage.weights.at(map_index).read();
        }
        packed_weights
    }

    /// Return all packed bias Fields in a single call.
    #[external("public")]
    #[view]
    fn get_packed_biases() -> [Field; PACKED_BIAS_FIELDS] {
        let mut packed_biases: [Field; PACKED_BIAS_FIELDS] = [0; PACKED_BIAS_FIELDS];
        for i in 0..PACKED_BIAS_FIELDS {
            let map_index = (i + 1) as Field;
            packed_biases[i] = self.storage.biases.at(map_index).read();
        }
        packed_biases
    }

    /// Return all biases unpacked and scaled to fixed-point in a single call.
    #[external("public")]
    #[view]
    fn get_all_biases() -> [Field; TOTAL_BIASES] {
        let mut packed_biases: [Field; PACKED_BIAS_FIELDS] = [0; PACKED_BIAS_FIELDS];
        for packed_idx in 0..PACKED_BIAS_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_biases[packed_idx] = self.storage.biases.at(map_index).read();
        }
        unpack_all_scaled::<TOTAL_BIASES, PACKED_BIAS_FIELDS, BIASES_PER_FIELD>(packed_biases)
    }

    /// Compute gradient-based Shapley attributions for a target class.
    /// Returns (input_contribs[64], base_bias, logit): per-pixel contributions,
    /// the class bias, and the total pre-softmax logit for the requested class.
    #[external("public")]
    #[view]
    fn shapley_for_class(
        input_pixels: [Field; INPUT_SIZE],
        class_index: Field,
    ) -> ([Field; INPUT_SIZE], Field, Field) {
        let class_idx: u32 = class_index as u32;
        assert(class_idx < OUTPUT_SIZE, "class_index must be 0-9");
        assert(class_index == (class_idx as Field), "class_index must be an integer");

        let mut packed_weights: [Field; PACKED_WEIGHT_FIELDS] = [0; PACKED_WEIGHT_FIELDS];
        for packed_idx in 0..PACKED_WEIGHT_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_weights[packed_idx] = self.storage.weights.at(map_index).read();
        }

        let mut packed_biases: [Field; PACKED_BIAS_FIELDS] = [0; PACKED_BIAS_FIELDS];
        for packed_idx in 0..PACKED_BIAS_FIELDS {
            let map_index = (packed_idx + 1) as Field;
            packed_biases[packed_idx] = self.storage.biases.at(map_index).read();
        }

        let weights: [Field; TOTAL_WEIGHTS] = unpack_all_scaled::<TOTAL_WEIGHTS, PACKED_WEIGHT_FIELDS, WEIGHTS_PER_FIELD>(
            packed_weights,
        );
        let biases: [Field; TOTAL_BIASES] =
            unpack_all_scaled::<TOTAL_BIASES, PACKED_BIAS_FIELDS, BIASES_PER_FIELD>(packed_biases);

        let base_bias = biases[class_idx];

        // input_contrib[i] = input[i] * weight[i, class]  (gradient wrt input)
        let mut input_contribs: [Field; INPUT_SIZE] = [0; INPUT_SIZE];
        let mut logit_sum: Field = base_bias;

        for i in 0..INPUT_SIZE {
            let weight_idx = i * OUTPUT_SIZE + class_idx;
            let weight = weights[weight_idx];
            let contrib = zkml::signed_fixed_multiply(input_pixels[i], weight);
            input_contribs[i] = contrib;
            logit_sum = logit_sum + contrib;
        }

        (input_contribs, base_bias, logit_sum)
    }
}
