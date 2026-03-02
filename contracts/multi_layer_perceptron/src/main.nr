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
pub contract MultiLayerPerceptron {
    use aztec::{
        macros::{functions::{external, initializer, only_self, view}, storage::storage},
        state_vars::{Map, PublicMutable},
    };

    // Architecture: 2-layer neural network (64 -> 16 -> 10)
    use zkml::{
        architectures::arch_64_16_10::{
            BIASES_PER_FIELD, compute_forward_pass, compute_new_weights, HIDDEN_SIZE, INPUT_SIZE,
            NetworkWeights, OUTPUT_SIZE, PACKED_BIAS_FIELDS, PACKED_WEIGHT_FIELDS, TOTAL_BIASES,
            TOTAL_WEIGHTS, TrainingExample, WEIGHTS_PER_FIELD,
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

    /// Constructor that initializes weights and biases to zero (packed format)
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

    /// Constructor that initializes with pre-trained weights and biases
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
    /// repacks the updated values (1211 -> 45 Fields for MLP), and enqueues
    /// apply_training_update with a hash commitment to prevent race conditions.
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

    /// Predict digit from input image using 2-layer forward pass
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

    /// Predict all probabilities using 2-layer forward pass
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

    /// Get all biases in a single call (fixed-point scaled values)
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

    /// Get all packed weight Fields
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

    /// Get packed biases Fields
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

    /// Compute Shapley attributions for 2-layer network
    #[external("public")]
    #[view]
    fn shapley_for_class(
        input_pixels: [Field; INPUT_SIZE],
        class_index: Field,
    ) -> ([Field; INPUT_SIZE], [Field; HIDDEN_SIZE], Field, Field) {
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

        let mut hidden_pre_relu: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
        let mut hidden_activated: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
        for j in 0..HIDDEN_SIZE {
            let mut sum = biases[j];
            for i in 0..INPUT_SIZE {
                let weight_idx = i * HIDDEN_SIZE + j;
                let weight = weights[weight_idx];
                let input_val = input_pixels[i];
                let product = zkml::signed_fixed_multiply(input_val, weight);
                sum = sum + product;
            }
            hidden_pre_relu[j] = sum;
            let neg_val = 0 - sum;
            let neg_val_u64 = neg_val as u64;
            let is_positive = neg_val_u64 >= 10_000_000_000_000;
            if is_positive {
                hidden_activated[j] = sum;
            } else {
                hidden_activated[j] = 0;
            }
        }

        let base_bias = biases[HIDDEN_SIZE + class_idx];

        let mut neuron_contribs: [Field; HIDDEN_SIZE] = [0; HIDDEN_SIZE];
        let mut sum_neuron_contrib: Field = 0;
        for i in 0..HIDDEN_SIZE {
            let weight_idx = INPUT_SIZE * HIDDEN_SIZE + i * OUTPUT_SIZE + class_idx;
            let weight = weights[weight_idx];
            let contrib = zkml::signed_fixed_multiply(hidden_activated[i], weight);
            neuron_contribs[i] = contrib;
            sum_neuron_contrib = sum_neuron_contrib + contrib;
        }

        let logit = base_bias + sum_neuron_contrib;

        let mut input_contribs: [Field; INPUT_SIZE] = [0; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            let mut gradient_sum: Field = 0;
            for j in 0..HIDDEN_SIZE {
                let neg_val = 0 - hidden_pre_relu[j];
                let neg_val_u64 = neg_val as u64;
                let relu_active = neg_val_u64 >= 10_000_000_000_000;

                if relu_active {
                    let weight_l1_idx = i * HIDDEN_SIZE + j;
                    let weight_l2_idx = INPUT_SIZE * HIDDEN_SIZE + j * OUTPUT_SIZE + class_idx;
                    let w1 = weights[weight_l1_idx];
                    let w2 = weights[weight_l2_idx];
                    let grad_contrib = zkml::signed_fixed_multiply(w1, w2);
                    gradient_sum = gradient_sum + grad_contrib;
                }
            }
            input_contribs[i] = zkml::signed_fixed_multiply(input_pixels[i], gradient_sum);
        }

        (input_contribs, neuron_contribs, base_bias, logit)
    }
}
