// Polynomial rolling hash over packed weight and bias Fields.
// Position-dependent construction prevents swapped or shifted fields
// from producing the same hash output.

// Base for the polynomial: 2^32, chosen so each Field occupies a distinct
// degree in the polynomial and the multiplications stay within Field bounds.
pub global HASH_BASE: Field = 4294967296;

/// Compute a commitment hash over packed_weights concatenated with packed_biases.
/// Uses the Horner-form polynomial:
///   hash = field[0]*BASE^(n-1) + field[1]*BASE^(n-2) + ... + field[n-1]
/// Swapping, reordering, or modifying any field produces a different output.
pub fn compute_secure_hash<let WEIGHT_FIELDS: u32, let BIAS_FIELDS: u32>(
    packed_weights: [Field; WEIGHT_FIELDS],
    packed_biases: [Field; BIAS_FIELDS],
) -> Field {
    let mut hash: Field = 0;

    for i in 0..WEIGHT_FIELDS {
        hash = hash * HASH_BASE;
        hash = hash + packed_weights[i];
    }

    for i in 0..BIAS_FIELDS {
        hash = hash * HASH_BASE;
        hash = hash + packed_biases[i];
    }

    hash
}
