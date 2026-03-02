use crate::hash;

// ============================================================================
// Hash Tests
// ============================================================================

#[test]
unconstrained fn test_compute_secure_hash_single_field() {
    // Test hash with single weight field and single bias field
    let weights: [Field; 1] = [100_000];
    let biases: [Field; 1] = [50_000];

    let hash = hash::compute_secure_hash::<1, 1>(weights, biases);

    // Hash should be non-zero
    assert(hash != 0, "Hash should be non-zero");

    // Hash = weights[0] * BASE + biases[0]
    // = 100_000 * 4294967296 + 50_000
    let expected = weights[0] * hash::HASH_BASE + biases[0];
    assert(hash == expected, "Hash should match expected polynomial calculation");
}

#[test]
unconstrained fn test_compute_secure_hash_multiple_fields() {
    // Test hash with multiple fields
    let weights: [Field; 2] = [100_000, 200_000];
    let biases: [Field; 2] = [50_000, 75_000];

    let hash = hash::compute_secure_hash::<2, 2>(weights, biases);

    // Hash should be non-zero
    assert(hash != 0, "Hash should be non-zero");

    // Hash = weights[0] * BASE^3 + weights[1] * BASE^2 + biases[0] * BASE + biases[1]
    let expected = weights[0] * hash::HASH_BASE * hash::HASH_BASE * hash::HASH_BASE
        + weights[1] * hash::HASH_BASE * hash::HASH_BASE
        + biases[0] * hash::HASH_BASE
        + biases[1];
    assert(hash == expected, "Hash should match expected polynomial calculation");
}

#[test]
unconstrained fn test_compute_secure_hash_position_dependent() {
    // Test that swapping values produces different hash (position-dependent)
    let weights1: [Field; 2] = [100_000, 200_000];
    let biases1: [Field; 1] = [50_000];

    let weights2: [Field; 2] = [200_000, 100_000]; // Swapped
    let biases2: [Field; 1] = [50_000];

    let hash1 = hash::compute_secure_hash::<2, 1>(weights1, biases1);
    let hash2 = hash::compute_secure_hash::<2, 1>(weights2, biases2);

    // Hashes should be different because positions matter
    assert(hash1 != hash2, "Swapping values should produce different hash");
}

#[test]
unconstrained fn test_compute_secure_hash_zero_values() {
    // Test hash with zero values
    let weights: [Field; 2] = [0, 0];
    let biases: [Field; 1] = [0];

    let hash = hash::compute_secure_hash::<2, 1>(weights, biases);

    // Hash should be 0 when all values are 0
    assert(hash == 0, "Hash should be zero when all values are zero");
}

#[test]
unconstrained fn test_compute_secure_hash_different_weights_same_sum() {
    // Test that different weight distributions produce different hashes
    // even if the sum is the same (position matters)
    let weights1: [Field; 2] = [100_000, 200_000]; // Sum = 300_000
    let biases1: [Field; 1] = [50_000];

    let weights2: [Field; 2] = [200_000, 100_000]; // Same sum, different positions
    let biases2: [Field; 1] = [50_000];

    let hash1 = hash::compute_secure_hash::<2, 1>(weights1, biases1);
    let hash2 = hash::compute_secure_hash::<2, 1>(weights2, biases2);

    // Hashes should be different
    assert(hash1 != hash2, "Different positions should produce different hash");
}

#[test]
unconstrained fn test_compute_secure_hash_large_values() {
    // Test hash with large values
    let weights: [Field; 1] = [1_000_000_000];
    let biases: [Field; 1] = [500_000_000];

    let hash = hash::compute_secure_hash::<1, 1>(weights, biases);

    // Hash should be non-zero and large
    assert(hash != 0, "Hash should be non-zero");
    // Hash should be larger than input due to BASE multiplication
    assert(weights[0].lt(hash), "Hash should be larger than input due to BASE multiplication");
}
