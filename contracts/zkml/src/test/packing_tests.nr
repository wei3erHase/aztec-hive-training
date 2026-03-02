use crate::packing;

// ============================================================================
// Packing/Unpacking Tests for 9-bit Precision (+/- 256 range)
// Pack 28 weights per Field, 26 biases per Field
// ============================================================================

#[test]
unconstrained fn test_pack_unpack_weights_round_trip() {
    // Test round-trip: pack then unpack should recover original weights
    // Values within +/- 256 range for 9-bit precision
    let weights: [Field; 28] = [
        100, -50, 200, -100, 50, -20, 150, -80, 30, -15, 0, 255, -256, 127, -128, 64, -64, 32, -32,
        16, -16, 8, -8, 4, -4, 2, -2, 1,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights match
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Weight should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_weights_boundary_values() {
    // Test with boundary values (max and min for 9-bit: +/- 256)
    let weights: [Field; 28] = [
        255, // Max positive weight (256 - 1)
        -256, // Min negative weight
        0, 128, -128, 64, -64, 32, -32, 16, -16, 8, -8, 4, -4, 2, -2, 1, -1, 200, -200, 150, -150,
        100, -100, 50, -50, 25,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights match
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Boundary weight should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_weights_all_zeros() {
    // Test with all zeros
    let weights: [Field; 28] = [0; 28];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights are zero
    for i in 0..28 {
        assert(unpacked[i] == 0, "Zero weight should remain zero");
    }
}

#[test]
unconstrained fn test_pack_unpack_weights_all_positive() {
    // Test with all positive values within +/- 256
    let weights: [Field; 28] = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        210, 220, 230, 240, 250, 255, 127, 63,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights match
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Positive weight should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_weights_all_negative() {
    // Test with all negative values within +/- 256
    let weights: [Field; 28] = [
        -10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -130, -140, -150, -160, -170,
        -180, -190, -200, -210, -220, -230, -240, -250, -255, -127, -63,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights match
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Negative weight should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_biases_round_trip() {
    // Test round-trip: pack then unpack should recover original biases
    // 26 biases for 2-layer network (16 + 10)
    let biases: [Field; 26] = [
        100, -50, 200, -100, 50, -20, 150, -80, 30, -15, 0, 255, -256, 127, -128, 64, -64, 32, -32,
        16, -16, 8, -8, 4, -4, 2,
    ];

    let packed = packing::pack(biases);
    let unpacked: [Field; 26] = packing::unpack(packed);

    // Verify all biases match
    for i in 0..26 {
        assert(biases[i] == unpacked[i], "Bias should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_biases_boundary_values() {
    // Test with boundary values (max and min for 9-bit: +/- 256)
    let biases: [Field; 26] = [
        255, // Max positive bias
        -256, // Min negative bias
        0, 128, -128, 64, -64, 32, -32, 16, -16, 8, -8, 4, -4, 2, -2, 1, -1, 200, -200, 150, -150,
        100, -100, 50,
    ];

    let packed = packing::pack(biases);
    let unpacked: [Field; 26] = packing::unpack(packed);

    // Verify all biases match
    for i in 0..26 {
        assert(biases[i] == unpacked[i], "Boundary bias should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_biases_all_zeros() {
    // Test with all zeros
    let biases: [Field; 26] = [0; 26];

    let packed = packing::pack(biases);
    let unpacked: [Field; 26] = packing::unpack(packed);

    // Verify all biases are zero
    for i in 0..26 {
        assert(unpacked[i] == 0, "Zero bias should remain zero");
    }
}

#[test]
unconstrained fn test_pack_unpack_biases_mixed_signs() {
    // Test with mixed positive and negative values for all 26 biases
    let biases: [Field; 26] = [
        21, -43, 110, 167, -126, 29, -81, 47, -181, -57, 100, -50, 75, -25, 200, -200, 150, -150,
        125, -125, 80, -80, 40, -40, 20, -20,
    ];

    let packed = packing::pack(biases);
    let unpacked: [Field; 26] = packing::unpack(packed);

    // Verify all biases match
    for i in 0..26 {
        assert(biases[i] == unpacked[i], "Mixed-sign bias should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_weights_pattern() {
    // Test with alternating pattern
    let weights: [Field; 28] = [
        53, -14, 69, 162, -25, -25, 168, 81, -50, 57, -94, -143, 118, -104, -217, 123, -158, 32,
        -169, -230, 41, -30, 46, -22, -54, 122, -86, 21,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights match exactly
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Pattern weight should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_biases_layer_pattern() {
    // Test with pattern representing L1 (16 biases) + L2 (10 biases)
    let biases: [Field; 26] = [
        // Layer 1 biases (16)
        -21, -139, 110, 167, 126, 29, -81, 147, -181, -57, 100, -50, 75, -25, 200, -200,
        // Layer 2 biases (10)
        150, -150, 125, -125, 80, -80, 40, -40, 20, -20,
    ];

    let packed = packing::pack(biases);
    let unpacked: [Field; 26] = packing::unpack(packed);

    // Verify all biases match exactly
    for i in 0..26 {
        assert(biases[i] == unpacked[i], "Layer pattern bias should match after round-trip");
    }
}

#[test]
unconstrained fn test_pack_unpack_preserves_order() {
    // Test that packing preserves the order of weights
    let weights: [Field; 28] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify order is preserved
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Weight order should be preserved");
    }
}

#[test]
unconstrained fn test_pack_unpack_max_values() {
    // Test with max values for 9-bit precision (+/- 256)
    let weights: [Field; 28] = [
        255, -256, 255, -256, 255, -256, 255, -256, 255, -256, 255, -256, 255, -256, 255, -256, 255,
        -256, 255, -256, 255, -256, 255, -256, 255, -256, 255, -256,
    ];

    let packed = packing::pack(weights);
    let unpacked: [Field; 28] = packing::unpack(packed);

    // Verify all weights match
    for i in 0..28 {
        assert(weights[i] == unpacked[i], "Max weight should match after round-trip");
    }
}

/// pack_scaled_values clamps out-of-range quantized values to prevent
/// wrap-around corruption (training can push weights past 9-bit range)
#[test]
unconstrained fn test_pack_scaled_values_clamps_overflow() {
    // Scaled value 600000 -> quantized ~307 -> would wrap to -205 without clamp
    // With clamp: quantized 307 -> 255, unpacks to 255 * 1953 = 498015
    let mut scaled: [Field; 28] = [0; 28];
    scaled[0] = 600000; // Would quantize to ~307, exceeds 255
    scaled[1] = 0 - 600000; // Would quantize to ~-307, below -256

    let packed = packing::pack_scaled_values::<28, 1, 28>(scaled);
    let unpacked = packing::unpack_all_scaled::<28, 1, 28>(packed);

    // First value: 307 clamped to 255 -> 255 * 1953 = 498015
    assert(unpacked[0] == 498015, "Positive overflow should clamp to 255*WEIGHT_SCALE");
    // Second value: -307 clamped to -256 -> -256 * 1953 = -499968
    assert(unpacked[1] == (0 - 499968), "Negative overflow should clamp to -256*WEIGHT_SCALE");
}
