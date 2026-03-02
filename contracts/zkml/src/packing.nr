// Weight packing: encodes multiple quantized values into single Fields for storage.
//
// Quantized range : [-256, +255]  (9-bit signed; stored as offset value + 256)
// Fixed-point range: multiply quantized value by WEIGHT_SCALE after unpacking.
// Capacity        : floor(254 / 9) = 28 values per Field (Field is ~254 bits).

global OFFSET: Field = 256; // bias so all stored values are non-negative (0..511)
global MASK_9_BITS: u64 = 511; // 2^9 - 1
global POW_2_9: Field = 512; // 2^9

// Clamping bounds for quantized values. Training updates can push weights
// outside this range; overflowing 9 bits corrupts the packed Field (e.g. 266 -> -245).
global QUANT_MAX: Field = 255;
global QUANT_MIN: Field = 0 - 256;
global NEG_CHECK_THRESHOLD: u64 = 10_000_000_000_000; // matches lib.nr

/// Clamp v to [QUANT_MIN, QUANT_MAX] before packing to prevent 9-bit wrap-around.
fn clamp_quantized(v: Field) -> Field {
    let neg_diff = 0 - (v - QUANT_MAX);
    if (neg_diff as u64) < NEG_CHECK_THRESHOLD {
        // v - QUANT_MAX is negative => v < 255; check lower bound.
        let neg_sum = 0 - (v + OFFSET);
        if (neg_sum as u64) < NEG_CHECK_THRESHOLD {
            QUANT_MIN
        } else {
            v
        }
    } else {
        QUANT_MAX
    }
}

// Maps quantized range [-256, +255] to fixed-point range ~[-500k, +500k].
// 500_000 / 256 ~= 1953, covering weights in roughly [-0.5, +0.5].
pub global WEIGHT_SCALE: Field = 1953;
global WEIGHT_SCALE_U64: u64 = 1953;
global HALF_WEIGHT_SCALE_U64: u64 = 976; // floor(WEIGHT_SCALE / 2), for round-to-nearest

/// Signed round-to-nearest division by WEIGHT_SCALE.
/// Rounding (vs. truncation) preserves small gradient updates that would
/// otherwise be lost during re-quantization after a training step.
fn signed_round_divide(a: Field) -> Field {
    let neg_a = 0 - a;
    let neg_a_u64 = neg_a as u64;
    if neg_a_u64 < NEG_CHECK_THRESHOLD {
        // a is negative
        let abs_result = (neg_a_u64 + HALF_WEIGHT_SCALE_U64) / WEIGHT_SCALE_U64;
        0 - (abs_result as Field)
    } else {
        // a is positive or zero
        let a_u64 = a as u64;
        let result = (a_u64 + HALF_WEIGHT_SCALE_U64) / WEIGHT_SCALE_U64;
        result as Field
    }
}

// floor(254 / 9) = 28: maximum 9-bit values that fit in a single ~254-bit Field.
pub global MAX_VALUES_PER_FIELD: u32 = 28;

/// Pack N quantized values (each in [-256, +255]) into a single Field.
/// Values are stored offset by +256 so every 9-bit slot is non-negative.
/// N must be <= MAX_VALUES_PER_FIELD.
pub fn pack<let N: u32>(values: [Field; N]) -> Field {
    let mut packed: Field = 0;
    let mut shift: Field = 1;

    for i in 0..N {
        let v_offset_u64 = (values[i] + OFFSET) as u64;
        let masked = (v_offset_u64 & MASK_9_BITS) as Field;
        packed = packed + (masked * shift);
        shift = shift * POW_2_9;
    }

    packed
}

/// Unpack a Field into N quantized values in [-256, +255].
/// Multiply each result by WEIGHT_SCALE to obtain fixed-point values.
/// N must match what was used during pack.
pub fn unpack<let N: u32>(packed: Field) -> [Field; N] {
    let mut values: [Field; N] = [0; N];
    let mut remaining = packed;

    for i in 0..N {
        let offset_u64 = (remaining as u64) & MASK_9_BITS;
        let offset_field = offset_u64 as Field;
        values[i] = offset_field - OFFSET;
        remaining = (remaining - offset_field) / POW_2_9;
    }

    values
}

/// Unpack PACKED_FIELDS packed Fields into TOTAL_VALUES fixed-point values.
/// Each slot is multiplied by WEIGHT_SCALE after extraction.
pub fn unpack_all_scaled<let TOTAL_VALUES: u32, let PACKED_FIELDS: u32, let VALUES_PER_FIELD: u32>(
    packed_values: [Field; PACKED_FIELDS],
) -> [Field; TOTAL_VALUES] {
    let mut values: [Field; TOTAL_VALUES] = [0; TOTAL_VALUES];

    for packed_idx in 0..PACKED_FIELDS {
        let unpacked: [Field; VALUES_PER_FIELD] =
            unpack::<VALUES_PER_FIELD>(packed_values[packed_idx]);
        let start_idx = packed_idx * VALUES_PER_FIELD;

        for i in 0..VALUES_PER_FIELD {
            let global_idx = start_idx + i;
            if global_idx < TOTAL_VALUES {
                values[global_idx] = unpacked[i] * WEIGHT_SCALE;
            }
        }
    }

    values
}

/// Inverse of unpack_all_scaled: quantize TOTAL_VALUES fixed-point values and
/// pack them into PACKED_FIELDS Fields ready for on-chain storage.
/// Each value is round-divided by WEIGHT_SCALE and clamped before packing.
pub fn pack_scaled_values<let TOTAL_VALUES: u32, let PACKED_FIELDS: u32, let VALUES_PER_FIELD: u32>(
    scaled_values: [Field; TOTAL_VALUES],
) -> [Field; PACKED_FIELDS] {
    let mut packed_fields: [Field; PACKED_FIELDS] = [0; PACKED_FIELDS];

    for packed_idx in 0..PACKED_FIELDS {
        let mut batch: [Field; VALUES_PER_FIELD] = [0; VALUES_PER_FIELD];
        let start_idx = packed_idx * VALUES_PER_FIELD;

        for i in 0..VALUES_PER_FIELD {
            let global_idx = start_idx + i;
            if global_idx < TOTAL_VALUES {
                let quantized = signed_round_divide(scaled_values[global_idx]);
                batch[i] = clamp_quantized(quantized);
            }
        }

        packed_fields[packed_idx] = pack(batch);
    }

    packed_fields
}
