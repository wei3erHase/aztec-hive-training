// Piecewise exponential approximation for fixed-point softmax.
// Numerically stable for large logit differences produced by trained weights.
// Uses u64 for Taylor-series arithmetic to get truncating integer division;
// results are cast to Field only at the final output step.

global SCALING_FACTOR: Field = 1_000_000;
pub global SF_U64: u64 = 1_000_000;

// Logit sums are typically in [-10M, +10M] fixed-point units; 10^11 safely
// exceeds the largest positive logit we expect, so values below this threshold
// are treated as negative when cast from Field.
global MAX_POSITIVE_LOGIT: u64 = 100_000_000_000;

/// Returns true if val encodes a negative signed integer.
/// Negative values wrap to p - |val| in the field; (0 - val) as u64 will
/// therefore be small (below MAX_POSITIVE_LOGIT) for negative inputs.
fn is_negative_field(val: Field) -> bool {
    let neg_val = 0 - val;
    let neg_as_u64 = neg_val as u64;
    neg_as_u64 < MAX_POSITIVE_LOGIT
}

/// Signed greater-than comparison for Field values: returns true if a > b.
pub fn signed_field_gt(a: Field, b: Field) -> bool {
    let a_neg = is_negative_field(a);
    let b_neg = is_negative_field(b);

    if a_neg & !b_neg {
        false
    } else if !a_neg & b_neg {
        true
    } else if a_neg & b_neg {
        // Both negative: smaller absolute value is greater.
        let abs_a = (0 - a) as u64;
        let abs_b = (0 - b) as u64;
        abs_a < abs_b
    } else {
        let a_u64 = a as u64;
        let b_u64 = b as u64;
        a_u64 > b_u64
    }
}

/// Signed subtraction a - b, returned as a non-negative Field.
/// Precondition: a >= b (signed), so the result is always >= 0.
/// Used to compute max_logit - logit[i] for the softmax shift.
fn signed_diff_positive(a: Field, b: Field) -> Field {
    let a_neg = is_negative_field(a);
    let b_neg = is_negative_field(b);

    if !a_neg & !b_neg {
        let a_u64 = a as u64;
        let b_u64 = b as u64;
        if a_u64 >= b_u64 {
            (a_u64 - b_u64) as Field
        } else {
            0
        }
    } else if !a_neg & b_neg {
        // a positive, b negative: a - b = a + |b|
        let a_u64 = a as u64;
        let abs_b = (0 - b) as u64;
        (a_u64 + abs_b) as Field
    } else if a_neg & !b_neg {
        // Precondition violated (a < b signed); guard against it.
        0
    } else {
        // Both negative: a - b = |b| - |a|
        let abs_a = (0 - a) as u64;
        let abs_b = (0 - b) as u64;
        if abs_b >= abs_a {
            (abs_b - abs_a) as Field
        } else {
            0
        }
    }
}

// Precomputed exp(-n) * SCALING_FACTOR (6-decimal fixed-point).
global EXP_NEG_1: u64 = 368_000; // exp(-1) ~= 0.368
global EXP_NEG_2: u64 = 135_000; // exp(-2) ~= 0.135
global EXP_NEG_3: u64 = 50_000; // exp(-3) ~= 0.050
global EXP_NEG_4: u64 = 18_000; // exp(-4) ~= 0.018
global EXP_NEG_5: u64 = 7_000; // exp(-5) ~= 0.007

/// Piecewise approximation of exp(-raw_diff / SF) * SF.
/// raw_diff = max_logit - logit[i] in fixed-point units (always >= 0).
/// For |y| < 1 a 4th-order Taylor series is used; for integer breakpoints
/// the precomputed EXP_NEG_N constants are multiplied by a local Taylor term.
pub fn exp_approx(raw_diff: Field) -> Field {
    let raw_diff_u64 = raw_diff as u64;
    let sf = SF_U64;

    let result_u64 = if raw_diff_u64 == 0 {
        sf
    } else if raw_diff_u64 < sf {
        // y in (0, 1): Taylor exp(-y) = 1 - y + y^2/2 - y^3/6 + y^4/24
        let x = raw_diff_u64;
        let x2 = x * x;
        let x2_div_sf = x2 / sf;
        let x3_term = (x2_div_sf * x) / (6 * sf);
        let x4_term = (x2_div_sf * x2_div_sf) / (24 * sf);

        let positive_terms = sf + x2 / (2 * sf) + x4_term;
        let negative_terms = x + x3_term;

        if positive_terms > negative_terms {
            let taylor = positive_terms - negative_terms;
            if taylor < 1 {
                1
            } else {
                taylor
            }
        } else {
            1
        }
    } else if raw_diff_u64 == sf {
        EXP_NEG_1
    } else if raw_diff_u64 < 2 * sf {
        // y in (1, 2): exp(-y) = exp(-1) * exp(-(y-1))
        let x = raw_diff_u64 - sf;
        let x2 = x * x;
        let x2_div_sf = x2 / sf;
        let x3_term = (x2_div_sf * x) / (6 * sf);

        let positive_terms = sf + x2 / (2 * sf);
        let negative_terms = x + x3_term;

        let taylor_safe = if positive_terms > negative_terms {
            let taylor = positive_terms - negative_terms;
            if taylor < 1 {
                1
            } else {
                taylor
            }
        } else {
            1
        };

        (EXP_NEG_1 * taylor_safe) / sf
    } else if raw_diff_u64 == 2 * sf {
        EXP_NEG_2
    } else if raw_diff_u64 < 3 * sf {
        // y in (2, 3): exp(-y) = exp(-2) * exp(-(y-2))
        let x = raw_diff_u64 - 2 * sf;
        let x2 = x * x;

        let positive_terms = sf + x2 / (2 * sf);
        let taylor_safe = if positive_terms > x {
            let taylor = positive_terms - x;
            if taylor < 1 {
                1
            } else {
                taylor
            }
        } else {
            1
        };

        (EXP_NEG_2 * taylor_safe) / sf
    } else if raw_diff_u64 == 3 * sf {
        EXP_NEG_3
    } else if raw_diff_u64 < 4 * sf {
        // y in (3, 4): exp(-y) = exp(-3) * exp(-(y-3))
        let x = raw_diff_u64 - 3 * sf;

        let taylor_safe = if x < sf {
            let taylor = sf - x;
            if taylor < 1 {
                1
            } else {
                taylor
            }
        } else {
            1
        };

        (EXP_NEG_3 * taylor_safe) / sf
    } else if raw_diff_u64 == 4 * sf {
        EXP_NEG_4
    } else if raw_diff_u64 < 5 * sf {
        // y in (4, 5): exp(-y) = exp(-4) * exp(-(y-4))
        let x = raw_diff_u64 - 4 * sf;

        let taylor_safe = if x < sf {
            let taylor = sf - x;
            if taylor < 1 {
                1
            } else {
                taylor
            }
        } else {
            1
        };

        (EXP_NEG_4 * taylor_safe) / sf
    } else {
        // y >= 5: contribution is negligible; clamp to exp(-5).
        EXP_NEG_5
    };

    result_u64 as Field
}

/// Find the maximum signed value in a logits array.
pub fn find_max_logit<let N: u32>(logits: [Field; N]) -> Field {
    let mut max_logit = logits[0];
    for i in 1..N {
        if signed_field_gt(logits[i], max_logit) {
            max_logit = logits[i];
        }
    }
    max_logit
}

/// Softmax over N logits. Returns probabilities scaled by SCALING_FACTOR (sum ~= 1,000,000).
pub fn softmax<let N: u32>(logits: [Field; N]) -> [Field; N] {
    let sf = SCALING_FACTOR;
    let max_logit = find_max_logit(logits);

    let mut exp_outputs: [Field; N] = [0; N];
    let mut sum_exp: Field = 0;

    for i in 0..N {
        let raw_diff = signed_diff_positive(max_logit, logits[i]);

        let exp_val = exp_approx(raw_diff);
        let clamped = if exp_val.lt(1) { 1 } else { exp_val };
        exp_outputs[i] = clamped;
        sum_exp = sum_exp + clamped;
    }

    // Guard against degenerate sum to prevent division by a very small value.
    let sum_exp_safe = if sum_exp.lt(sf) { sf } else { sum_exp };

    // Probability computation in u64 to get truncating integer division.
    let sum_exp_u64 = sum_exp_safe as u64;
    let sf_u64 = SF_U64;

    let mut probabilities: [Field; N] = [0; N];
    for i in 0..N {
        let exp_u64 = exp_outputs[i] as u64;
        let prob_u64 = (exp_u64 * sf_u64) / sum_exp_u64;
        let prob_clamped = if prob_u64 > sf_u64 { sf_u64 } else { prob_u64 };
        probabilities[i] = prob_clamped as Field;
    }

    probabilities
}

/// Softmax variant that also returns the raw exp values and their sum, needed for confidence scoring.
pub fn softmax_with_exp<let N: u32>(logits: [Field; N]) -> ([Field; N], Field, [Field; N]) {
    let sf = SCALING_FACTOR;
    let max_logit = find_max_logit(logits);

    let mut exp_outputs: [Field; N] = [0; N];
    let mut sum_exp: Field = 0;

    for i in 0..N {
        let raw_diff = signed_diff_positive(max_logit, logits[i]);

        let exp_val = exp_approx(raw_diff);
        let clamped = if exp_val.lt(1) { 1 } else { exp_val };
        exp_outputs[i] = clamped;
        sum_exp = sum_exp + clamped;
    }

    let sum_exp_safe = if sum_exp.lt(sf) { sf } else { sum_exp };

    let sum_exp_u64 = sum_exp_safe as u64;
    let sf_u64 = SF_U64;

    let mut probabilities: [Field; N] = [0; N];
    for i in 0..N {
        let exp_u64 = exp_outputs[i] as u64;
        let prob_u64 = (exp_u64 * sf_u64) / sum_exp_u64;
        let prob_clamped = if prob_u64 > sf_u64 { sf_u64 } else { prob_u64 };
        probabilities[i] = prob_clamped as Field;
    }

    (exp_outputs, sum_exp_safe, probabilities)
}

/// Index of the maximum signed value in an array (argmax).
pub fn argmax<let N: u32>(values: [Field; N]) -> u32 {
    let mut max_index: u32 = 0;
    let mut max_value: Field = values[0];

    for i in 1..N {
        if signed_field_gt(values[i], max_value) {
            max_value = values[i];
            max_index = i;
        }
    }

    max_index
}

/// Confidence of the predicted class: exp_outputs[max_index] / sum_exp_safe.
/// Returns a value in [0, SCALING_FACTOR] (i.e. [0, 1] in fixed-point).
/// exp_outputs and sum_exp_safe must come from softmax_with_exp.
pub fn compute_confidence<let N: u32>(
    exp_outputs: [Field; N],
    sum_exp_safe: Field,
    max_index: u32,
) -> Field {
    let exp_max_u64 = exp_outputs[max_index] as u64;
    let sum_exp_u64 = sum_exp_safe as u64;

    let confidence_u64 = if exp_max_u64 > sum_exp_u64 {
        SF_U64
    } else {
        (exp_max_u64 * SF_U64) / sum_exp_u64
    };

    let confidence_clamped = if confidence_u64 > SF_U64 {
        SF_U64
    } else {
        confidence_u64
    };

    confidence_clamped as Field
}
