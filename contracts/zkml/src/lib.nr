// ZKML Neural Network Library
// Library-agnostic primitives: fixed-point arithmetic, convolution, pooling, softmax, packing.
// Architecture-specific forward/backward passes live in contracts/zkml/src/architectures/.

/*
  by             .__________                 ___ ___
  __  _  __ ____ |__\_____  \  ___________  /   |   \_____    ______ ____
  \ \/ \/ // __ \|  | _(__  <_/ __ \_  __ \/    ~    \__  \  /  ___// __ \
   \     /\  ___/|  |/       \  ___/|  | \/\    Y    // __ \_\___ \\  ___/
    \/\_/  \___  >__/______  /\___  >__|    \___|_  /(____  /____  >\___  >
               \/          \/     \/              \/      \/     \/     \/
               */

pub mod softmax;
pub mod packing;
pub mod conv;
pub mod pooling;
pub mod hash;
pub mod architectures;
pub mod test;

// Scaling factor: 10^6 gives 6 decimal places of fixed-point precision.
pub global SCALING_FACTOR: Field = 1000000;
global SCALING_FACTOR_U64: u64 = 1000000;

// Sign-detection threshold: a Field value is treated as negative when
// (0 - value) as u64 falls below this bound.
// Chosen to exceed the largest positive product we ever compute:
// max_pixel (~1M) * max_weight (~256M fixed-point) ~= 2.56e14 < 10^13 after scaling.
global MAX_POSITIVE_PRODUCT: u64 = 10_000_000_000_000;

/// Signed fixed-point multiplication: (a * b) / SCALING_FACTOR.
/// Field arithmetic wraps correctly for signed operands; sign is detected
/// via the MAX_POSITIVE_PRODUCT threshold rather than Field.lt() to avoid
/// expensive range checks.
pub fn signed_fixed_multiply(a: Field, b: Field) -> Field {
    let product = a * b;
    let neg_product = 0 - product;
    let neg_as_u64 = neg_product as u64;

    if neg_as_u64 < MAX_POSITIVE_PRODUCT {
        let abs_result = neg_as_u64 / SCALING_FACTOR_U64;
        0 - (abs_result as Field)
    } else {
        let prod_u64 = product as u64;
        (prod_u64 / SCALING_FACTOR_U64) as Field
    }
}

/// Signed integer division: a / divisor.
/// Used to convert fixed-point scaled weights back to quantized form.
/// Sign is detected via the MAX_POSITIVE_PRODUCT threshold.
pub fn signed_fixed_divide(a: Field, divisor: Field) -> Field {
    let divisor_u64 = divisor as u64;
    let neg_a = 0 - a;
    let neg_a_u64 = neg_a as u64;

    if neg_a_u64 < MAX_POSITIVE_PRODUCT {
        let abs_result = neg_a_u64 / divisor_u64;
        0 - (abs_result as Field)
    } else {
        let a_u64 = a as u64;
        (a_u64 / divisor_u64) as Field
    }
}

/// ReLU activation: max(0, x).
/// Negativity is detected via the MAX_POSITIVE_PRODUCT threshold.
pub fn relu(x: Field) -> Field {
    let neg_x = 0 - x;
    let neg_x_u64 = neg_x as u64;
    if neg_x_u64 < MAX_POSITIVE_PRODUCT {
        0
    } else {
        x
    }
}
