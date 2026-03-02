// Pooling operations: Global Average Pooling (GAP), Max Pooling, Average Pooling.
// All arithmetic uses the same signed fixed-point conventions as the rest of zkml.

use crate::relu;
use crate::softmax;

global SCALING_FACTOR_U64: u64 = 1_000_000;
global NEG_CHECK_THRESHOLD: u64 = 10_000_000_000_000;

/// Signed integer division a / divisor for Field values.
/// Uses the same NEG_CHECK_THRESHOLD sign-detection pattern as lib.nr.
fn signed_int_divide(a: Field, divisor: u64) -> Field {
    let neg_a = 0 - a;
    let neg_a_u64 = neg_a as u64;
    if neg_a_u64 < NEG_CHECK_THRESHOLD {
        let abs_result = neg_a_u64 / divisor;
        0 - (abs_result as Field)
    } else {
        let a_u64 = a as u64;
        (a_u64 / divisor) as Field
    }
}

/// Global Average Pooling: reduces an H*W*C tensor to C values by
/// averaging each channel over all spatial positions.
/// Input is in fixed-point; spatial_size is a plain integer divisor.
pub fn global_avg_pool<let H: u32, let W: u32, let C: u32>(
    input: [Field; H * W * C],
) -> [Field; C] {
    let spatial_size_u64 = (H * W) as u64;
    let mut output: [Field; C] = [0; C];

    for c in 0..C {
        let mut sum: Field = 0;
        for h in 0..H {
            for w in 0..W {
                sum = sum + input[(h * W + w) * C + c];
            }
        }
        output[c] = (sum as u64 / spatial_size_u64) as Field;
    }

    output
}

/// GAP with ReLU fused before accumulation.
/// Avoids materializing a separate post-ReLU array, saving an extra loop and allocation.
pub fn global_avg_pool_relu<let H: u32, let W: u32, let C: u32>(
    input: [Field; H * W * C],
) -> [Field; C] {
    let spatial_size_u64 = (H * W) as u64;
    let mut output: [Field; C] = [0; C];

    for c in 0..C {
        let mut sum: Field = 0;
        for h in 0..H {
            for w in 0..W {
                sum = sum + relu(input[(h * W + w) * C + c]);
            }
        }
        output[c] = (sum as u64 / spatial_size_u64) as Field;
    }

    output
}

/// Backward pass through GAP: broadcasts each channel's output gradient
/// uniformly to all H*W spatial positions (divided by spatial size).
pub fn gap_backward<let H: u32, let W: u32, let C: u32>(
    output_grad: [Field; C],
) -> [Field; H * W * C] {
    let spatial_size_u64 = (H * W) as u64;

    let mut input_grad: [Field; H * W * C] = [0; H * W * C];

    for c in 0..C {
        let grad_per_position = signed_int_divide(output_grad[c], spatial_size_u64);

        for h in 0..H {
            for w in 0..W {
                let idx = (h * W + w) * C + c;
                input_grad[idx] = grad_per_position;
            }
        }
    }

    input_grad
}

/// GAP backward with ReLU mask fused.
/// conv_out holds the pre-ReLU activations; positions where conv_out <= 0
/// receive zero gradient (ReLU was inactive there).
pub fn gap_backward_relu_masked<let H: u32, let W: u32, let C: u32>(
    conv_out: [Field; H * W * C],
    output_grad: [Field; C],
) -> [Field; H * W * C] {
    let spatial_size_u64 = (H * W) as u64;

    let mut relu_grad: [Field; H * W * C] = [0; H * W * C];

    for c in 0..C {
        let grad_per_position = signed_int_divide(output_grad[c], spatial_size_u64);

        for h in 0..H {
            for w in 0..W {
                let idx = (h * W + w) * C + c;
                let neg_val = 0 - conv_out[idx];
                let neg_val_u64 = neg_val as u64;
                let relu_active = neg_val_u64 >= NEG_CHECK_THRESHOLD;
                if relu_active {
                    relu_grad[idx] = grad_per_position;
                }
            }
        }
    }

    relu_grad
}

/// Max pooling with K*K kernel and stride S.
/// OUT_H and OUT_W must be supplied as compile-time generics.
pub fn max_pool2d<let H: u32, let W: u32, let C: u32, let K: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    input: [Field; H * W * C],
) -> [Field; OUT_H * OUT_W * C] {
    let mut output: [Field; OUT_H * OUT_W * C] = [0; OUT_H * OUT_W * C];

    for out_y in 0..OUT_H {
        for out_x in 0..OUT_W {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for c in 0..C {
                let mut max_val: Field = get_input_value(input, in_y_start, in_x_start, c, H, W, C);

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;
                        let val = get_input_value(input, in_y, in_x, c, H, W, C);

                        if softmax::signed_field_gt(val, max_val) {
                            max_val = val;
                        }
                    }
                }

                let out_idx = (out_y * OUT_W + out_x) * C + c;
                output[out_idx] = max_val;
            }
        }
    }

    output
}

/// Average pooling with K*K kernel and stride S.
/// OUT_H and OUT_W must be supplied as compile-time generics.
pub fn avg_pool2d<let H: u32, let W: u32, let C: u32, let K: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    input: [Field; H * W * C],
) -> [Field; OUT_H * OUT_W * C] {
    let out_h = OUT_H;
    let out_w = OUT_W;
    let kernel_area = K * K;
    let kernel_area_u64 = kernel_area as u64;

    let mut output: [Field; OUT_H * OUT_W * C] = [0; OUT_H * OUT_W * C];

    for out_y in 0..out_h {
        for out_x in 0..out_w {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for c in 0..C {
                let mut sum: Field = 0;

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;
                        let val = get_input_value(input, in_y, in_x, c, H, W, C);
                        sum = sum + val;
                    }
                }

                let sum_u64 = sum as u64;
                let avg_u64 = sum_u64 / kernel_area_u64;
                let out_idx = (out_y * OUT_W + out_x) * C + c;
                output[out_idx] = avg_u64 as Field;
            }
        }
    }

    output
}

/// Bounds-checked input accessor: returns 0 for out-of-range (h, w, c) coordinates.
fn get_input_value<let H: u32, let W: u32, let C: u32>(
    input: [Field; H * W * C],
    h: u32,
    w: u32,
    c: u32,
    height: u32,
    width: u32,
    channels: u32,
) -> Field {
    if h < height {
        if w < width {
            if c < channels {
                input[(h * width + w) * channels + c]
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    }
}

/// Max pooling backward: routes each output gradient to the input position
/// that held the maximum value in that window (all others receive zero).
pub fn max_pool2d_backward<let H: u32, let W: u32, let C: u32, let K: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    input: [Field; H * W * C],
    output_grad: [Field; OUT_H * OUT_W * C],
) -> [Field; H * W * C] {
    let mut input_grad: [Field; H * W * C] = [0; H * W * C];

    for out_y in 0..OUT_H {
        for out_x in 0..OUT_W {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for c in 0..C {
                let out_idx = (out_y * OUT_W + out_x) * C + c;
                let grad_out = output_grad[out_idx];

                let mut max_val: Field = get_input_value(input, in_y_start, in_x_start, c, H, W, C);
                let mut max_h = in_y_start;
                let mut max_w = in_x_start;

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;
                        let val = get_input_value(input, in_y, in_x, c, H, W, C);

                        if softmax::signed_field_gt(val, max_val) {
                            max_val = val;
                            max_h = in_y;
                            max_w = in_x;
                        }
                    }
                }

                // Route gradient exclusively to the max position.
                if max_h < H {
                    if max_w < W {
                        let input_idx = (max_h * W + max_w) * C + c;
                        input_grad[input_idx] = input_grad[input_idx] + grad_out;
                    }
                }
            }
        }
    }

    input_grad
}

/// Average pooling backward: distributes each output gradient evenly across
/// all K*K positions in its corresponding input window.
pub fn avg_pool2d_backward<let H: u32, let W: u32, let C: u32, let K: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    output_grad: [Field; OUT_H * OUT_W * C],
) -> [Field; H * W * C] {
    let out_h = OUT_H;
    let out_w = OUT_W;
    let kernel_area = K * K;
    let kernel_area_u64 = kernel_area as u64;

    let mut input_grad: [Field; H * W * C] = [0; H * W * C];

    for out_y in 0..out_h {
        for out_x in 0..out_w {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for c in 0..C {
                let out_idx = (out_y * OUT_W + out_x) * C + c;
                let grad_out = output_grad[out_idx];

                let grad_u64 = grad_out as u64;
                let grad_per_position_u64 = grad_u64 / kernel_area_u64;
                let grad_per_position = grad_per_position_u64 as Field;

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;

                        if in_y < H {
                            if in_x < W {
                                let input_idx = (in_y * W + in_x) * C + c;
                                input_grad[input_idx] = input_grad[input_idx] + grad_per_position;
                            }
                        }
                    }
                }
            }
        }
    }

    input_grad
}
