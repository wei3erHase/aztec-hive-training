// Generic 2D convolution with padding, stride, and multi-channel support.
// All arithmetic uses signed fixed-point via signed_fixed_multiply.

use crate::signed_fixed_multiply;

/// Compute 2D convolution output height
pub fn conv_output_height(input_height: u32, kernel_size: u32, padding: u32, stride: u32) -> u32 {
    (input_height + 2 * padding - kernel_size) / stride + 1
}

/// Compute 2D convolution output width
pub fn conv_output_width(input_width: u32, kernel_size: u32, padding: u32, stride: u32) -> u32 {
    (input_width + 2 * padding - kernel_size) / stride + 1
}

/// Returns the input value at (h, w, c) with implicit zero-padding of P rows/cols.
/// Using a compile-time P constant lets the compiler eliminate the boundary
/// branches entirely when P=0, reducing AVM gas.
fn get_padded_input<let H: u32, let W: u32, let C: u32, let P: u32>(
    input: [Field; H * W * C],
    h: u32,
    w: u32,
    c: u32,
) -> Field {
    if h < P {
        0
    } else if w < P {
        0
    } else {
        let h_in = h - P;
        let w_in = w - P;
        if h_in >= H {
            0
        } else if w_in >= W {
            0
        } else {
            let idx = (h_in * W + w_in) * C + c;
            input[idx]
        }
    }
}

/// 2D convolution with compile-time kernel size K, padding P, and stride S.
/// Encoding these as generics lets the compiler constant-fold padding/stride
/// expressions (P=0 => checks eliminated; S=1 => multiplications removed),
/// significantly reducing AVM gas versus runtime parameters.
pub fn conv2d<let H: u32, let W: u32, let C_IN: u32, let C_OUT: u32, let K: u32, let P: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    input: [Field; H * W * C_IN],
    weights: [Field; K * K * C_IN * C_OUT],
    biases: [Field; C_OUT],
) -> [Field; OUT_H * OUT_W * C_OUT] {
    let mut output: [Field; OUT_H * OUT_W * C_OUT] = [0; OUT_H * OUT_W * C_OUT];

    for out_y in 0..OUT_H {
        for out_x in 0..OUT_W {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for out_c in 0..C_OUT {
                let mut sum = biases[out_c];

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;

                        for in_c in 0..C_IN {
                            let input_val =
                                get_padded_input::<H, W, C_IN, P>(input, in_y, in_x, in_c);

                            let weight_idx = ((ky * K + kx) * C_IN + in_c) * C_OUT + out_c;
                            let weight = weights[weight_idx];

                            let product = signed_fixed_multiply(input_val, weight);
                            sum = sum + product;
                        }
                    }
                }

                let out_idx = (out_y * OUT_W + out_x) * C_OUT + out_c;
                output[out_idx] = sum;
            }
        }
    }

    output
}

/// Backward pass through convolution (same compile-time K, P, S generics as conv2d).
/// Returns (input_grad, weight_grad, bias_grad).
pub fn conv2d_backward<let H: u32, let W: u32, let C_IN: u32, let C_OUT: u32, let K: u32, let P: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    input: [Field; H * W * C_IN],
    output_grad: [Field; OUT_H * OUT_W * C_OUT],
    weights: [Field; K * K * C_IN * C_OUT],
) -> ([Field; H * W * C_IN], // input_grad
[Field; K * K * C_IN * C_OUT], // weight_grad
[Field; C_OUT] // bias_grad
) {
    let mut input_grad: [Field; H * W * C_IN] = [0; H * W * C_IN];
    let mut weight_grad: [Field; K * K * C_IN * C_OUT] = [0; K * K * C_IN * C_OUT];
    let mut bias_grad: [Field; C_OUT] = [0; C_OUT];

    for out_c in 0..C_OUT {
        let mut sum: Field = 0;
        for out_y in 0..OUT_H {
            for out_x in 0..OUT_W {
                let out_idx = (out_y * OUT_W + out_x) * C_OUT + out_c;
                sum = sum + output_grad[out_idx];
            }
        }
        bias_grad[out_c] = sum;
    }

    for out_y in 0..OUT_H {
        for out_x in 0..OUT_W {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for out_c in 0..C_OUT {
                let out_idx = (out_y * OUT_W + out_x) * C_OUT + out_c;
                let grad_out = output_grad[out_idx];

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;

                        for in_c in 0..C_IN {
                            let input_val =
                                get_padded_input::<H, W, C_IN, P>(input, in_y, in_x, in_c);

                            let weight_idx = ((ky * K + kx) * C_IN + in_c) * C_OUT + out_c;
                            let grad = signed_fixed_multiply(input_val, grad_out);
                            weight_grad[weight_idx] = weight_grad[weight_idx] + grad;

                            if in_y >= P {
                                if in_x >= P {
                                    let h_in = in_y - P;
                                    let w_in = in_x - P;

                                    if h_in < H {
                                        if w_in < W {
                                            let weight_val = weights[weight_idx];
                                            let grad_in =
                                                signed_fixed_multiply(weight_val, grad_out);
                                            let input_idx = (h_in * W + w_in) * C_IN + in_c;
                                            input_grad[input_idx] = input_grad[input_idx] + grad_in;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (input_grad, weight_grad, bias_grad)
}

/// Backward pass computing input gradients only.
/// Skips weight/bias grad accumulation - used by Shapley attribution where
/// only the input gradient is needed. When P=0 the boundary guards are
/// always-true and the compiler removes them entirely.
pub fn conv2d_input_grad_only<let H: u32, let W: u32, let C_IN: u32, let C_OUT: u32, let K: u32, let P: u32, let S: u32, let OUT_H: u32, let OUT_W: u32>(
    output_grad: [Field; OUT_H * OUT_W * C_OUT],
    weights: [Field; K * K * C_IN * C_OUT],
) -> [Field; H * W * C_IN] {
    let mut input_grad: [Field; H * W * C_IN] = [0; H * W * C_IN];

    for out_y in 0..OUT_H {
        for out_x in 0..OUT_W {
            let in_y_start = out_y * S;
            let in_x_start = out_x * S;

            for out_c in 0..C_OUT {
                let out_idx = (out_y * OUT_W + out_x) * C_OUT + out_c;
                let grad_out = output_grad[out_idx];

                for ky in 0..K {
                    for kx in 0..K {
                        let in_y = in_y_start + ky;
                        let in_x = in_x_start + kx;

                        for in_c in 0..C_IN {
                            if in_y >= P {
                                if in_x >= P {
                                    let h_in = in_y - P;
                                    let w_in = in_x - P;

                                    if h_in < H {
                                        if w_in < W {
                                            let weight_idx =
                                                ((ky * K + kx) * C_IN + in_c) * C_OUT + out_c;
                                            let weight_val = weights[weight_idx];
                                            let grad_in =
                                                signed_fixed_multiply(weight_val, grad_out);
                                            let input_idx = (h_in * W + w_in) * C_IN + in_c;
                                            input_grad[input_idx] = input_grad[input_idx] + grad_in;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    input_grad
}
