use crate::conv;

// ============================================================================
// Convolution Tests
// ============================================================================

#[test]
unconstrained fn test_conv_output_height_basic() {
    // Test basic output height calculation
    // output_height = (input_height + 2*padding - kernel_size) / stride + 1
    let height = conv::conv_output_height(8, 3, 1, 1);
    assert(height == 8, "Output height should be 8: (8 + 2*1 - 3) / 1 + 1 = 8");
}

#[test]
unconstrained fn test_conv_output_height_with_stride() {
    // Test with stride 2
    let height = conv::conv_output_height(8, 3, 1, 2);
    assert(height == 4, "Output height should be 4: (8 + 2*1 - 3) / 2 + 1 = 4");
}

#[test]
unconstrained fn test_conv_output_height_no_padding() {
    // Test without padding
    let height = conv::conv_output_height(8, 3, 0, 1);
    assert(height == 6, "Output height should be 6: (8 + 2*0 - 3) / 1 + 1 = 6");
}

#[test]
unconstrained fn test_conv_output_width_basic() {
    // Test basic output width calculation
    let width = conv::conv_output_width(8, 3, 1, 1);
    assert(width == 8, "Output width should be 8: (8 + 2*1 - 3) / 1 + 1 = 8");
}

#[test]
unconstrained fn test_conv_output_width_with_stride() {
    // Test with stride 2
    let width = conv::conv_output_width(8, 3, 1, 2);
    assert(width == 4, "Output width should be 4: (8 + 2*1 - 3) / 2 + 1 = 4");
}

#[test]
unconstrained fn test_conv2d_simple() {
    // Simple 2x2 input, 1x1 kernel, 1 channel
    // Input: 2x2x1 = [1, 2, 3, 4]
    // Kernel: 1x1x1x1 = [1]
    // Bias: [0]
    // Expected output: [1, 2, 3, 4] (same as input since kernel is identity)
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let weights: [Field; 1] = [1_000_000]; // Weight = 1.0 in fixed-point
    let biases: [Field; 1] = [0];

    let output = conv::conv2d::<2, 2, 1, 1, 1, 0, 1, 2, 2>(input, weights, biases);

    // Output should match input (since weight = 1.0 and bias = 0)
    assert(output[0] == input[0], "First output should match first input");
    assert(output[1] == input[1], "Second output should match second input");
    assert(output[2] == input[2], "Third output should match third input");
    assert(output[3] == input[3], "Fourth output should match fourth input");
}

#[test]
unconstrained fn test_conv2d_with_bias() {
    // Test convolution with non-zero bias
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let weights: [Field; 1] = [1_000_000]; // Weight = 1.0
    let biases: [Field; 1] = [100_000]; // Bias = 0.1

    let output = conv::conv2d::<2, 2, 1, 1, 1, 0, 1, 2, 2>(input, weights, biases);

    // Output should be input + bias
    assert(output[0] == input[0] + biases[0], "Output should include bias");
}

#[test]
unconstrained fn test_conv2d_with_padding() {
    // Test convolution with padding
    // 2x2 input with 1x1 kernel and padding=1 should produce 4x4 output
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let weights: [Field; 1] = [1_000_000];
    let biases: [Field; 1] = [0];

    let output = conv::conv2d::<2, 2, 1, 1, 1, 1, 1, 4, 4>(input, weights, biases);

    // Center 2x2 should match input, edges should be 0 (from padding)
    // Output layout: 4x4 flattened
    // [0, 0, 0, 0,
    //  0, 1, 2, 0,
    //  0, 3, 4, 0,
    //  0, 0, 0, 0]
    assert(output[5] == input[0], "Center top-left should match input[0]");
    assert(output[6] == input[1], "Center top-right should match input[1]");
    assert(output[9] == input[2], "Center bottom-left should match input[2]");
    assert(output[10] == input[3], "Center bottom-right should match input[3]");
}

#[test]
unconstrained fn test_conv2d_backward_shape() {
    // Test that backward pass produces correct shapes
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let output_grad: [Field; 4] = [100_000, 200_000, 300_000, 400_000];
    let weights: [Field; 1] = [1_000_000];

    let (input_grad, weight_grad, bias_grad) =
        conv::conv2d_backward::<2, 2, 1, 1, 1, 0, 1, 2, 2>(input, output_grad, weights);

    // Verify output shapes
    assert(input_grad.len() == 4, "Input gradient should have 4 elements");
    assert(weight_grad.len() == 1, "Weight gradient should have 1 element");
    assert(bias_grad.len() == 1, "Bias gradient should have 1 element");
}
