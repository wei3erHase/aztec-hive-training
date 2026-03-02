use crate::pooling;

// ============================================================================
// Pooling Tests
// ============================================================================

#[test]
unconstrained fn test_global_avg_pool_simple() {
    // Test Global Average Pooling on 2x2x1 input
    // Inputs are in fixed-point: [1.0, 2.0, 3.0, 4.0] = [1M, 2M, 3M, 4M]
    // GAP = sum / spatial_size = 10M / 4 = 2.5M  (represents float 2.5)
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let output = pooling::global_avg_pool::<2, 2, 1>(input);

    // Expected: (1M + 2M + 3M + 4M) / 4 = 2.5M
    let expected: Field = 2_500_000;
    let diff = if expected.lt(output[0]) {
        output[0] - expected
    } else {
        expected - output[0]
    };
    assert(diff.lt(1001), "GAP output should be approximately 2.5M");
}

#[test]
unconstrained fn test_global_avg_pool_multi_channel() {
    // Test GAP with 2 channels; inputs in fixed-point (multiply by 1M).
    // Input layout: flattened H*W*C where idx = (h * W + w) * C + c
    // Channel 0 values: [1M, 3M, 5M, 7M] -> avg = 16M / 4 = 4M
    // Channel 1 values: [2M, 4M, 6M, 8M] -> avg = 20M / 4 = 5M
    let input: [Field; 8] = [
        1_000_000, 2_000_000, // h=0,w=0: ch0=1M, ch1=2M
        3_000_000, 4_000_000, // h=0,w=1: ch0=3M, ch1=4M
        5_000_000, 6_000_000, // h=1,w=0: ch0=5M, ch1=6M
        7_000_000, 8_000_000, // h=1,w=1: ch0=7M, ch1=8M
    ];
    let output = pooling::global_avg_pool::<2, 2, 2>(input);

    // Channel 0: (1M + 3M + 5M + 7M) / 4 = 4M
    let expected_0: Field = 4_000_000;
    let diff_0 = if expected_0.lt(output[0]) {
        output[0] - expected_0
    } else {
        expected_0 - output[0]
    };
    assert(diff_0.lt(1001), "Channel 0 GAP should be approximately 4M");

    // Channel 1: (2M + 4M + 6M + 8M) / 4 = 5M
    let expected_1: Field = 5_000_000;
    let diff_1 = if expected_1.lt(output[1]) {
        output[1] - expected_1
    } else {
        expected_1 - output[1]
    };
    assert(diff_1.lt(1001), "Channel 1 GAP should be approximately 5M");
}

#[test]
unconstrained fn test_gap_backward() {
    // Test backward pass through GAP
    // Output gradient: [100_000] (single channel)
    // Input gradient: [100_000/4, 100_000/4, 100_000/4, 100_000/4] (broadcast)
    let output_grad: [Field; 1] = [100_000];
    let input_grad = pooling::gap_backward::<2, 2, 1>(output_grad);

    // Each position should get output_grad / spatial_size = 100_000 / 4 = 25_000
    let expected: Field = 25_000;
    for i in 0..4 {
        let diff = if expected.lt(input_grad[i]) {
            input_grad[i] - expected
        } else {
            expected - input_grad[i]
        };
        assert(diff.lt(1001), "Input gradient should be broadcast correctly");
    }
}

#[test]
unconstrained fn test_gap_backward_negative_gradient() {
    // Negative gradients must propagate correctly through GAP backward.
    // A negative Field value like (0 - 200_000) should divide by spatial_size
    // and remain negative: (0 - 200_000) / 4 = (0 - 50_000).
    let neg_grad: Field = 0 - 200_000;
    let output_grad: [Field; 1] = [neg_grad];
    let input_grad = pooling::gap_backward::<2, 2, 1>(output_grad);

    let expected: Field = 0 - 50_000;
    for i in 0..4 {
        assert(
            input_grad[i] == expected,
            "Negative gradient should divide correctly through GAP backward",
        );
    }
}

#[test]
unconstrained fn test_gap_backward_mixed_sign_multichannel() {
    // 2x2 spatial, 2 channels: one positive gradient, one negative.
    let output_grad: [Field; 2] = [100_000, 0 - 80_000];
    let input_grad = pooling::gap_backward::<2, 2, 2>(output_grad);

    let expected_ch0: Field = 25_000; // 100_000 / 4
    let expected_ch1: Field = 0 - 20_000; // -80_000 / 4

    // Channel-interleaved layout: idx = (h*W+w)*C + c
    for i in 0..4 {
        let idx_ch0 = i * 2;
        let idx_ch1 = i * 2 + 1;
        assert(input_grad[idx_ch0] == expected_ch0, "Positive ch0 gradient must propagate");
        assert(input_grad[idx_ch1] == expected_ch1, "Negative ch1 gradient must propagate");
    }
}

#[test]
unconstrained fn test_max_pool2d_simple() {
    // Test Max Pooling on 2x2 input with 2x2 kernel
    // Input: [1, 2, 3, 4] (2x2)
    // Kernel: 2x2, stride: 2
    // Expected output: [4] (max of all values)
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let output = pooling::max_pool2d::<2, 2, 1, 2, 2, 1, 1>(input);

    assert(output[0] == 4_000_000, "Max pool should return maximum value");
}

#[test]
unconstrained fn test_max_pool2d_multiple_pools() {
    // Test Max Pooling on 4x4 input with 2x2 kernel, stride 2
    // Input: 4x4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    // Output: 2x2 = [max(1,2,5,6), max(3,4,7,8), max(9,10,13,14), max(11,12,15,16)]
    //         = [6, 8, 14, 16]
    let input: [Field; 16] = [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000,
        9_000_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000, 15_000_000,
        16_000_000,
    ];
    let output = pooling::max_pool2d::<4, 4, 1, 2, 2, 2, 2>(input);

    assert(output[0] == 6_000_000, "First pool max should be 6");
    assert(output[1] == 8_000_000, "Second pool max should be 8");
    assert(output[2] == 14_000_000, "Third pool max should be 14");
    assert(output[3] == 16_000_000, "Fourth pool max should be 16");
}

#[test]
unconstrained fn test_avg_pool2d_simple() {
    // Test Average Pooling on 2x2 input with 2x2 kernel
    // Inputs in fixed-point: [1M, 2M, 3M, 4M] -> avg = 10M / 4 = 2.5M
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let output = pooling::avg_pool2d::<2, 2, 1, 2, 2, 1, 1>(input);

    let expected: Field = 2_500_000;
    let diff = if expected.lt(output[0]) {
        output[0] - expected
    } else {
        expected - output[0]
    };
    assert(diff.lt(1001), "Avg pool should return average value");
}

#[test]
unconstrained fn test_max_pool2d_backward() {
    // Test backward pass through Max Pooling
    // Input: [1, 2, 3, 4] (2x2)
    // Max is at position 3 (value 4)
    // Output gradient: [100_000]
    // Input gradient: [0, 0, 0, 100_000] (only max position gets gradient)
    let input: [Field; 4] = [1_000_000, 2_000_000, 3_000_000, 4_000_000];
    let output_grad: [Field; 1] = [100_000];

    let input_grad = pooling::max_pool2d_backward::<2, 2, 1, 2, 2, 1, 1>(input, output_grad);

    // Only the max position (index 3) should get the gradient
    assert(input_grad[0] == 0, "Non-max position should have zero gradient");
    assert(input_grad[1] == 0, "Non-max position should have zero gradient");
    assert(input_grad[2] == 0, "Non-max position should have zero gradient");
    assert(input_grad[3] == 100_000, "Max position should get the gradient");
}

#[test]
unconstrained fn test_avg_pool2d_backward() {
    // Test backward pass through Average Pooling
    // Output gradient: [100_000]
    // Input gradient: [100_000/4, 100_000/4, 100_000/4, 100_000/4] (broadcast)
    let output_grad: [Field; 1] = [100_000];
    let input_grad = pooling::avg_pool2d_backward::<2, 2, 1, 2, 2, 1, 1>(output_grad);

    // Each position should get output_grad / kernel_area = 100_000 / 4 = 25_000
    let expected: Field = 25_000;
    for i in 0..4 {
        let diff = if expected.lt(input_grad[i]) {
            input_grad[i] - expected
        } else {
            expected - input_grad[i]
        };
        assert(diff.lt(1001), "Input gradient should be broadcast correctly");
    }
}
