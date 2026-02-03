import torch

from ac4k_kernel.ops import quantize, linear


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    assert a_sf_swizzled.ndim == 3

    tmp = a_sf_swizzled.permute(1, 0, 2)
    tmp = tmp.reshape(tmp.shape[0], -1)[:m, :k].contiguous()
    return tmp


def break_fp4_bytes(a):
    assert a.dtype == torch.uint8

    kE2M1ToFloatArray = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]

    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    map_tensor = torch.tensor(kE2M1ToFloatArray,
                              dtype=torch.float32,
                              device=a.device)
    fH = map_tensor[highHalfByte.long()].to(a.device)
    fL = map_tensor[lowHalfByte.long()].to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC]
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def dequantize_to_dtype(tensor_fp4, tensor_sf, global_scale, block_size=16):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


def get_torch_results(a_fp4,
                      b_fp4,
                      a_sf,
                      b_sf,
                      a_global_scale,
                      b_global_scale,
                      m,
                      n,
                      dtype,
                      block_size,
                      bias=None):
    assert a_fp4.dtype == torch.uint8
    assert b_fp4.dtype == torch.uint8
    assert a_fp4.dim() == 2
    assert a_fp4.size(0) == m
    assert b_fp4.dim() == 2
    assert b_fp4.size(0) == n
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k

    a_in_dtype = dequantize_to_dtype(a_fp4,
                                     a_sf,
                                     a_global_scale,
                                     block_size=block_size)
    b_in_dtype = dequantize_to_dtype(b_fp4,
                                     b_sf,
                                     b_global_scale,
                                     block_size=block_size)
    out = torch.matmul(a_in_dtype, b_in_dtype.t())

    if bias is not None:
        out += bias

    return out.to(dtype)


@torch.inference_mode()
def test_dot_scale_performance(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    test_runs: int = 10,
    warmup_runs: int = 5,
) -> None:
    print("test linear: ", shape)

    m, n, k = shape
    block_size = 16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = torch.randn((n, k), dtype=dtype, device="cuda")
    bias = None

    a_fp4, a_sf, a_global_scale = quantize(a, 0, 1)
    b_fp4, b_sf, b_global_scale = quantize(b, 0, 1)
    out = linear(a_fp4, a_sf, a_global_scale, b_fp4, b_sf, b_global_scale,
                    bias)

    # ref
    expected_out = get_torch_results(a_fp4,
                                     b_fp4,
                                     a_sf,
                                     b_sf,
                                     1 / a_global_scale,
                                     1 / b_global_scale,
                                     m,
                                     n,
                                     dtype,
                                     block_size,
                                     bias=bias)

    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("Warming up...")
    for _ in range(warmup_runs):
        linear(a_fp4,
                  a_sf,
                  a_global_scale,
                  b_fp4,
                  b_sf,
                  b_global_scale,
                  bias=bias,
                  out=out)
    torch.cuda.synchronize()  # Make sure the warmup is done

    # Test performance: measure the time taken for multiple runs
    print(f"Testing {test_runs} runs...")
    start_event.record()
    for _ in range(test_runs):
        linear(a_fp4,
                  a_sf,
                  a_global_scale,
                  b_fp4,
                  b_sf,
                  b_global_scale,
                  bias=bias,
                  out=out)
    end_event.record()

    # Synchronize and calculate the average time taken
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / test_runs
    throughput_tflops = (m * n * k * 2) / (avg_time_ms * 1e-3) / 1e12

    # Print the results
    print("=" * 50)
    print(f"test config: m={m}, n={n}, k={k}")
    print(f"total time: {total_time_ms:.2f} ms")
    print(f"avg time: {avg_time_ms:.4f} ms")
    print(f"throughput: {throughput_tflops:.2f} TFLOPS")
    print("=" * 50)
    print("\n")


if __name__ == "__main__":
    torch.manual_seed(345)
    test_dot_scale_performance(torch.bfloat16, (8192, 8192, 8192),
                               test_runs=20,
                               warmup_runs=5)
    test_dot_scale_performance(torch.bfloat16, (75600, 5120, 5120),
                               test_runs=20,
                               warmup_runs=5)
    test_dot_scale_performance(torch.bfloat16, (75600, 13824, 5120),
                               test_runs=20,
                               warmup_runs=5)
    test_dot_scale_performance(torch.bfloat16, (75600, 5120, 13824 // 2),
                               test_runs=20,
                               warmup_runs=5)
    test_dot_scale_performance(torch.bfloat16, (75600, 5120, 13824),
                               test_runs=20,
                               warmup_runs=5)
