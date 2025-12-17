import torch

from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant
from utils import get_global_scale, get_torch_results


@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    test_runs: int = 10,
    warmup_runs: int = 5,
) -> None:
    print("test nvfp4_gemm: ", shape)
    m, n, k = shape
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")
    bias = torch.randn((1, n), dtype=dtype, device="cuda")

    a_global_scale = get_global_scale(a_dtype)
    b_global_scale = get_global_scale(b_dtype)

    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = nvfp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = nvfp4_quant(b_dtype, b_global_scale)

    expected_out = get_torch_results(a_fp4,
                                     b_fp4,
                                     a_scale_interleaved,
                                     b_scale_interleaved,
                                     a_global_scale,
                                     b_global_scale,
                                     m,
                                     n,
                                     dtype,
                                     block_size,
                                     bias=bias)

    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    out = nvfp4_matmul(a_fp4,
                       b_fp4,
                       a_scale_interleaved,
                       b_scale_interleaved,
                       alpha,
                       bias=bias,
                       out=out)
    torch.testing.assert_close(out, expected_out, atol=1e-1, rtol=1e-1)
    print("The result comparison has passed.")

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("Warming up...")
    for _ in range(warmup_runs):
        nvfp4_matmul(a_fp4,
                     b_fp4,
                     a_scale_interleaved,
                     b_scale_interleaved,
                     alpha,
                     bias=bias,
                     out=out)
    torch.cuda.synchronize()  # Make sure the warmup is done

    # Test performance: measure the time taken for multiple runs
    print(f"Testing {test_runs} runs...")
    start_event.record()
    for _ in range(test_runs):
        nvfp4_matmul(a_fp4,
                     b_fp4,
                     a_scale_interleaved,
                     b_scale_interleaved,
                     alpha,
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
    test_nvfp4_gemm(torch.bfloat16, (75600, 5120, 13824),
                    test_runs=20,
                    warmup_runs=5)
    test_nvfp4_gemm(torch.bfloat16, (75600, 13824, 5120),
                    test_runs=20,
                    warmup_runs=5)
    test_nvfp4_gemm(torch.bfloat16, (75600, 5120, 5120),
                    test_runs=20,
                    warmup_runs=5)
    # test_nvfp4_gemm(torch.bfloat16, (75600, 5120, 13824 // 2),
    #                 test_runs=20,
    #                 warmup_runs=5)
