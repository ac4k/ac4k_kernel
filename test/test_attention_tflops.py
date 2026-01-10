from ac4k_kernel.ops import attention

import torch


def test_performance(B, N, H, D, warmup_runs: int = 5, repeat: int = 10):
    q = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("Warming up...")
    for _ in range(warmup_runs):
        attention(q, k, v, precision="int8+fp8e4m3")

    torch.cuda.synchronize()  # Make sure the warmup is done

    print("Running benchmark...")
    start_event.record()
    for _ in range(repeat):
        attention(q, k, v, precision="int8+fp8e4m3")
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded

    elapsed_time = start_event.elapsed_time(end_event) / repeat

    print(f"Average time per attention: {elapsed_time} ms")
    print(
        f"Compute Power: {2 * 2 * B * H * N * N * D / (elapsed_time * 1e-3) / 1e12} TFLOPs"
    )


if __name__ == "__main__":
    torch.manual_seed(9567)
    test_performance(1, 75600, 40, 128)
    test_performance(1, 75600, 40, 64)
