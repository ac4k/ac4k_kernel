import torch
from ac4k_kernel.ops import nvfp4_quantize


def test_quantize_bench(shape,
                        cross_dim,
                        reduce_dim,
                        swizzle=False,
                        warmup=5,
                        repeat=10):
    print(
        "test quantize cross_dim={}, reduce_dim={}, shape={}, swize={}".format(
            cross_dim, reduce_dim, shape, swizzle))
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    nvfp4_quantize(input, cross_dim, reduce_dim, swizzle=swizzle)

    for _ in range(warmup):
        nvfp4_quantize(input, cross_dim, reduce_dim, swizzle=swizzle)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  # Make sure the warmup is done

    start_event.record()
    for i in range(repeat):
        output, sf, _ = nvfp4_quantize(input,
                                       cross_dim,
                                       reduce_dim,
                                       swizzle=swizzle)
    end_event.record()
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / repeat
    print(f"Average time for quantize_kernel: {avg_time_ms} ms")
    io = input.nbytes + output.nbytes + sf.nbytes
    print(f"Throughput: {io / avg_time_ms / 1e6} GB/s")


if __name__ == "__main__":
    test_quantize_bench((1, 75600, 40, 128), 1, 3)
    test_quantize_bench((1, 75600, 40, 128), 3, 1, True)
