import torch

from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant
from utils import get_global_scale, get_torch_results


@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    print("test nvfp4_gemm: ", shape)
    m, n, k = shape
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")
    bias = torch.randn((1, n), dtype=dtype, device="cuda")

    a_global_scale = get_global_scale(a_dtype)
    b_global_scale = get_global_scale(b_dtype)

    print(f"a_global_scale : {a_global_scale}, {a_global_scale.shape}")
    print(f"b_global_scale : {b_global_scale}, {b_global_scale.shape}")

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

    out = nvfp4_matmul(a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved,
                       alpha, bias)

    torch.testing.assert_close(out, expected_out, atol=1e-1, rtol=1e-1)

    print("test passed\n")


if __name__ == "__main__":
    torch.manual_seed(345)
    test_nvfp4_gemm(torch.bfloat16, (128, 128, 128))
    test_nvfp4_gemm(torch.bfloat16, (10, 512, 512))
    test_nvfp4_gemm(torch.bfloat16, (1024, 1024, 384))
    test_nvfp4_gemm(torch.bfloat16, (130, 384, 256))
    test_nvfp4_gemm(torch.bfloat16, (666, 768, 512))
    test_nvfp4_gemm(torch.bfloat16, (123, 321, 256))
    test_nvfp4_gemm(torch.bfloat16, (10, 15, 512))
    test_nvfp4_gemm(torch.bfloat16, (8192, 8192, 8192))
