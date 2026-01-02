import torch
import torch.nn.functional as F

from ac4k_kernel.ops import attention


def cosine_similarity_4d_global(tensor1, tensor2, eps=1e-8):
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()

    similarity = F.cosine_similarity(t1_flat.unsqueeze(0),
                                     t2_flat.unsqueeze(0),
                                     dim=1,
                                     eps=eps)
    return similarity.item()


def test_attention_bench(B,
                         N,
                         H,
                         D,
                         layout,
                         precision="nvfp4",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.98):
    assert layout in ["BHND", "BNHD"], "Unsupported layout {}".format(layout)

    if layout == "BNHD":
        q = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")

    def get_ac4k(q, k, v, layout, precision):
        o = attention(q, k, v, layout=layout, precision=precision)

        return o

    def get_sdp(q, k, v, layout):
        if layout == "BHND":
            return F.scaled_dot_product_attention(q, k, v)
        else:
            return F.scaled_dot_product_attention(q.transpose(1, 2),
                                                  k.transpose(1, 2),
                                                  v.transpose(1, 2)).transpose(
                                                      1, 2).contiguous()

    o_sdp = get_sdp(q, k, v, layout)
    print("o sdp:")
    print(o_sdp[0, :4, 0, :8])

    o = get_ac4k(q, k, v, layout, precision)
    print("o ac4k:")
    print(o[0, :4, 0, :8])

    similarity = cosine_similarity_4d_global(o, o_sdp)
    print(f"o vs o_sdp similarity: {similarity:.4f}")
    assert similarity > stol
    torch.testing.assert_close(o, o_sdp, atol=atol, rtol=rtol)


if __name__ == "__main__":
    torch.manual_seed(9567)
    test_attention_bench(1,
                         75600,
                         1,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(1,
                         100,
                         1,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(10,
                         330,
                         12,
                         100,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(1,
                         256,
                         1,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(1,
                         128,
                         1,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(2,
                         128,
                         3,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(1,
                         33,
                         1,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(1,
                         444,
                         1,
                         88,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(2,
                         111,
                         2,
                         100,
                         "BHND",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)
    test_attention_bench(1,
                         256,
                         1,
                         128,
                         "BNHD",
                         precision="int8+fp8e4m3",
                         atol=2e-1,
                         rtol=1e-2,
                         stol=0.99)

    test_attention_bench(1,
                         256,
                         1,
                         128,
                         "BNHD",
                         precision="nvfp4",
                         atol=2e-1,
                         rtol=1e-1)
    test_attention_bench(1,
                         64,
                         1,
                         128,
                         "BNHD",
                         precision="nvfp4",
                         atol=2e-1,
                         rtol=1e-1)
