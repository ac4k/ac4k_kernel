import torch
import torch.nn.functional as F
import math

from ac4k_kernel.ops import nvfp4_matmul, nvfp4_attention, quantize


def cosine_similarity_4d_global(tensor1, tensor2, eps=1e-8):
    """
    计算两个4D Tensor的整体余弦相似度
    将整个Tensor展平为向量后计算相似度
    """
    # 展平为1D向量 [batch, channel, height, width] -> [n]
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()

    # 计算余弦相似度
    similarity = F.cosine_similarity(
        t1_flat.unsqueeze(0),  # 添加batch维度
        t2_flat.unsqueeze(0),
        dim=1,
        eps=eps)
    return similarity.item()  # 返回标量值


def test_attention_bench(B, N, H, D, layout):
    assert layout in ["BHND", "BNHD"], "Unsupported layout {}".format(layout)

    if layout == "BNHD":
        q = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")

    def ceil_div(a, b):
        return (a + b - 1) // b

    def align_up(a, b):
        return ceil_div(a, b) * b

    def pad(tensor, align_size):
        # pad last dim
        if tensor.shape[-1] % align_size != 0:
            pad = (0, align_size - (tensor.shape[-1] % align_size)
                   ) + (0, 0) * (len(tensor.shape) - 1)
            return F.pad(tensor, pad=pad, mode='constant', value=0)
        return tensor

    def get_ref(q, k, v, B, N, H, D):
        q_fp4, q_sf, q_alpha = quantize(q, 1, 3)
        k_fp4, k_sf, k_alpha = quantize(k, 1, 3)
        v_fp4, v_sf, v_alpha = quantize(v, 3, 1)
        qk_alpha = q_alpha * k_alpha

        OUT = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")

        TILE_SIZE = 64
        FLOAT4_E2M1_MAX = 6.0
        FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
        alpha = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX /
                 torch.full([], 1, dtype=torch.bfloat16, device="cuda")).to(
                     torch.float32)
        for b in range(B):
            for h in range(H):
                max = torch.full((N, 1),
                                 -torch.inf,
                                 dtype=torch.float32,
                                 device="cuda")
                sum_exp = torch.full((N, 1),
                                     0,
                                     dtype=torch.float32,
                                     device="cuda")
                o = torch.zeros((N, D), dtype=torch.float32, device="cuda")
                for n in range(0, N, TILE_SIZE):
                    n_end = n + TILE_SIZE if n + TILE_SIZE < N else N
                    assert n != n_end
                    n_end_align = align_up(n_end, 16)
                    n_end_align64 = align_up(n_end, 64)
                    # dv_align = align_up(D, 16)
                    sub_q = q_fp4[b, h, :, :].contiguous()
                    sub_k = k_fp4[b, h, n:n_end, :].contiguous()
                    sub_v = v_fp4[b, h, :,
                                  n // 2:n_end_align64 // 2].contiguous()
                    sub_q_sf = q_sf[b, h, :, :, :].contiguous()
                    sub_k_sf = k_sf[b, h, :, n:n_end_align, :].contiguous()
                    sub_v_sf = v_sf[b, h, n // 64:(n + TILE_SIZE) //
                                    64, :, :].contiguous()

                    sub_q = sub_q.reshape(-1, sub_q.shape[-1])
                    sub_k = sub_k.reshape(-1, sub_k.shape[-1])
                    sub_v = sub_v.reshape(-1, sub_v.shape[-1])
                    sub_q_sf = sub_q_sf.reshape(-1, sub_q_sf.shape[-2], 4)
                    sub_k_sf = sub_k_sf.reshape(-1, sub_k_sf.shape[-2], 4)
                    sub_v_sf = sub_v_sf.reshape(-1, sub_v_sf.shape[-2], 4)

                    # s = q @ k
                    s = nvfp4_matmul(sub_q, sub_k, sub_q_sf, sub_k_sf,
                                     qk_alpha)

                    s = s.to(torch.float32)
                    s = s / math.sqrt(D)

                    # softmax
                    max_new = torch.max(
                        torch.max(s, dim=-1, keepdim=True).values, max)
                    p = torch.exp(s - max_new)
                    sum_exp_new = torch.sum(p, dim=-1, keepdim=True)

                    # update
                    o = torch.exp(max - max_new) * o
                    sum_exp = torch.exp(max - max_new) * sum_exp + sum_exp_new
                    max = max_new

                    # qunatize
                    p = p.to(torch.bfloat16)
                    p_fp4, p_sf, p_alpha = quantize(p, 0, 1, alpha=alpha)

                    # o = p @ v
                    o = nvfp4_matmul(
                        p_fp4, sub_v, p_sf, sub_v_sf,
                        torch.full([], 1, dtype=torch.float32,
                                   device="cuda")).to(torch.float32) + o

                o = o * v_alpha / alpha
                o = o / sum_exp
                OUT[b, :, h, :] = o.to(torch.bfloat16)

        print("get_ref o:")
        print(OUT[0, :4, 0, :8])

        return OUT

    def get_ac4k(q, k, v, layout):
        o = nvfp4_attention(q, k, v, layout=layout)

        return o

    def get_sdp(q, k, v, layout):
        if layout == "BHND":
            return F.scaled_dot_product_attention(q, k, v)
        else:
            return F.scaled_dot_product_attention(q.transpose(1, 2),
                                                  k.transpose(1, 2),
                                                  v.transpose(1, 2)).transpose(
                                                      1, 2).contiguous()

    # o_ref = get_ref(q, k, v, B, N, H, D)
    o_sdp = get_sdp(q, k, v, layout)
    print("o ac4k:")
    print(o_sdp[0, :4, 0, :8])

    o = get_ac4k(q, k, v, layout)
    print("o ac4k:")
    print(o[0, :4, 0, :8])

    similarity = cosine_similarity_4d_global(o, o_sdp)
    print(f"o vs o_sdp 整体余弦相似度: {similarity:.4f}")
    # similarity1 = cosine_similarity_4d_global(o, o_ref)
    # print(f"o vs o_ref 整体余弦相似度1: {similarity1:.4f}")
    # similarity2 = cosine_similarity_4d_global(o_ref, o_sdp)
    # print(f"o_ref vs o_sdp 整体余弦相似度2: {similarity2:.4f}")

    torch.testing.assert_close(o, o_sdp, atol=2e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(9567)
    test_attention_bench(1, 64, 1, 128, "BNHD")
    test_attention_bench(2, 128, 3, 128, "BNHD")
    test_attention_bench(1, 33, 1, 128, "BNHD")
    test_attention_bench(10, 330, 12, 100, "BNHD")
    test_attention_bench(1, 444, 1, 88, "BNHD")

    test_attention_bench(2, 111, 2, 100, "BHND")
