import torch
import torch.nn.functional as F
import math

from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant, nvfp4_attention, quantize
from utils import get_global_scale


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


def test_attention_bench(B, N, H, D):
    B = 11
    N = 128
    H = 12
    D = 128
    q = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((B, N, H, D), dtype=torch.bfloat16, device="cuda")

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

        def get_single(q, k, v, B, N, H, D):
            # quantize q & k
            q_rcp_global_scale = get_global_scale(q)
            q_fp4, q_sf = nvfp4_quant(q, q_rcp_global_scale)
            k_rcp_global_scale = get_global_scale(k)
            k_fp4, k_sf = nvfp4_quant(k, k_rcp_global_scale)
            qk_alpha = 1.0 / (q_rcp_global_scale * k_rcp_global_scale)

            # quantize v
            v = v.T.contiguous()
            # pad v
            v = pad(v, 128)
            # quantize v
            v_rcp_global_scale = get_global_scale(v)
            v_fp4, v_sf = nvfp4_quant(v, v_rcp_global_scale)
            v_alpha = 1.0 / v_rcp_global_scale

            TILE_SIZE = 64
            max = torch.full((N, 1), -torch.inf, device="cuda")
            sum_exp = torch.zeros((N, 1), dtype=torch.float32, device="cuda")
            o = torch.zeros((N, D), dtype=torch.float32, device="cuda")
            for i in range(0, N, TILE_SIZE):
                # tile sub k
                k_fp4_sub = k_fp4[i:i + TILE_SIZE].contiguous()
                k_sf_sub = k_sf.view(torch.int32)
                k_sf_sub = k_sf_sub.reshape(-1, k_sf_sub.shape[1], 32, 4)
                k_sf_sub = k_sf_sub[i // 128, :, :, ((i // 64) % 2) *
                                    2:((i // 64) % 2) * 2 + 2]
                k_sf_sub = pad(k_sf_sub, 4).contiguous()
                k_sf_sub = k_sf_sub.view(torch.float8_e4m3fn)
                k_sf_sub = k_sf_sub.reshape(-1, k_sf.shape[1])

                # s = q @ k
                s = nvfp4_matmul(q_fp4, k_fp4_sub, q_sf, k_sf_sub, qk_alpha)
                s = s.to(torch.float32)
                s = s / math.sqrt(D)

                # softmax
                max_new = torch.max(
                    torch.max(s, dim=-1, keepdim=True).values, max)
                exp = torch.exp(s - max_new)
                sum_exp_new = torch.sum(exp, dim=-1, keepdim=True)

                # quantize exp
                exp = exp.to(torch.bfloat16)
                exp_rcp_global_scale = get_global_scale(exp)
                exp_alpha = 1.0 / exp_rcp_global_scale
                exp_fp4, exp_sf = nvfp4_quant(exp, exp_rcp_global_scale)

                # tile sub v
                v_fp4_sub = v_fp4[:, i // 2:(i + TILE_SIZE) // 2].contiguous()
                v_sf_sub = v_sf.view(torch.int32)
                v_sf_sub = v_sf_sub.reshape(v_sf_sub.shape[0] // 128,
                                            v_sf_sub.shape[1], -1)
                v_sf_sub = v_sf_sub[:, i // 64:(i + TILE_SIZE) //
                                    64, :].contiguous()
                v_sf_sub = v_sf_sub.view(torch.float8_e4m3fn)
                v_sf_sub = v_sf_sub.reshape(v_sf.shape[0], -1)

                # o = exp @ v
                o_new = nvfp4_matmul(exp_fp4, v_fp4_sub, exp_sf, v_sf_sub,
                                     exp_alpha * v_alpha).to(torch.float32)

                # update
                sum_exp = torch.exp(max - max_new) * sum_exp + sum_exp_new
                o = torch.exp(max - max_new) * o + o_new
                max = max_new

            o = (o / sum_exp).to(torch.bfloat16)

            return o

        o = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        for b in range(B):
            for h in range(H):
                q_sub = q[b, :, h, :].contiguous()
                k_sub = k[b, :, h, :].contiguous()
                v_sub = v[b, :, h, :].contiguous()
                o_sub = get_single(q_sub, k_sub, v_sub, 1, N, 1, D)
                o[b, :, h, :] = o_sub

        print("o ref:")
        print(o[:2, :2, :2, :2])

        return o.reshape(B, N, H, D)

    def get_ref1(q, k, v, B, N, H, D):
        s = torch.matmul(q, k.transpose(-2, -1)).to(torch.float32)
        s = s / math.sqrt(D)

        p = torch.softmax(s, dim=-1).to(torch.bfloat16)

        o = torch.matmul(p, v)

        print("o ref1:")
        print(o)

        return o.reshape(B, N, H, D)

    def get_sdp(q, k, v):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(
            1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()

        print("o sdp:")
        print(o[:2, :2, :2, :2])

        return o

    def get_ac4k(q, k, v, B, N, H, D):
        # ## new2
        # q = q.view(B, N, H, -1)
        # q_fp4, q_sf, q_alpha = quantize(q, dim=-1)

        # k = k.view(B, N, H, -1)
        # k_fp4, k_sf, k_alpha = quantize(k, dim=-1)

        # qk_alpha = q_alpha * k_alpha

        # v = v.view(B, N, H, -1)
        # v_fp4, v_sf, v_alpha = quantize(v, dim=1, swizzle=True)

        # o = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        # o = nvfp4_attention(q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, qk_alpha,
        #                     v_alpha, o)

        ## new2
        q_fp4, q_sf, q_alpha = quantize(q, dim=-1)

        k_fp4, k_sf, k_alpha = quantize(k, dim=-1)

        qk_alpha = q_alpha * k_alpha

        v_fp4, v_sf, v_alpha = quantize(v, dim=1, swizzle=True)

        o = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        o = nvfp4_attention(q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, qk_alpha,
                            v_alpha, o)

        print("o ac4k:")
        print(o[:2, :2, :2, :2])

        return o

    # o_ref = get_ref(q, k, v, B, N, H, D)
    # o_ref1 = get_ref1(q, k, v, B, N, H, D)
    o_ref = get_ref(q, k, v, B, N, H, D)
    o_sdp = get_sdp(q, k, v)
    o = get_ac4k(q, k, v, B, N, H, D)

    similarity = cosine_similarity_4d_global(o, o_sdp)
    print(f"o vs o_sdp 整体余弦相似度: {similarity:.4f}")
    similarity1 = cosine_similarity_4d_global(o, o_ref)
    print(f"o vs o_ref 整体余弦相似度1: {similarity1:.4f}")
    similarity2 = cosine_similarity_4d_global(o_ref, o_sdp)
    print(f"o_ref vs o_sdp 整体余弦相似度2: {similarity2:.4f}")

    torch.testing.assert_close(o, o_ref, atol=2e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(9567)
    test_attention_bench(1, 64, 1, 128)
