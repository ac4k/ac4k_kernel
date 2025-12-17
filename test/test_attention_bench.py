import torch
import torch.nn.functional as F
import math

from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant, nvfp4_attention, quantize
from utils import get_global_scale


def test_attention_bench(B, N, H, D):
    B = 1
    N = 64
    H = 1
    D = 128
    q = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")

    # q_global_scale = get_global_scale(q)
    # k_global_scale = get_global_scale(k)
    # v = v.T.contiguous()
    # v_reorder = v.reshape(D, -1, 4, 4, 2)
    # v_reorder = v_reorder.permute(0, 1, 3, 2, 4)
    # v_reorder = v_reorder.contiguous()
    # v_reorder = v_reorder.reshape(D, -1)
    # v_reorder = v_reorder.contiguous()
    # v_reorder_global_scale = get_global_scale(v_reorder)
    # # v_global_scale = get_global_scale(v)

    # qk_alpha = 1.0 / (q_global_scale * k_global_scale)
    # # v_alpha = 1.0 / v_global_scale
    # v_reorder_alpha = 1.0 / v_reorder_global_scale

    # q_fp4, q_sf = nvfp4_quant(q, q_global_scale)
    # k_fp4, k_sf = nvfp4_quant(k, k_global_scale)
    # # v_fp4, v_sf = nvfp4_quant(v, v_global_scale)
    # v_reorder_fp4, v_reorder_sf = nvfp4_quant(v_reorder, v_reorder_global_scale)

    ####
    def get_ref(q, k, v, B, N, H, D):
        vt = v.T.contiguous()

        q_global_scale = get_global_scale(q)
        k_global_scale = get_global_scale(k)
        qk_alpha = 1.0 / (q_global_scale * k_global_scale)

        q_fp4, q_sf = nvfp4_quant(q, q_global_scale)
        k_fp4, k_sf = nvfp4_quant(k, k_global_scale)

        s = nvfp4_matmul(q_fp4, k_fp4, q_sf, k_sf, qk_alpha)
        s = s / math.sqrt(D)

        max = torch.max(s, dim=-1, keepdim=True)

        exp = torch.exp(s - max.values)

        sum_exp = torch.sum(exp, dim=-1, keepdim=True)

        # pad and quantize
        exp_pad = F.pad(exp, pad=(0, 64, 0, 0), mode='constant', value=0)
        # quantize exp
        exp_pad_global_scale = get_global_scale(exp_pad)
        exp_pad_alpha = 1.0 / exp_pad_global_scale
        exp_pad_fp4, exp_pad_sf = nvfp4_quant(exp_pad, exp_pad_global_scale)

        # pad v
        v_pad = F.pad(vt, pad=(0, 64, 0, 0), mode='constant', value=0)
        # quantize v
        v_pad_global_scale = get_global_scale(v_pad)
        v_pad_alpha = 1.0 / v_pad_global_scale
        v_pad_fp4, v_pad_sf = nvfp4_quant(v_pad, v_pad_global_scale)

        # o
        float1 = torch.ones([], dtype=torch.float32, device="cuda")
        o = nvfp4_matmul(exp_pad_fp4, v_pad_fp4, exp_pad_sf, v_pad_sf, float1)

        o *= exp_pad_alpha * v_pad_alpha

        o = o / sum_exp

        print("o ref:")
        print(o)

        return o.reshape(B, N, H, D)

    def get_ac4k(q, k, v, B, N, H, D):
        q_global_scale = get_global_scale(q)
        k_global_scale = get_global_scale(k)
        qk_alpha = 1.0 / (q_global_scale * k_global_scale)
        q_fp4, q_sf = nvfp4_quant(q, q_global_scale)
        k_fp4, k_sf = nvfp4_quant(k, k_global_scale)

        q_fp4 = q_fp4.view(B, N, H, -1)
        k_fp4 = k_fp4.view(B, N, H, -1)

        q_sf = q_sf.view(B, H, q_sf.shape[0], q_sf.shape[1])
        k_sf = k_sf.view(B, H, k_sf.shape[0], k_sf.shape[1])

        v = v.view(B, N, H, -1)
        v_fp4, v_sf, v_alpha = quantize(v, dim=1, swizzle=True)

        o = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")
        o = nvfp4_attention(q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, qk_alpha,
                            v_alpha, o)

        print("o ac4k:")
        print(o)
        return o

    o_ref = get_ref(q, k, v, B, N, H, D)

    o = get_ac4k(q, k, v, B, N, H, D)

    # q_fp4 = q_fp4.view(B, N, H, -1)
    # k_fp4 = k_fp4.view(B, N, H, -1)
    # v_reorder_fp4 = v_reorder_fp4.view(B, D, H, -1)

    # q_sf = q_sf.view(B, H, q_sf.shape[0], q_sf.shape[1])
    # k_sf = k_sf.view(B, H, k_sf.shape[0], k_sf.shape[1])
    # v_reorder_sf = v_reorder_sf.view(B, H, v_reorder_sf.shape[0], v_reorder_sf.shape[1])

    # o = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")

    # o = nvfp4_attention(q_fp4, k_fp4, v_reorder_fp4, q_sf, k_sf, v_reorder_sf, qk_alpha, v_reorder_alpha, o)
    # print("o")
    # print(o)

    torch.testing.assert_close(o, o_ref, atol=2e-1, rtol=1e-1)


if __name__ == "__main__":
    torch.manual_seed(9567)
    test_attention_bench(1, 64, 1, 128)

# ##########################
# 20251216
# ##########################
# import torch
# import torch.nn.functional as F
# import math

# from ac4k_kernel.ops import nvfp4_matmul, nvfp4_quant, nvfp4_attention, quantize
# from utils import get_global_scale

# def test_attention_bench(B, N, H, D):
#     B = 1
#     N = 64
#     H = 1
#     D = 128
#     q = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")
#     k = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")
#     v = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")

#     q_global_scale = get_global_scale(q)
#     k_global_scale = get_global_scale(k)
#     v = v.T.contiguous()
#     v_reorder = v.reshape(D, -1, 4, 4, 2)
#     v_reorder = v_reorder.permute(0, 1, 3, 2, 4)
#     v_reorder = v_reorder.contiguous()
#     v_reorder = v_reorder.reshape(D, -1)
#     v_reorder = v_reorder.contiguous()
#     v_reorder_global_scale = get_global_scale(v_reorder)
#     # v_global_scale = get_global_scale(v)

#     qk_alpha = 1.0 / (q_global_scale * k_global_scale)
#     # v_alpha = 1.0 / v_global_scale
#     v_reorder_alpha = 1.0 / v_reorder_global_scale

#     q_fp4, q_sf = nvfp4_quant(q, q_global_scale)
#     k_fp4, k_sf = nvfp4_quant(k, k_global_scale)
#     # v_fp4, v_sf = nvfp4_quant(v, v_global_scale)
#     v_reorder_fp4, v_reorder_sf = nvfp4_quant(v_reorder, v_reorder_global_scale)

#     ####
#     def ref(q, k, v, B, N, H, D):
#         q_global_scale = get_global_scale(q)
#         k_global_scale = get_global_scale(k)
#         qk_alpha = 1.0 / (q_global_scale * k_global_scale)

#         q_fp4, q_sf = nvfp4_quant(q, q_global_scale)
#         k_fp4, k_sf = nvfp4_quant(k, k_global_scale)

#         s = nvfp4_matmul(q_fp4, k_fp4, q_sf, k_sf, qk_alpha)
#         s = s / math.sqrt(D)

#         max = torch.max(s, dim=-1, keepdim=True)

#         exp = torch.exp(s - max.values)

#         l = torch.sum(exp, dim=-1, keepdim=True)

#         # pad and quantize
#         exp_pad = F.pad(exp, pad=(0, 64, 0, 0), mode='constant', value=0)
#         # quantize exp
#         exp_pad_global_scale = get_global_scale(exp_pad)
#         exp_pad_alpha = 1.0 / exp_pad_global_scale
#         exp_pad_fp4, exp_pad_sf = nvfp4_quant(exp_pad, exp_pad_global_scale)

#         # pad v
#         v_pad = F.pad(v, pad=(0, 64, 0, 0), mode='constant', value=0)
#         # quantize v
#         v_pad_global_scale = get_global_scale(v_pad)
#         v_pad_alpha = 1.0 / v_pad_global_scale
#         v_pad_fp4, v_pad_sf = nvfp4_quant(v_pad, v_pad_global_scale)

#         # o
#         float1 = torch.ones([], dtype=torch.float32, device="cuda")
#         o = nvfp4_matmul(exp_pad_fp4, v_pad_fp4, exp_pad_sf, v_pad_sf, float1)
#         o *= exp_pad_alpha * v_pad_alpha

#         o = o / l

#         print("o ref:")
#         print(o)

#         return o.reshape(B, N, H, D)

#     o_ref = ref(q, k, v, B, N, H, D)

#     q_fp4 = q_fp4.view(B, N, H, -1)
#     k_fp4 = k_fp4.view(B, N, H, -1)
#     v_reorder_fp4 = v_reorder_fp4.view(B, D, H, -1)

#     q_sf = q_sf.view(B, H, q_sf.shape[0], q_sf.shape[1])
#     k_sf = k_sf.view(B, H, k_sf.shape[0], k_sf.shape[1])
#     v_reorder_sf = v_reorder_sf.view(B, H, v_reorder_sf.shape[0], v_reorder_sf.shape[1])

#     o = torch.empty((B, N, H, D), dtype=torch.bfloat16, device="cuda")

#     o = nvfp4_attention(q_fp4, k_fp4, v_reorder_fp4, q_sf, k_sf, v_reorder_sf, qk_alpha, v_reorder_alpha, o)
#     print("o")
#     print(o)

#     torch.testing.assert_close(o, o_ref, atol=2e-1, rtol=1e-1)

# if __name__ == "__main__":
#     torch.manual_seed(9567)
#     test_attention_bench(1, 64, 1, 128)
