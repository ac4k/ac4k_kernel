import torch
import triton
import triton.language as tl


@triton.jit
def compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :],
                mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d,
             x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(k, dim=-2,
                           keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    LSE,
    OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)


@triton.jit
def _attn_bwd_preprocess(
    OS,
    DOS,
    DELTAS,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    OS += idx_bh * L * D
    DOS += idx_bh * L * D
    DELTAS += idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :],
                  mask=offs_m[:, None] < L)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :],
                   mask=offs_m[:, None] < L)

    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L)


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    LSE,
    DELTAS,
    DOS,
    DQ,
    LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    lse_offset = idx_bh * L
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    # load Q, DOS, DOL, LSE, DELTA, S: they stay in SRAM throughout the inner loop.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L)
    lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)

        # Compute dP and dS.
        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        # Compute dQ.
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)


@triton.jit
def _attn_bwd_dkdv(
    Q,
    K,
    V,
    DOS,
    DK,
    DV,
    qk_scale,
    KBID,
    LSE,
    DELTAS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * L

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    KBID_ptr = KBID + kbid_offset + idx_n

    # load K, V and CK: they stay in SRAM throughout the inner loop.
    k = tl.load(K_ptrs, mask=offs_n[:, None] < L)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for idx_m in tl.range(0, L, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634
                                    )  # = 1 / ln(2)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            # Compute dV.
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        # Increment pointers
        Q_ptrs += BLOCK_M2 * D
        DOS_ptrs += BLOCK_M2 * D
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    # Write back dK, dV and dCK
    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                k_block_id,
                lut,
                topk,
                BLOCK_M,
                BLOCK_N,
                qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        # We recommend the following two settings
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, D = q.shape
        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        o_s = torch.empty_like(v)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](q,
                        k,
                        v,
                        qk_scale,
                        topk,
                        lut,
                        lse,
                        o_s,
                        L,
                        M_BLOCKS,
                        D,
                        BLOCK_M,
                        BLOCK_N,
                        num_warps=4 if q.shape[-1] == 64 else 8,
                        num_stages=3)

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            o_s,
            do_s,
            delta_s,
            L,
            D,
            BLOCK_M,
        )

        grid = (M_BLOCKS, B * H)
        _attn_bwd_dq[grid](q,
                           k,
                           v,
                           lse,
                           delta_s,
                           do_s,
                           dq,
                           lut,
                           ctx.qk_scale,
                           ctx.topk,
                           L,
                           M_BLOCKS,
                           D,
                           BLOCK_M,
                           BLOCK_N,
                           num_warps=4 if q.shape[-1] == 64 else 8,
                           num_stages=4 if q.shape[-1] == 64 else 5)

        grid = (N_BLOCKS, B * H)
        _attn_bwd_dkdv[grid](q,
                             k,
                             v,
                             do_s,
                             dk,
                             dv,
                             ctx.qk_scale,
                             k_block_id,
                             lse,
                             delta_s,
                             L,
                             M_BLOCKS,
                             N_BLOCKS,
                             D,
                             BLOCK_M,
                             BLOCK_N,
                             BLOCK_SLICE_FACTOR=BLOCK_M // 64,
                             num_warps=4 if q.shape[-1] == 64 else 8,
                             num_stages=4 if q.shape[-1] == 64 else 5)

        return dq, dk, dv, None, None, None, None, None, None


class SparseLinearAttention(torch.nn.Module):

    def __init__(self,
                 head_dim,
                 topk_ratio,
                 kernel_type='softmax',
                 BLOCK_Q=64,
                 BLOCK_KV=64,
                 use_bf16=True):
        '''
        Sparse Linear Attention.
        o = sparse_attn + Proj(linear_atten)
        Pc = softmax(pool(Q) @ pool(K))
        Mc = 1 if Pc is in topk else 0
        for j in range(0, L, BLOCK_KV):
            if Mc = 1:
                full attention
            else:
                linear attention

        Args:
            head_dim: dimension of each head.
            topk_ratio: ratio of keys selected for sparse attention, shared across all queries.
            kernel_type: kernel type for linear attention, one of ['elu', 'relu', 'softmax'].
            BLOCK_Q: block size for query.
            BLOCK_KV: block size for key and value.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
        '''

        assert kernel_type in ['elu', 'relu', 'softmax'
                               ], f'Not supported feature map {kernel_type}.'

        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk_ratio = topk_ratio
        self.BLOCK_Q = BLOCK_Q
        self.BLOCK_KV = BLOCK_KV
        self.proj_l = torch.nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if kernel_type == 'elu':

            def elu_kernel(x):
                return torch.nn.functional.elu(x) + 1

            self.q_kernel = elu_kernel
            self.k_kernel = elu_kernel
        elif kernel_type == 'relu':
            self.q_kernel = torch.nn.ReLU()
            self.k_kernel = torch.nn.ReLU()
        else:

            def softmax_kernel(x):
                return torch.nn.functional.softmax(x, dim=-1)

            self.q_kernel = softmax_kernel
            self.k_kernel = softmax_kernel

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            torch.nn.init.zeros_(self.proj_l.weight)
            torch.nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        '''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        '''
        dtype = q.dtype

        # Preprocess q/k/v
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Sparse attention
        sparse_map, lut, real_topk = get_block_map(q,
                                                   k,
                                                   topk_ratio=self.topk_ratio,
                                                   BLKQ=self.BLOCK_Q,
                                                   BLKK=self.BLOCK_KV)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        o_s = _attention.apply(q, k, v, sparse_map, lut, real_topk,
                               self.BLOCK_Q, self.BLOCK_KV)

        # Linear attention
        def linear_attn(q, k, v):
            epsilon = 1e-5
            q = self.q_kernel(q).contiguous().to(self.dtype)
            k = self.k_kernel(k).contiguous().to(self.dtype)
            h = k.transpose(-1, -2) @ v
            z = torch.sum(k, dim=-2, keepdim=True)
            return (q @ h) / (epsilon + (q * z).sum(dim=-1, keepdim=True))

        o_l = linear_attn(q, k, v)

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o
