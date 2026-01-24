import torch
import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


def _align_up(a, b):
    return _ceil_div(a, b) * b


@triton.jit
def _triton_mean_pool(X, Y, B, H, L, D, x_stride_b, x_stride_h, x_stride_l,
                      x_stride_d, y_stride_b, y_stride_h, y_stride_l,
                      y_stride_d, BLOCK_L: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D) % D

    x_offs = pid_b * x_stride_b + pid_h * x_stride_h + offs_l[:,
                                                              None] * x_stride_l + offs_d[
                                                                  None, :] * x_stride_d
    x = tl.load(X + x_offs, mask=offs_l[:, None] < L)

    num = min(BLOCK_L, L - pid_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / num

    y_offs = pid_b * y_stride_b + pid_h * y_stride_h + pid_l * y_stride_l + offs_d * y_stride_d
    tl.store(Y + y_offs, x_mean.to(Y.dtype.element_ty))


def mean_pool(x, WINDOW_SIZE):
    assert WINDOW_SIZE == triton.next_power_of_2(WINDOW_SIZE)

    B, H, L, D = x.shape
    L_BLOCKS = _ceil_div(L, WINDOW_SIZE)
    y = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, H, B)
    BLOCK_D = triton.next_power_of_2(D)
    _triton_mean_pool[grid](x,
                            y,
                            B,
                            H,
                            L,
                            D,
                            x.stride(0),
                            x.stride(1),
                            x.stride(2),
                            x.stride(3),
                            y.stride(0),
                            y.stride(1),
                            y.stride(2),
                            y.stride(3),
                            BLOCK_L=WINDOW_SIZE,
                            BLOCK_D=BLOCK_D)
    return y


def get_sparse_map(q, k, topk_ratio, BLOCK_Q, BLOCK_K):
    # Smooth
    smooth_k = k - torch.mean(k, dim=-2, keepdim=True)
    prediction = mean_pool(q, BLOCK_Q) @ mean_pool(smooth_k,
                                                   BLOCK_K).transpose(-1, -2)

    topk = int(topk_ratio * prediction.shape[-1])
    # Sparse attention loopup table
    sa_lut = torch.topk(prediction, topk, dim=-1, sorted=False).indices

    sparse_mask = torch.zeros_like(prediction, dtype=torch.int8)
    sparse_mask.scatter_(-1, sa_lut, 1)
    return sparse_mask, sa_lut


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
def _triton_attn_fwd(
    q_ptr,  # Q tensor
    q_stride_b,
    q_stride_h,
    q_stride_l,
    q_stride_d,
    k_ptr,  # K tensor
    k_stride_b,
    k_stride_h,
    k_stride_l,
    k_stride_d,
    v_ptr,  # V tensor
    v_stride_b,
    v_stride_h,
    v_stride_l,
    v_stride_d,
    sa_lut_ptr,  # Sparse attention loopup table tensor
    sa_lut_stride_b,
    sa_lut_stride_h,
    sa_lut_stride_num_q,
    sa_lut_stride_topk,
    o_ptr,  # Output tensor
    o_stride_b,
    o_stride_h,
    o_stride_l,
    o_stride_d,
    lse_ptr,  # LogSumExp tensor
    lse_stride_b,
    lse_stride_h,
    lse_stride_l,
    Lqo,
    Lkv,
    sm_scale,  # Softmax scale
    TOP_K: tl.constexpr,
    BLOCK_D_QK: tl.constexpr,
    BLOCK_D_VO: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_l = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    # TODO(bug): triton has bug with the flowing code
    # offs_qo_l = (pid_l * BLOCK_Q + tl.arange(0, BLOCK_Q)) % Lqo
    offs_qo_l = pid_l * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_kv_l = tl.arange(0, BLOCK_KV)
    offs_qk_d = tl.arange(0, BLOCK_D_QK)
    offs_vo_d = tl.arange(0, BLOCK_D_VO)

    q_ptrs = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h + offs_qo_l[:, None] * q_stride_l + offs_qk_d[
        None, :] * q_stride_d
    k_ptrs = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h + offs_kv_l[
        None, :] * k_stride_l + offs_qk_d[:, None] * k_stride_d
    v_ptrs = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h + offs_kv_l[:, None] * v_stride_l + offs_vo_d[
        None, :] * v_stride_d
    o_ptrs = o_ptr + pid_b * o_stride_b + pid_h * o_stride_h + offs_qo_l[:, None] * o_stride_l + offs_vo_d[
        None, :] * o_stride_d
    lse_ptrs = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h + offs_qo_l * lse_stride_l

    # Init m/l/o
    # Max
    m = tl.full([BLOCK_Q], -float('inf'), dtype=tl.float32)
    # SumExp
    se = tl.zeros([BLOCK_Q], dtype=tl.float32)
    # Attention out
    o = tl.zeros([BLOCK_Q, BLOCK_D_VO], dtype=tl.float32)

    # TODO(mask D)
    # Pre-load Q
    q = tl.load(q_ptrs, mask=offs_qo_l[:, None] < Lqo)

    # main loop
    for block_idx in tl.range(TOP_K):
        kv_index = tl.load(sa_lut_ptr + pid_b * sa_lut_stride_b +
                           pid_h * sa_lut_stride_h +
                           pid_l * sa_lut_stride_num_q +
                           block_idx * sa_lut_stride_topk)
        kv_mask = offs_kv_l < Lkv - kv_index * BLOCK_KV

        # Load K
        k = tl.load(k_ptrs + kv_index * BLOCK_KV * k_stride_l,
                    mask=kv_mask[None, :])

        # s = q @ k
        s = tl.dot(q, k) * (sm_scale * 1.4426950408889634)

        # Mask to -inf, so that softmax will zero out these positions
        if Lkv - kv_index * BLOCK_KV < BLOCK_KV:
            s = tl.where(kv_mask[None, :], s, float("-inf"))

        # Load V
        v = tl.load(v_ptrs + kv_index * BLOCK_KV * v_stride_l,
                    mask=kv_mask[:, None])

        # block max
        m_block = tl.max(s, 1)
        m_new = tl.maximum(m, m_block)

        # softmax
        p = tl.math.exp2(s - m_new[:, None])

        # block sumexp
        se_block = tl.sum(p, 1)

        # update m/l/o
        scale = tl.math.exp2(m - m_new)
        o = o * scale[:, None] + tl.dot(p.to(v.dtype), v)
        se = se * scale + se_block
        m = m_new

    # epilogue
    o = o * (1.0 / se[:, None])
    tl.store(o_ptrs,
             o.to(o_ptr.type.element_ty),
             mask=offs_qo_l[:, None] < Lqo)

    # lse = m + log(se)
    tl.store(lse_ptrs, m + tl.math.log2(se), mask=offs_qo_l < Lqo)


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
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)
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
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
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
                sparse_mask,
                lut,
                BLOCK_Q,
                BLOCK_KV,
                sm_scale=None):
        assert BLOCK_Q == 64 or BLOCK_Q == 128
        assert BLOCK_KV == 64

        B, H, Lqo, Dqk = q.shape
        _, _, Lkv, Dvo = v.shape

        # Scale for Q @ K^T
        if sm_scale is None:
            sm_scale = Dqk**-0.5

        # Attention output
        o = torch.empty([B, H, Lqo, Dvo], device=q.device, dtype=q.dtype)
        # LogSumExp for backward with shape [B, H, Lqo]
        lse = torch.empty([B, H, Lqo], device=q.device, dtype=torch.float32)

        grid = (_ceil_div(Lqo, BLOCK_Q), H, B)
        BLOCK_D_QK = triton.next_power_of_2(Dqk)
        BLOCK_D_VO = triton.next_power_of_2(Dvo)
        _triton_attn_fwd[grid](q,
                               q.stride(0),
                               q.stride(1),
                               q.stride(2),
                               q.stride(3),
                               k,
                               k.stride(0),
                               k.stride(1),
                               k.stride(2),
                               k.stride(3),
                               v,
                               v.stride(0),
                               v.stride(1),
                               v.stride(2),
                               v.stride(3),
                               lut,
                               lut.stride(0),
                               lut.stride(1),
                               lut.stride(2),
                               lut.stride(3),
                               o,
                               o.stride(0),
                               o.stride(1),
                               o.stride(2),
                               o.stride(3),
                               lse,
                               lse.stride(0),
                               lse.stride(1),
                               lse.stride(2),
                               Lqo,
                               Lkv,
                               sm_scale,
                               TOP_K=lut.shape[-1],
                               BLOCK_D_QK=BLOCK_D_QK,
                               BLOCK_D_VO=BLOCK_D_VO,
                               BLOCK_Q=BLOCK_Q,
                               BLOCK_KV=BLOCK_KV,
                               num_warps=4 if BLOCK_D_QK == 64 else 8,
                               num_stages=3)

        ctx.save_for_backward(q, k, v, sparse_mask, lut, lse, o)
        ctx.sm_scale = sm_scale
        ctx.topk = lut.shape[-1]
        ctx.BLOCK_M = BLOCK_Q
        ctx.BLOCK_N = BLOCK_KV
        return o

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, sparse_mask, lut, lse, o_s = ctx.saved_tensors
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
                           ctx.sm_scale,
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
                             ctx.sm_scale,
                             sparse_mask,
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
            topk_ratio: ratio of keys selected for sparse attention, shared across all queries. Must be in (0, 1].
            kernel_type: kernel type for linear attention, one of ['elu', 'relu', 'softmax'].
            BLOCK_Q: block size for query.
            BLOCK_KV: block size for key and value.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
        '''

        assert topk_ratio > 0 and topk_ratio <= 1, f'Invalid topk_ratio {topk_ratio}.'
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

    def forward(self, q, k, v):
        '''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
        '''
        dtype = q.dtype

        # Preprocess q/k/v
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Sparse prediction
        # Sparse mask, sparse lookup table
        sparse_mask, sa_lut = get_sparse_map(q, k, self.topk_ratio,
                                             self.BLOCK_Q, self.BLOCK_KV)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        o_s = _attention.apply(q, k, v, sparse_mask, sa_lut, self.BLOCK_Q,
                               self.BLOCK_KV)

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

        return o
