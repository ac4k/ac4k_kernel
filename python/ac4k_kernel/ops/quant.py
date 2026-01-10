import torch
from functools import lru_cache
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _triton_global_scale_kernel(x_ptr, out_ptr, N, max_scale,
                                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    abs_x = tl.abs(x)
    block_max = tl.max(abs_x, 0).to(tl.float32)
    out_ptrs = out_ptr + tl.zeros((BLOCK_SIZE, ), dtype=tl.int32)
    tl.atomic_max(out_ptrs,
                  block_max / max_scale,
                  mask=(tl.arange(0, BLOCK_SIZE) == 0))


def _global_scale(x: torch.Tensor, max_scale: float) -> torch.Tensor:
    x_flat = x.flatten()
    n_elements = x_flat.numel()

    def grid(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )

    out = torch.zeros((), dtype=torch.float32, device="cuda")
    _triton_global_scale_kernel[grid](x_flat, out, n_elements, max_scale)

    return out


@lru_cache(maxsize=1)
def _load_cuda_nvfp4_quantize():
    try:
        from ._cuda_ops import nvfp4_quantize_sm120
        return nvfp4_quantize_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_quantize_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


@lru_cache(maxsize=1)
def _load_cuda_fp8_quantize():
    try:
        from ._cuda_ops import fp8_quantize_sm120
        return fp8_quantize_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'fp8_quantize_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


@lru_cache(maxsize=1)
def _load_cuda_i8_quantize():
    try:
        from ._cuda_ops import int8_quantize_sm120
        return int8_quantize_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'int8_quantize_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


class Ac4kQuantizeOp(torch.nn.Module):

    def __init__(self):
        super(Ac4kQuantizeOp, self).__init__()

    @staticmethod
    def ceil_div(x, y):
        return (x + y - 1) // y

    @staticmethod
    def align_up(x, y):
        return Ac4kQuantizeOp.ceil_div(x, y) * y

    def forward(self,
                input: torch.Tensor,
                cross_dim,
                reduce_dim,
                max_scale=None,
                precision="nvfp4",
                swizzle=False,
                output=None,
                sf=None):
        assert input.ndim >= 2 and input.ndim <= 4, "input tensor must be 2D, 3D or 4D"
        assert precision in ["nvfp4", "fp8e4m3", "int8"]

        # Refine dim to positive
        if cross_dim < 0:
            cross_dim += input.ndim
        if reduce_dim < 0:
            reduce_dim += input.ndim
        assert cross_dim != reduce_dim, "cross_dim and reduce_dim must be different"

        # Quantize to fp8e4m3
        if precision == "fp8e4m3":
            quantize_kernel = _load_cuda_fp8_quantize()

            CROSS_DIM_ALIGN_SIZE = 16
            REDUCE_DIM_ALIGN_SIZE = 16
            MAX_SCALE = 448

            # shape inference for out and sf
            input_shape = input.shape
            # out
            # out layout: [xx, xx, cross_dim, reduce_dim_align] x f8
            output_shape = []
            for i in range(len(input_shape)):
                if i != cross_dim and i != reduce_dim:
                    output_shape.append(input_shape[i])
            output_shape.append(input_shape[cross_dim])
            output_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[reduce_dim],
                                        REDUCE_DIM_ALIGN_SIZE))
            # sf
            # sf layout: [xx, xx, cross_dim_align] x f32
            sf_shape = []
            for i in range(len(input_shape)):
                if i != cross_dim and i != reduce_dim:
                    sf_shape.append(input_shape[i])
            sf_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[cross_dim],
                                        CROSS_DIM_ALIGN_SIZE))

            # alloc buffer for output and sf
            if output is None:
                output = torch.empty(output_shape,
                                     dtype=torch.float8_e4m3fn,
                                     device=input.device)
            if sf is None:
                sf = torch.empty(sf_shape,
                                 dtype=torch.float32,
                                 device=input.device)

            if max_scale is None:
                max_scale = MAX_SCALE
            max_scale = torch.full([],
                                   max_scale,
                                   dtype=torch.float32,
                                   device=input.device)
            quantize_kernel(output, sf, input, max_scale, cross_dim,
                            reduce_dim, swizzle)

            return output, sf
        elif precision == "int8":
            quantize_kernel = _load_cuda_i8_quantize()

            CROSS_DIM_ALIGN_SIZE = 16
            REDUCE_DIM_ALIGN_SIZE = 16
            MAX_SCALE = 127

            # shape inference for out and sf
            input_shape = input.shape
            # out
            # out layout: [xx, xx, cross_dim, reduce_dim_align] x int8
            output_shape = []
            for i in range(len(input_shape)):
                if i != cross_dim and i != reduce_dim:
                    output_shape.append(input_shape[i])
            output_shape.append(input_shape[cross_dim])
            output_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[reduce_dim],
                                        REDUCE_DIM_ALIGN_SIZE))
            # sf
            # sf layout: [xx, xx, cross_dim_align] x f32
            sf_shape = []
            for i in range(len(input_shape)):
                if i != cross_dim and i != reduce_dim:
                    sf_shape.append(input_shape[i])
            sf_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[cross_dim],
                                        CROSS_DIM_ALIGN_SIZE))

            # alloc buffer for output and sf
            if output is None:
                output = torch.empty(output_shape,
                                     dtype=torch.int8,
                                     device=input.device)
            if sf is None:
                sf = torch.empty(sf_shape,
                                 dtype=torch.float32,
                                 device=input.device)

            if max_scale is None:
                max_scale = MAX_SCALE
            max_scale = torch.full([],
                                   max_scale,
                                   dtype=torch.int32,
                                   device=input.device)
            quantize_kernel(output, sf, input, max_scale, cross_dim,
                            reduce_dim)

            return output, sf
        else:
            quantize_kernel = _load_cuda_nvfp4_quantize()

            BLOCK_SIZE = 16
            NVFP4_ELES_PER_BYTE = 2
            PACK_SF = 4
            CROSS_DIM_ALIGN_SIZE = 16
            REDUCE_DIM_ALIGN_SIZE = BLOCK_SIZE * PACK_SF
            FLOAT4_E2M1_MAX = 6.0
            FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

            if max_scale is None:
                max_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX
            global_scale = _global_scale(input, float(max_scale))

            # shape inference for out and sf
            input_shape = input.shape
            # out
            # out layout: [xx, xx, cross_dim, reduce_dim_align / 2] x u8
            output_shape = []
            for i in range(len(input_shape)):
                if i != cross_dim and i != reduce_dim:
                    output_shape.append(input_shape[i])
            output_shape.append(input_shape[cross_dim])
            output_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[reduce_dim],
                                        REDUCE_DIM_ALIGN_SIZE) //
                NVFP4_ELES_PER_BYTE)
            # sf
            # sf layout: [xx, xx, reduce_dim_align / 64, cross_dim_align, 4] x f8
            sf_shape = []
            for i in range(len(input_shape)):
                if i != cross_dim and i != reduce_dim:
                    sf_shape.append(input_shape[i])
            sf_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[reduce_dim],
                                        REDUCE_DIM_ALIGN_SIZE) //
                (BLOCK_SIZE * PACK_SF))
            sf_shape.append(
                Ac4kQuantizeOp.align_up(input_shape[cross_dim],
                                        CROSS_DIM_ALIGN_SIZE))
            sf_shape.append(4)

            # alloc buffer for output and sf
            if output is None:
                output = torch.empty(output_shape,
                                     dtype=torch.uint8,
                                     device=input.device)
            if sf is None:
                sf = torch.empty(sf_shape,
                                 dtype=torch.float8_e4m3fn,
                                 device=input.device)

            # quantize_kernel(output, sf, input, alpha, cross_dim, reduce_dim,
            #                 swizzle)
            quantize_kernel(output, sf, input, global_scale, cross_dim,
                            reduce_dim, swizzle)

            return output, sf, global_scale


def quantize(input,
             cross_dim,
             reduce_dim,
             max_scale=None,
             precision="nvfp4",
             swizzle=False,
             output=None,
             sf=None):
    """
    Quantize input tensor from bfloat16 to fp8e4m3, nvfp4 or int8.

    Arguments:
        input: The input tensor need to be quantized. Only support input is bf16 dtype now.
        cross_dim: M or N dimension(parallel dim) in the input tensor.
        reduce_dim: K dimension(reduction dim) in the input tensor.
        max_scale: The max scale for quantization. Default is None. If None, the max_scale will be 127 for int8, 6 * 448 for nvfp4 and 448 for fp8e4m3.
        precision: The precision of the output tensor. Support "nvfp4", "fp8e4m3" and "int8".
        swizzle: Whether to swizzle the reduce dim, only used when precision is nvfp4 and fp8e4m3. Default is False. If True, the reduce dimension whill permute with shape [-1, 4, 4, 2] and [0, 2, 1, 3] layout for nvfp4. If True, the reduce dimension whill permute with shape [-1, 2, 4, 2] and [0, 2, 1, 3] layout for fp8e4m3.
        output: The output tensor to store the quantized tensor:
        sf: The output tensor to store the scaling factors:

    Returns:
        output: The quantized tensor.
            If quantize to nvfp4:
                output shape is: [xx, xx, corss_dim, align_up(reduce_dim, 64) // 2] x u8
            If quantize to fp8e4m3:
                output shape is: [xx, xx, corss_dim, align_up(reduce_dim, 16)] x fp8e4m3
            If quantize to int8:
                output shape is: [xx, xx, corss_dim, align_up(reduce_dim, 16)] x int8
            xx mean the other dimension in the input tensor.
        sf: The scaling factors.
            If quantize to nvfp4:
                sf shape is: [xx, xx, ceil_div(reduce_dim, 64), align_up(cross_dim, 16), 4] x fp8e4m3
            If quantize to fp8e4m3:
                sf shape is: [xx, xx, align_up(cross_dim, 16)] x fp32
            If quantize to int8:
                sf shape is: [xx, xx, align_up(cross_dim, 16)] x fp32
            xx mean the other dimension in the input tensor.
        global_scale(option): The global scale for quantization when precision is nvfp4.
    """

    op = Ac4kQuantizeOp()
    return op(input,
              cross_dim,
              reduce_dim,
              max_scale=max_scale,
              precision=precision,
              swizzle=swizzle,
              output=output,
              sf=sf)
