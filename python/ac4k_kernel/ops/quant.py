import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_cuda_nvfp4_quant():
    try:
        from ._cuda_ops import nvfp4_quant_sm120
        return nvfp4_quant_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_quant_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def nvfp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in a sizzled layout.
    """

    _nvfp4_quant_sm120 = _load_cuda_nvfp4_quant()

    m, n = input.shape
    block_size = 16
    device = input.device

    # Two nvfp4 elements will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    # rounded_m = ((m + 128 - 1) // 128) * 128
    # scale_n = n // block_size
    # rounded_n = ((scale_n + 4 - 1) // 4) * 4
    output_sf = torch.zeros(
        (((m + 128 - 1) // 128) * 128, (n // block_size + 4 - 1) // 4),
        device=device,
        dtype=torch.int32)

    _nvfp4_quant_sm120(output, output_sf, input, input_global_scale)
    output_sf = output_sf.view(torch.float8_e4m3fn)
    return output, output_sf


@lru_cache(maxsize=1)
def _load_cuda_quantize():
    try:
        from ._cuda_ops import quantize_sm120
        return quantize_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'quantize_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def quantize(input: torch.Tensor,
             cross_dim,
             reduce_dim,
             swizzle=False,
             output=None,
             sf=None):
    """
    Quantize input tensor from bfloat16 to nvfp4.
    Args:
        input: The input tensor to be quantized to nvfp4.
        cross_dim: M or N dimension in the input tensor.
        reduce_dim: K dimension in the input tensor.
        swizzle: Whether to swizzle the output tensor.
        output: The output tensor to store the quantized tensor.
        sf: The output tensor to store the scaling factors.
    """

    assert input.ndim >= 2 and input.ndim <= 4, "input tensor must be 2D, 3D or 4D"

    BLOCK_SIZE = 16
    NVFP4_ELES_PER_BYTE = 2
    PACK_SF = 4
    CROSS_DIM_ALIGN_SIZE = 16
    REDUCE_DIM_ALIGN_SIZE = BLOCK_SIZE * PACK_SF

    quantize_sm120 = _load_cuda_quantize()

    def ceil_div(x, y):
        return (x + y - 1) // y

    def align_up(x, y):
        return ceil_div(x, y) * y

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    # alpha = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / (torch.amax(
    #     torch.abs(input.view(-1))).to(torch.float32))
    alpha = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
             torch.amax(torch.abs(input.view(-1)), dim=-1)).to(torch.float32)
    global_scale = 1 / alpha

    # refine dim
    if cross_dim < 0:
        cross_dim += input.ndim
    if reduce_dim < 0:
        reduce_dim += input.ndim
    assert cross_dim != reduce_dim, "cross_dim and reduce_dim must be different"

    # shape inference for out and sf
    input_shape = input.shape
    # out
    # out layout: [xx, xx, cross_dim_align, reduce_dim_align / 2] x u8
    output_shape = []
    for i in range(len(input_shape)):
        if i != cross_dim and i != reduce_dim:
            output_shape.append(input_shape[i])
    output_shape.append(input_shape[cross_dim])
    output_shape.append(
        align_up(input_shape[reduce_dim], REDUCE_DIM_ALIGN_SIZE) //
        NVFP4_ELES_PER_BYTE)
    # sf
    # sf layout: [xx, xx, reduce_dim_align / 64, cross_dim_align, 4] x f8
    sf_shape = []
    for i in range(len(input_shape)):
        if i != cross_dim and i != reduce_dim:
            sf_shape.append(input_shape[i])
    sf_shape.append(
        align_up(input_shape[reduce_dim], REDUCE_DIM_ALIGN_SIZE) //
        (BLOCK_SIZE * PACK_SF))
    sf_shape.append(align_up(input_shape[cross_dim], CROSS_DIM_ALIGN_SIZE))
    sf_shape.append(4)

    # alloc buffer for output and sf
    if output is None:
        output = torch.empty(output_shape,
                             dtype=torch.uint8,
                             device=input.device)
    if sf is None:
        # FIXME: fix zero memset
        sf = torch.zeros(sf_shape,
                         dtype=torch.float8_e4m3fn,
                         device=input.device)

    quantize_sm120(output, sf, input, alpha, cross_dim, reduce_dim, swizzle)

    return output, sf, global_scale
