import torch
import torch.nn.functional as F
import random
from functools import reduce
import operator

from ac4k_kernel.ops import nvfp4_quant, nvfp4_quantize
from utils import get_global_scale


def ceil_div(x, y):
    return (x + y - 1) // y


def align_up(x, y):
    return ceil_div(x, y) * y


def get_ref(input, cross_dim, reduce_dim, swizzle):
    if reduce_dim < 0:
        reduce_dim = input.ndim + reduce_dim
    if cross_dim < 0:
        cross_dim = input.ndim + cross_dim

    origin_cross_dim_size = input.shape[cross_dim]
    origin_reduce_dim_size = input.shape[reduce_dim]
    total_size = reduce(operator.mul, input.shape, 1)

    # transpsoe to [xx, xx, cross_dim, reduce_dim]
    transpose = []
    for i in range(input.ndim):
        if i != reduce_dim and i != cross_dim:
            transpose.append(i)
    transpose.append(cross_dim)
    transpose.append(reduce_dim)
    input = torch.permute(input, transpose).contiguous()

    # reshape
    input = input.reshape(-1, input.shape[-1])

    # pad
    if input.shape[-1] % 64 != 0:
        pad = [0, 64 - (input.shape[-1] % 64), 0, 0]
        input = F.pad(input, pad, value=0).contiguous()

    quantize_dim = input.shape[-1]

    # swizzle
    # quantize dim layout
    # [-1, 4, 4, 2]
    # apply swizzle, transpose layout is [0, 2, 1, 3]
    # [-1, 4, 4, 2]
    if swizzle:
        input = input.reshape(-1, 4, 4, 2)
        input = torch.permute(input, (0, 2, 1, 3)).contiguous()
        input = input.reshape(-1, quantize_dim)

    scale = get_global_scale(input)
    # output layout
    # [non_dim, quantize_dim / 2]
    # sf layout
    # [non_dim / 128, quant_dim / 64, 32, 4, 4]xfloat8_e4m3fn
    output, sf = nvfp4_quant(input, scale)

    # reshape output
    output = output.reshape(-1, origin_cross_dim_size,
                            align_up(origin_reduce_dim_size, 64) // 2)

    # reshape sf
    sf = sf.reshape(ceil_div(total_size // origin_reduce_dim_size, 128),
                    ceil_div(origin_reduce_dim_size, 64), 32, 4, 4)
    sf = torch.permute(sf, (1, 0, 3, 2, 4)).contiguous()
    sf = sf.reshape(ceil_div(origin_reduce_dim_size, 64), -1, 4)
    sf = sf[:, :total_size // origin_reduce_dim_size, :].contiguous()
    sf = sf.reshape(sf.shape[0], -1, origin_cross_dim_size, 4)
    sf = torch.permute(sf, (1, 0, 2, 3)).contiguous()

    return output, sf


def test_quantize_bench(shape, cross_dim, reduce_dim, swizzle=False):
    print(
        "test quantize cross_dim={}, reduce_dim={}, shape={}, swize={}".format(
            cross_dim, reduce_dim, shape, swizzle))
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    origin_cross_dim_size = input.shape[cross_dim]
    origin_reduce_dim_size = input.shape[reduce_dim]

    output, sf, _ = nvfp4_quantize(input,
                                   cross_dim,
                                   reduce_dim,
                                   swizzle=swizzle)
    # reshape output
    output = output.reshape(-1, origin_cross_dim_size,
                            align_up(origin_reduce_dim_size, 64) // 2)
    # reshape sf
    sf = sf.reshape(-1, ceil_div(origin_reduce_dim_size, 64),
                    align_up(origin_cross_dim_size, 16), 4)
    sf = sf[:, :, :origin_cross_dim_size, :].contiguous()

    output_ref, sf_ref = get_ref(input, cross_dim, reduce_dim, swizzle)

    torch.testing.assert_close(output, output_ref)
    torch.testing.assert_close(sf.view(torch.uint8), sf_ref.view(torch.uint8))
    print("\ttest pass")


def test_random_bennch(num):
    for _ in range(num):
        rank = random.randint(2, 4)
        shape = [random.randint(1, 512) for _ in range(rank)]
        swizzle = random.choice([False, True])

        # avoid too large input
        while reduce(operator.mul, shape, 1) > 50 * 1024 * 1024:
            for i in range(rank):
                if shape[i] > 1:
                    shape[i] = shape[i] // 2
                    break

        cross_dim = random.randint(0, rank - 1)
        reduce_dim = random.randint(0, rank - 1)
        while cross_dim == reduce_dim:
            reduce_dim = random.randint(0, rank - 1)
            reduce_dim = random.randint(0, rank - 1)

        test_quantize_bench(shape, cross_dim, reduce_dim, swizzle)


if __name__ == "__main__":
    torch.manual_seed(9567)
    random.seed(9527)
    test_quantize_bench((1024, 1024), -2, -1)
    test_quantize_bench((1024, 1024), 1, 0)
    test_quantize_bench((128, 512), 0, 1)
    test_random_bennch(666)
