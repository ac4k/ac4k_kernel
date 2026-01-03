import torch
import torch.nn.functional as F
import random
from functools import reduce
import operator

from ac4k_kernel.ops import nvfp4_quant, quantize
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

    output, sf, _ = quantize(input, cross_dim, reduce_dim, swizzle=swizzle)
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


def test_fp8_quantize_bench(shape, cross_dim, reduce_dim, swizzle=False):
    print(
        "test quantize cross_dim={}, reduce_dim={}, shape={}, swize={}".format(
            cross_dim, reduce_dim, shape, swizzle))
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    quantize(input,
             cross_dim,
             reduce_dim,
             swizzle=swizzle,
             precision="fp8e4m3")

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


def qunatize_fp8(shape, cross_dim, reduce_dim, swizzle=False):
    print(
        "test quantize cross_dim={}, reduce_dim={}, shape={}, swize={}".format(
            cross_dim, reduce_dim, shape, swizzle))

    val = [-8, -6, -5, -4, 1, 2, 3, 4, 5, 6, 8, 9]
    val_tensor = torch.tensor(val, dtype=torch.bfloat16)
    input = val_tensor[torch.randint(0, len(val), shape)]

    size0 = 1
    for i in range(len(shape)):
        if i < reduce_dim:
            size0 *= shape[i]
    size1 = 1
    for i in range(len(shape)):
        if i > reduce_dim:
            size1 *= shape[i]
    input = input.reshape(size0, -1, size1)
    for i in range(size0):
        for j in range(size1):
            max_value = abs(random.choice(val))
            mask = (input[i, :, j] > max_value) | (input[i, :, j] < -max_value)
            input[i, :, j][mask] = max_value

    input = input.reshape(shape)
    input = input.cuda()
    output, sf = quantize(input,
                          cross_dim,
                          reduce_dim,
                          max_scale=2.25,
                          swizzle=swizzle,
                          precision="fp8e4m3")

    # ref
    # transpose
    layout = []
    for i in range(len(shape)):
        if i != cross_dim and i != reduce_dim:
            layout.append(i)
    layout.append(cross_dim)
    layout.append(reduce_dim)
    input = input.permute(*layout)
    # pad
    if input.shape[-1] % 16 != 0 or input.shape[-2] % 16 != 0:
        pad = [0, 0] * len(input.shape)
        pad_size0 = 0
        if input.shape[-1] % 16 != 0:
            pad_size0 = 16 - (input.shape[-1] % 16)
        pad_size1 = 0
        if input.shape[-2] % 16:
            pad_size1 = 16 - (input.shape[-2] % 16)
        pad[1] = pad_size0
        pad[3] = pad_size1
        input = F.pad(input, pad=pad, mode='constant', value=0).contiguous()

    # swizzle
    if swizzle:
        shape_backup = input.shape
        input = input.reshape(-1, 2, 4, 2)
        input = input.permute(0, 2, 1, 3).contiguous()
        input = input.reshape(shape_backup)

    # quantize
    # sf
    abs_max = torch.max(torch.abs(input), dim=-1,
                        keepdim=True)[0].to(torch.float32)
    sf_ref = abs_max.squeeze(-1) / 2.25

    # quantize output
    output_ref = input.to(torch.float32) / abs_max * 2.25
    output_ref = output_ref.to(torch.float8_e4m3fn)
    output_ref = output_ref.reshape(-1, output_ref.shape[-2],
                                    output_ref.shape[-1])
    output_ref = output_ref[:, :shape[cross_dim], :].reshape(
        output.shape).contiguous()

    assert sf.equal(sf_ref)
    assert output.equal(output_ref)


if __name__ == "__main__":
    torch.manual_seed(9567)
    random.seed(9527)
    qunatize_fp8((1024, 1024), 0, 1)
    qunatize_fp8((1, 75600, 40, 128), 3, 1, swizzle=True)
    qunatize_fp8((1024, 10), 0, 1)
    qunatize_fp8((16, 16), 0, 1, swizzle=True)
    qunatize_fp8((777, 879), 0, 1, swizzle=True)
    test_quantize_bench((1024, 1024), 1, 0)
    test_quantize_bench((128, 512), 0, 1)
    test_random_bennch(111)
