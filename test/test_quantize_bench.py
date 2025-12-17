import torch
import torch.nn.functional as F
import random
from functools import reduce
import operator

from ac4k_kernel.ops import nvfp4_quant, quantize
from utils import get_global_scale


def get_ref(input, dim, swizzle):
    origin_shape = input.shape
    origin_quantize_dim = input.shape[dim]
    origin_non_quantize_dim = reduce(operator.mul, input.shape,
                                     1) // origin_quantize_dim

    def ceil_div(x, y):
        return (x + y - 1) // y

    if dim < 0:
        dim = input.ndim + dim

    rank = input.ndim
    if dim != rank - 1:
        # transpose
        transpose = []
        for i in range(rank):
            if i != dim:
                transpose.append(i)
        transpose.append(dim)
        input = torch.permute(input, transpose).contiguous()

    # reshape
    input = input.reshape(-1, input.shape[-1])

    # pad to 64
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
    output_shape = []
    for i in range(len(origin_shape)):
        if i != dim:
            output_shape.append(origin_shape[i])
    output_shape.append(output.shape[-1])
    output = output.reshape(*output_shape)

    # reshape sf
    sf = sf.reshape(ceil_div(origin_non_quantize_dim, 128),
                    ceil_div(origin_quantize_dim, 64), 32, 4, 4)
    sf = torch.permute(sf, (1, 0, 3, 2, 4)).contiguous()
    sf = sf.reshape(ceil_div(origin_quantize_dim, 64), -1, 4)
    sf = sf[:, :origin_non_quantize_dim, :].contiguous()

    return output, sf


def test_quantize_bench(shape, dim, swizzle=False):
    print("test quantize dim={}, shape={}, swize={}".format(
        dim, shape, swizzle))
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    output, sf, _ = quantize(input, dim=dim, swizzle=swizzle)

    output_ref, sf_ref = get_ref(input, dim, swizzle)

    torch.testing.assert_close(output, output_ref)
    torch.testing.assert_close(sf.view(torch.uint8), sf_ref.view(torch.uint8))
    print("\ttest pass")


def test_random_bennch(num):
    for _ in range(num):
        rank = random.randint(2, 4)
        shape = [random.randint(1, 512) for _ in range(rank)]
        swizzle = random.choice([False, True])

        # avoid too large input
        while reduce(operator.mul, shape, 1) > 2 * 1024 * 1024 * 1024:
            for i in range(rank):
                if shape[i] > 1:
                    shape[i] = shape[i] // 2
                    break

        dim = random.randint(-rank, rank - 1)
        test_quantize_bench(shape, dim, swizzle)


if __name__ == "__main__":
    torch.manual_seed(9567)
    random.seed(9527)
    test_quantize_bench((1024, 1024), -1)
    test_quantize_bench((1024, 1024), 0)
    test_quantize_bench((128, 512), 1)
    test_random_bennch(345)
