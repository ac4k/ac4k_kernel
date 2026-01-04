import torch
import torch.nn.functional as F
import random
from functools import reduce
import operator

from ac4k_kernel.ops import quantize


def ceil_div(x, y):
    return (x + y - 1) // y


def align_up(x, y):
    return ceil_div(x, y) * y


def get_ref(input, cross_dim, reduce_dim, swizzle):
    global_scale_max = torch.amax(torch.abs(input)).to(
        torch.float32) / (6 * 448)

    if reduce_dim < 0:
        reduce_dim = input.ndim + reduce_dim
    if cross_dim < 0:
        cross_dim = input.ndim + cross_dim

    origin_cross_dim_size = input.shape[cross_dim]
    origin_reduce_dim_size = input.shape[reduce_dim]
    origin_cross_dim_size_paded = align_up(origin_cross_dim_size, 16)
    origin_reduce_dim_size_paded = align_up(origin_reduce_dim_size, 64)
    origin_input_shape = input.shape

    # transpsoe to [xx, xx, cross_dim, reduce_dim]
    transpose = []
    for i in range(input.ndim):
        if i != reduce_dim and i != cross_dim:
            transpose.append(i)
    transpose.append(cross_dim)
    transpose.append(reduce_dim)
    input = torch.permute(input, transpose).contiguous()
    # reshape
    input = input.reshape(-1, origin_cross_dim_size, origin_reduce_dim_size)

    # pad
    # pad cross dim
    if origin_cross_dim_size_paded != origin_cross_dim_size:
        pad = [
            0, 0, 0, origin_cross_dim_size_paded - origin_cross_dim_size, 0, 0
        ]
        input = F.pad(input, pad, value=0).contiguous()
    # pad reduce dim
    if origin_reduce_dim_size_paded != origin_reduce_dim_size:
        pad = [
            0, origin_reduce_dim_size_paded - origin_reduce_dim_size, 0, 0, 0,
            0
        ]
        input = F.pad(input, pad, value=0).contiguous()

    # swizzle
    if swizzle:
        input = input.reshape(-1, 4, 4, 2)
        input = torch.permute(input, (0, 2, 1, 3)).contiguous()
        input = input.reshape(-1, origin_cross_dim_size_paded,
                              origin_reduce_dim_size_paded)

    sf = torch.empty((input.shape[0], origin_reduce_dim_size_paded // 64,
                      origin_cross_dim_size_paded, 4),
                     dtype=torch.float8_e4m3fn,
                     device="cuda")
    out = torch.empty(input.shape, dtype=torch.float32, device="cuda")
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(0, input.shape[2], 16):
                vec = input[i, j, k:k + 16].to(torch.float32)
                abs_max = torch.amax(torch.abs(vec))
                if abs_max == 0:
                    sf[i, k // 64, j, (k % 64) // 16] = 0
                    out[i, j, k:k + 16] = 0
                else:
                    scale_factor = (abs_max / global_scale_max / 6)
                    sf[i, k // 64, j, (k % 64) // 16] = scale_factor
                    vec = vec / global_scale_max / scale_factor
                    out[i, j, k:k + 16] = vec
    # input = input.reshape(-1, 16)
    # abs_max = torch.amax(torch.abs(input), dim=-1, keepdim=True)
    # scale_factor = (abs_max.to(torch.float32) / global_scale_max / 6)
    # # sf = scale_factor.reshape(-1, 1, 1, 4)
    # out = input.to(torch.float32) / global_scale_max / scale_factor
    # out = out.reshape(-1, origin_cross_dim_size_paded, origin_reduce_dim_size_paded)
    # sf = scale_factor.to(torch.float8_e4m3fn).reshape(-1, origin_cross_dim_size_paded, origin_reduce_dim_size_paded // 64, 4)
    # sf = sf.permute(0, 2, 1, 3).contiguous()

    # reshape out
    out_shape = []
    for i in range(len(origin_input_shape)):
        if i != cross_dim and i != reduce_dim:
            out_shape.append(origin_input_shape[i])
    out_shape.append(origin_cross_dim_size)
    out_shape.append(origin_reduce_dim_size_paded)
    out = out[:, :origin_cross_dim_size, :].reshape(out_shape)
    # reshape sf
    sf_shape = []
    for i in range(len(out_shape) - 2):
        sf_shape.append(out_shape[i])
    sf_shape.append(sf.shape[-3])
    sf_shape.append(sf.shape[-2])
    sf_shape.append(sf.shape[-1])
    sf = sf.reshape(sf_shape)
    return out, sf, global_scale_max


def break_fp4_bytes(a):
    assert a.dtype == torch.uint8

    kE2M1ToFloatArray = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]

    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    map_tensor = torch.tensor(kE2M1ToFloatArray,
                              dtype=torch.float32,
                              device=a.device)
    fH = map_tensor[highHalfByte.long()].to(a.device)
    fL = map_tensor[lowHalfByte.long()].to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC]
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def test_quantize_bench(shape, cross_dim, reduce_dim, swizzle=False):
    print(
        "test quantize cross_dim={}, reduce_dim={}, shape={}, swize={}".format(
            cross_dim, reduce_dim, shape, swizzle))

    val = [
        # 0.5,
        1.0,
        # 1.5,
        2.0,
        # 3.0,
        4.0,
        6.0,
        # -0.5,
        -1.0,
        # -1.5,
        -2.0,
        # -3.0,
        -4.0,
        -6.0,
    ]

    val_tensor = torch.tensor(val, dtype=torch.bfloat16, device="cuda")
    input = val_tensor[torch.randint(0, len(val), shape)]

    output, sf, global_scale = quantize(input,
                                        cross_dim,
                                        reduce_dim,
                                        swizzle=swizzle,
                                        precision="nvfp4")

    # post process out
    output_f32 = break_fp4_bytes(output)

    # ref
    output_ref, sf_ref, global_scale_ref = get_ref(input, cross_dim,
                                                   reduce_dim, swizzle)

    # assert output_f32.equal(output_ref)
    torch.testing.assert_close(output_f32, output_ref, atol=1e-3, rtol=1e-6)
    torch.testing.assert_close(sf.to(torch.float32),
                               sf_ref.to(torch.float32),
                               atol=1e-3,
                               rtol=1e-6)
    torch.testing.assert_close(global_scale,
                               global_scale_ref,
                               atol=1e-3,
                               rtol=1e-6)
    print("\ttest pass")


def test_random_bennch(num):
    for _ in range(num):
        rank = 2
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
    test_random_bennch(25)
