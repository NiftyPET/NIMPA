import logging

import numpy as np
from pytest import fixture, importorskip, mark

from niftypet.nimpa.prc import prc

cu = importorskip("cuvec")
improc = importorskip("niftypet.nimpa.prc.improc")


def check_conv(src, dst):
    err = "\nsrc:\n%s\ndst:\n%s" % (src[2, :3, :3], dst[2, :3, :3])
    assert dst[2, 2, 2] == 1 / 8, err
    assert dst[1, 2, 2] == 1 / 16, err
    assert dst[1, 2, 1] == 1 / 32, err
    assert dst[1, 1, 1] == 1 / 64, err


def rmse(x, y):
    return (((x - y)**2).mean() / (y**2).mean())**0.5


@fixture(scope="module")
def knl():
    res = cu.zeros((3, 17), dtype='float32')
    res[:, 17//2 - 1] = 0.25
    res[:, 17 // 2] = 0.5
    res[:, 17//2 + 1] = 0.25
    return res


def test_convolve_unpadded(knl):
    src = cu.zeros((64, 64, 64), dtype='float32')
    src[2, 2, 2] = 1

    # won't memcopy since already cudaMallocManaged
    dst = cu.asarray(improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG))
    check_conv(src, dst)


def test_convolve_autopad(knl):
    src_np = np.zeros((27, 13, 7), dtype='float32')
    src_np[2, 2, 2] = 1

    src = cu.asarray(src_np) # memcopy to cudaMallocManaged
    assert knl.dtype == src.dtype

    # won't memcopy since already cudaMallocManaged
    dst = cu.asarray(improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG))
    check_conv(src, dst)

    # in-place
    dst[:] = 0
    assert not dst.any()
    improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG, output=dst.cuvec, memset=False)
    check_conv(src, dst)

    # in-place with auto memset
    dst[2, 2, 2] = 0
    assert dst[2, 2, 2] == 0
    improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG, output=dst.cuvec)
    check_conv(src, dst)


@mark.parametrize("knl_size", [(3, 17), (2, 17), (3, 5), (2, 5), (1, 3)])
def test_conv_separable(knl_size):
    knl = np.random.random(knl_size)
    src = np.random.random((64,) * knl_size[0]).astype('float32')
    dst_gpu = prc.conv_separable(src, knl)
    dst_cpu = prc.conv_separable(src, knl, dev_id=False)
    assert hasattr(dst_gpu, 'cuvec') or knl_size[0] != 3
    assert not hasattr(dst_cpu, 'cuvec')
    assert rmse(dst_gpu, dst_cpu) < 1e-7


@mark.parametrize("half_width", [0, 1, 2])
@mark.parametrize("sigma", [0.5, 1, 2])
@mark.parametrize("width", [10, 125])
def test_nlm(half_width, sigma, width):
    src = np.random.random((width,) * 3).astype('float32')
    ref = np.random.random((width,) * 3)
    dst_gpu = prc.nlm(src, ref, half_width=half_width, sigma=sigma)
    assert hasattr(dst_gpu, 'cuvec')
    assert (dst_gpu - src).mean() < 1e-2


if __name__ == "__main__":
    from textwrap import dedent

    from argopt import argopt
    from tqdm import trange
    logging.basicConfig(level=logging.WARNING)
    args = argopt(
        dedent("""\
    Performance testing `conv_separable()`
    Usage:
        test_improc [options] [<repeats>]

    Options:
        -d DIMS  : Up to [default: 3:int]
        -k WIDTH  : Up to [default: 17:int]
        -i WIDTH  : input width [default: 234:int]

    Arguments:
        <repeats>  : [default: 10:int]
    """)).parse_args()
    assert 0 < args.d <= 3
    assert 0 < args.k <= 17
    KNL = cu.asarray(np.random.random((args.d, args.k)), dtype='float32')
    SRC = cu.asarray(np.random.random((args.i,) * args.d), dtype='float32')
    for _ in trange(args.repeats, unit="repeats", desc=f"{args.i}^{args.d} (*) {args.k}^{args.d}"):
        dst_gpu = prc.conv_separable(SRC, KNL)
    assert hasattr(dst_gpu, 'cuvec') or args.d < 3
