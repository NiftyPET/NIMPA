import logging

import numpy as np
from pytest import fixture, importorskip, mark

from niftypet.nimpa.prc.prc import conv_separable

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
    dst_gpu = conv_separable(src, knl)
    dst_cpu = conv_separable(src, knl, dev_id=False)
    assert hasattr(dst_gpu, 'cuvec') or knl_size[0] != 3
    assert not hasattr(dst_cpu, 'cuvec')
    assert rmse(dst_gpu, dst_cpu) < 1e-7
