import logging

import numpy as np
from pytest import fixture, importorskip

cu = importorskip("cuvec")
improc = importorskip("niftypet.nimpa.prc.improc")


def check_conv(src, dst):
    err = "\nsrc:\n%s\ndst:\n%s" % (src[2, :3, :3], dst[2, :3, :3])
    assert dst[2, 2, 2] == 1 / 8, err
    assert dst[1, 2, 2] == 1 / 16, err
    assert dst[1, 2, 1] == 1 / 32, err
    assert dst[1, 1, 1] == 1 / 64, err


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
