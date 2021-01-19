import logging

import numpy as np
from pytest import importorskip

cu = importorskip("cuvec")
improc = importorskip("niftypet.nimpa.prc.improc")


def test_convolve():
    src_np = np.zeros((64, 64, 64), dtype='float32')
    src_np[32, 32, 32] = 1

    knl = cu.zeros((3, 17), dtype='float32')
    knl[:, 17//2 - 1] = 0.25
    knl[:, 17 // 2] = 0.5
    knl[:, 17//2 + 1] = 0.25

    src = cu.asarray(src_np) # memcopy to cudaMallocManaged
    assert knl.dtype == src.dtype

    # won't memcopy since already cudaMallocManaged
    dst = cu.asarray(improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG))

    def check_conv():
        assert dst[32, 32, 32] == 1 / 8
        assert dst[31, 32, 32] == 1 / 16
        assert dst[31, 32, 31] == 1 / 32
        assert dst[31, 31, 31] == 1 / 64

    check_conv()

    # in-place
    dst[:] = 0
    assert not dst.any()
    improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG, output=dst.cuvec, memset=False)
    check_conv()

    # in-place with auto memset
    dst[32, 32, 32] = 0
    assert dst[32, 32, 32] == 0
    improc.convolve(src.cuvec, knl.cuvec, log=logging.DEBUG, output=dst.cuvec)
    check_conv()
