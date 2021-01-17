import logging

import numpy as np
from pytest import importorskip

cuvec = importorskip("cuvec")
improc = importorskip("niftypet.nimpa.prc.improc")


def test_convolve():
    src = cuvec.zeros((64, 64, 64))
    knl = cuvec.zeros((3, 17))

    src_arr = np.asarray(src)
    knl_arr = np.asarray(knl)
    src_arr[32, 32, 32] = 1
    knl_arr[:, 17//2 - 1] = 0.25
    knl_arr[:, 17 // 2] = 0.5
    knl_arr[:, 17//2 + 1] = 0.25

    dst = improc.convolve(src, knl, log=logging.DEBUG)
    dst_arr = np.asarray(dst)

    def check_conv():
        assert dst_arr[32, 32, 32] == 1 / 8
        assert dst_arr[31, 32, 32] == 1 / 16
        assert dst_arr[31, 32, 31] == 1 / 32
        assert dst_arr[31, 31, 31] == 1 / 64

    check_conv()

    # in-place
    dst_arr[:] = 0
    assert not dst_arr.any()
    improc.convolve(src, knl, log=logging.DEBUG, output=dst, memset=False)
    check_conv()

    # in-place with auto memset
    dst_arr[32, 32, 32] = 0
    assert dst_arr[32, 32, 32] == 0
    improc.convolve(src, knl, log=logging.DEBUG, output=dst)
    check_conv()
