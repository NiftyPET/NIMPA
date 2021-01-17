import logging

import numpy as np
from pytest import importorskip

cuvec = importorskip("cuvec")
improc = importorskip("niftypet.nimpa.prc.improc")


def test_convolve():
    dst = cuvec.vector((64, 64, 64))
    src = cuvec.vector((64, 64, 64))
    knl = cuvec.vector((3, 17))

    dst_arr = np.asarray(dst)
    src_arr = np.asarray(src)
    knl_arr = np.asarray(knl)
    src_arr[32, 32, 32] = 1
    knl_arr[:, 17//2 - 1] = 0.25
    knl_arr[:, 17 // 2] = 0.5
    knl_arr[:, 17//2 + 1] = 0.25

    improc.convolve(dst, src=src, knl=knl, log=logging.DEBUG, memset=False)

    assert dst_arr[31, 31, 31] == 1 / 64
    assert dst_arr[31, 32, 31] == 1 / 32
    assert dst_arr[31, 32, 32] == 1 / 16
    assert dst_arr[32, 32, 32] == 1 / 8
