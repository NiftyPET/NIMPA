import logging

import cuvec as cu
import numpy as np
from pytest import fixture, importorskip, mark

from niftypet.nimpa.prc import num

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
    dst = cu.asarray(improc.convolve(src, knl, log=logging.DEBUG))
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
    improc.convolve(src, knl, log=logging.DEBUG, output=dst, memset=False)
    check_conv(src, dst)

    # in-place with auto memset
    dst[2, 2, 2] = 0
    assert dst[2, 2, 2] == 0
    improc.convolve(src, knl, log=logging.DEBUG, output=dst)
    check_conv(src, dst)


@mark.parametrize("knl_size", [(3, 17), (2, 17), (3, 5), (2, 5), (1, 3)])
def test_conv_separable(knl_size):
    knl = np.random.random(knl_size)
    src = np.random.random((64,) * knl_size[0]).astype('float32')
    dst_gpu = num.conv_separable(src, knl)
    dst_cpu = num.conv_separable(src, knl, dev_id=False)
    assert hasattr(dst_gpu, 'cuvec') or knl_size[0] != 3
    assert not hasattr(dst_cpu, 'cuvec')
    assert rmse(dst_gpu, dst_cpu) < 1e-7


@mark.parametrize("half_width", [0, 1, 2])
@mark.parametrize("sigma", [0.5, 1, 2])
@mark.parametrize("width", [10, 125])
def test_nlm(half_width, sigma, width):
    src = np.random.random((width,) * 3).astype('float32')
    ref = np.random.random((width,) * 3)
    dst_gpu = num.nlm(src, ref, half_width=half_width, sigma=sigma)
    assert hasattr(dst_gpu, 'cuvec')
    assert (dst_gpu - src).mean() < 1e-2


def test_isub():
    src = cu.asarray(np.random.random((42, 2)).astype('float32'))
    idxs = cu.asarray((np.random.random((12,)) * 42).astype('uint32'))

    ref = src[idxs]
    res = num.isub(src, idxs)
    assert (res == ref).all()

    out = cu.zeros(res.shape, res.dtype)
    res = num.isub(src, idxs, output=out)
    assert (res == out).all()


if __name__ == "__main__":
    from sys import version_info
    from textwrap import dedent

    from argopt import argopt
    from tqdm import trange
    logging.basicConfig(level=logging.WARNING)

    parser = argopt(
        dedent("""\
        Usage:
            test_improc [options]

        Options:
            -r REP, --repeats REP  : [default: 10:int]
        """))
    if version_info[:2] >= (3, 7):
        subs = parser.add_subparsers(required=True)
    else:
        subs = parser.add_subparsers()

    def sub_parser(prog=None, **kwargs):
        return subs.add_parser(prog, **kwargs)

    def conv(args):
        """\
        Performance testing `conv_separable()`
        Usage:
            conv [options]

        Options:
            -d DIMS  : Up to [default: 3:int]
            -k WIDTH  : Up to [default: 17:int]
            -i WIDTH  : input width [default: 234:int]
        """
        assert 0 < args.d <= 3
        assert 0 < args.k <= 17
        KNL = cu.asarray(np.random.random((args.d, args.k)), dtype='float32')
        SRC = cu.asarray(np.random.random((args.i,) * args.d), dtype='float32')
        for _ in trange(args.repeats, unit="repeats",
                        desc=f"{args.i}^{args.d} (*) {args.k}^{args.d}"):
            dst_gpu = num.conv_separable(SRC, KNL)
        assert hasattr(dst_gpu, 'cuvec') or args.d < 3

    argopt(dedent(conv.__doc__), argparser=sub_parser).set_defaults(func=conv)

    def nlm(args):
        """\
        Performance testing `nlm()`
        Usage:
            nlm [options]

        Options:
            -s SIGMA  : NLM parameter [default: 1:float]
            -k HALF_WIDTH  : kernel half-width [default: 2:int]
            -i WIDTH  : input width [default: 123:int]
        """
        assert 0 < args.k
        assert 0 < args.i
        IMG = cu.asarray(np.random.random((args.i,) * 3), dtype='float32')
        REF = cu.asarray(np.random.random((args.i,) * 3), dtype='float32')
        for _ in trange(args.repeats, unit="repeats", desc=f"{args.i}^3 (*) {args.k*2+1}^3"):
            dst_gpu = num.nlm(IMG, REF, sigma=args.s, half_width=args.k)
        assert hasattr(dst_gpu, 'cuvec')

    argopt(dedent(nlm.__doc__), argparser=sub_parser).set_defaults(func=nlm)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:     # py<=3.6
        parser.parse_args(['-h'])
