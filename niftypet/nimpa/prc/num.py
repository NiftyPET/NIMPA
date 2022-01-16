import logging

import numpy as np
import scipy.ndimage as ndi

try:
    import cuvec as cu

    from . import improc
except ImportError: # GPU routines not compiled
    cu, improc = None, None

__all__ = ['conv_separable', 'isub', 'nlm']
log = logging.getLogger(__name__)
FLOAT_MAX = np.float32(np.inf)


def conv_separable(vol, knl, dev_id=0, output=None, sync=True):
    """
    Args:
      vol(ndarray): Can be any number of dimensions `ndim`
        (GPU requires `ndim <= 3`).
      knl(ndarray): `ndim` x `width` separable kernel
        (GPU requires `width <= 17`).
      dev_id(int or bool): GPU device ID to try [default: 0].
        Set to `False` to force CPU fallback.
    """
    assert vol.ndim == len(knl)
    assert knl.ndim == 2
    if len(knl) > 3 or knl.shape[1] > 17:
        log.warning("kernel larger than 3 x 17 not supported on GPU")
        dev_id = False
    if improc is not None and dev_id is not False:
        log.debug("GPU conv")

        pad = 3 - len(knl)        # <3 dims
        k_pad = 17 - knl.shape[1] # kernel width < 17

        if pad or k_pad:
            knl = np.pad(knl, [(0, pad), (k_pad // 2, k_pad//2 + (k_pad%2))])
            if pad:
                knl[-pad:, 17 // 2] = 1
                vol = vol.reshape(vol.shape + (1,) * pad)
        src = cu.asarray(vol, 'float32')
        knl = cu.asarray(knl, 'float32')
        if output is not None:
            output = cu.asarray(output, 'float32')
        dst = improc.convolve(src, knl, output=output, dev_id=dev_id, sync=sync,
                              log=log.getEffectiveLevel())
        res = cu.asarray(dst, vol.dtype)
        return res[(slice(0, None),) * (res.ndim - pad) + (-1,) * pad] if pad else res
    else:
        log.debug("CPU conv")
        for dim in range(len(knl)):
            h = knl[dim].reshape((1,) * dim + (-1,) + (1,) * (len(knl) - dim - 1))
            vol = ndi.convolve(vol, h, output=output, mode='constant', cval=0.)
        return vol


def check_cuvec(a, shape, dtype, allow_none=True):
    """Asserts that CuVec `a` is of `shape` and `dtype`"""
    if a is None:
        assert allow_none, "must not be None"
        return
    if not isinstance(a, cu.CuVec):
        raise TypeError("must be a CuVec")
    if np.dtype(a.dtype) != np.dtype(dtype):
        raise TypeError(f"dtype must be {dtype}: got {a.dtype}")
    if a.shape != shape:
        raise IndexError(f"shape must be {shape}: got {a.shape}")


def nlm(img, ref, sigma=1, half_width=4, output=None, dev_id=0, sync=True):
    """
    3D Non-local means (NLM) guided filter.
    Args:
      img(3darray): input image to be filtered.
      ref(3darray): reference (guidance) image.
      sigma(float): NLM parameter.
      half_width(int): neighbourhood half-width.
      output(CuVec): pre-existing output memory.
      sync(bool): whether to `cudaDeviceSynchronize()` after GPU operations.
    Reference: https://doi.org/10.1109/CVPR.2005.38
    """
    img = cu.asarray(img, 'float32')
    ref = cu.asarray(ref, 'float32')
    if img.shape != ref.shape:
        raise IndexError(f"{img.shape} and {ref.shape} don't match")
    if img.ndim != 3:
        raise IndexError(f"must be 3D: got {img.ndim}D")
    check_cuvec(output, img.shape, 'float32')
    return cu.asarray(
        improc.nlm(img, ref, sigma=sigma, half_width=half_width, output=output, dev_id=dev_id,
                   sync=sync, log=log.getEffectiveLevel()))


def isub(img, idxs, output=None, dev_id=0, sync=True):
    """
    output = img[idxs, ...]
    Args:
      img(2darray): input image.
      idxs(1darray['int32']): indicies into the first dimension of `img`.
      output(CuVec): pre-existing output memory.
      sync(bool): whether to `cudaDeviceSynchronize()` after GPU operations.
    """
    img = cu.asarray(img, 'float32')
    idxs = cu.asarray(idxs, 'int32')
    if img.ndim != 2:
        raise IndexError(f"must be 2D: got {img.ndim}D")
    check_cuvec(output, (idxs.shape[0], img.shape[1]), 'float32')
    return cu.asarray(
        improc.isub(img, idxs, output=output, dev_id=dev_id, sync=sync,
                    log=log.getEffectiveLevel()))
