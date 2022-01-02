"""
NIMPA: functions for neuro image processing and analysis
Generates images.
"""
import logging
import math
import pathlib

import numpy as np
import scipy.ndimage as ndi

try:
    from miutil.plot import imscroll
except ImportError as err: # NOQA: F841

    def imscroll(*_, **__):
        """delay matplotlib import error for later"""
        raise err  # NOQA: F821


from ..prc import imio

log = logging.getLogger(__name__)


def absmax(a):
    amax = a.max()
    amin = a.min()
    return np.where(-amin > amax, amin, amax)


def create_disk(shape_in, r=1, a=0, b=0, gen_scale=1, threshold=None):
    if len(shape_in) == 2:
        shape = (1,) + shape_in
    if len(shape_in) == 3:
        shape = shape_in

    imsk = np.zeros((gen_scale * shape[1], gen_scale * shape[2]), dtype=np.float32)
    for t in np.arange(0, math.pi, math.pi / (gen_scale*400)):
        x = gen_scale * r * np.cos(t) + gen_scale*a
        y = gen_scale * r * np.sin(t) + gen_scale*b

        for ys in np.arange(-y + 2*b*gen_scale, y, 0.5):
            v = 0.5 * gen_scale * shape[1] - np.ceil(ys)
            u = 0.5 * gen_scale * shape[2] + np.floor(x)
            imsk[v.astype(np.int16), u.astype(np.int16)] = 1.

    if gen_scale > 1:
        imsk = ndi.interpolation.zoom(imsk, gen_scale**-1, order=1)

    if threshold:
        imsk = imsk > threshold

    if len(shape_in) == 3:
        msk = np.repeat(imsk.reshape((1, shape[1], shape[1])), shape[0], axis=0)
    elif len(shape_in) == 2:
        msk = imsk
    return msk


def profile_points(im, p0, p1, steps=100):
    p = np.array([p1[0] - p0[0], p1[1] - p0[1]])
    nrm = np.sum(p**2)**.5
    p = p / nrm

    tt = np.linspace(0, nrm, steps)
    profile = np.zeros(len(tt), dtype=im.dtype)

    c = 0
    for t in tt:
        u = t * p[0] + p0[0]
        v = t * p[1] + p0[1]

        profile[c] = im[int(round(v)), int(round(u))]
        c += 1

    return profile


# ----------------------------------------------------------------------------------------------------------------------------
def imdiff(imref, imnew, verbose=False, plot=False, cmap='bwr'):
    """
    Compare the new image (imnew) to the reference image (imref),
    returning (and optional plotting) the difference.
    """
    if isinstance(imref, (str, pathlib.Path)):
        imref = imio.getnii(imref)
        log.info('using NIfTI files as image input for the reference')
    elif isinstance(imref, (np.ndarray, np.generic)):
        log.info('using Numpy arrays as input for the reference image')
    elif isinstance(imref, dict):
        imref = imref['im']
        log.info('using the input dictionary with an image for reference')

    if isinstance(imnew, str):
        imnew = imio.getnii(imnew)
        log.info('using NIfTI files as input for the new image')
    elif isinstance(imnew, (np.ndarray, np.generic)):
        log.info('using Numpy arrays as input for the new image')
    elif isinstance(imnew, dict):
        imnew = imnew['im']
        log.info('using the input dictionary with a new image for comparison')

    # -----------------------------------------------------------------------------------
    # MAXIMUM PROJECTION IMAGE
    if plot:
        import matplotlib.pyplot as plt

        # > threshold percentage used in plotting (helps ignoring singular hot values)
        maxproj_thrshl = 0.7

        # > maximum projection along axis ax
        def maxproj(imdiff, ax):
            # > maximum projection image
            imp = np.max(imdiff, axis=ax)
            # > minimum projection image
            imn = np.min(imdiff, axis=ax)
            # > max mask
            mmsk = imp > abs(imn)
            # > form the highest intensity projection image (positive and negative)
            im = imn.copy()
            im[mmsk] = imp[mmsk]
            # > maximum/minimum value in the difference image for a symmetrical colour map
            valmax = max(np.max(imp), np.min(imn))
            return im, valmax

    # -----------------------------------------------------------------------------------

    if imnew.ndim == 3 and imref.ndim == 3 and imnew.shape == imref.shape:
        imnew = imnew[np.newaxis, ...]
        imref = imref[np.newaxis, ...]
        log.info('using 3D images as input')
        Nim = 1
    elif imnew.ndim == 4 and imref.ndim == 4 and imnew.shape == imref.shape:
        log.info('using 4D images as input')
        # > this assumes that axis 0 encodes image frames (e.g., along time)
        Nim = imnew.shape[0]
    else:
        raise ValueError('the input images have to be of the same dimensions and shape')

    mape = np.zeros(Nim)
    mae = np.zeros(Nim)
    mad = np.zeros(Nim)

    for i in range(Nim):
        # > maximum voxel value of reference image
        mx = np.max(imref[i, ...])

        # > create a image mask based on the max the 0.1% value
        msk = imref[i, ...] > 0.001 * mx

        # > image difference
        imdiff = imref[i, ...] - imnew[i, ...]

        # > mean absolute percentage difference
        mape[i] = np.mean(abs(imdiff[msk] / imref[i, msk])) * 100

        # > mean absolute error
        mae[i] = np.mean(abs(imdiff[msk]))

        # > maximum absolute difference
        mad[i] = np.max(abs(imdiff[msk]))

        if verbose:
            print('---------------------------------------------------------------')
            print('>> frame {}: mean absolute relative image difference [%]:'.format(i))
            print(mape[i])

            print('>> frame {}: mean absolute image difference:'.format(i))
            print(mae[i])

            print('>> frame {}:maximum absolute image difference:'.format(i))
            print(mad[i])

        if plot:

            fig = plt.figure(figsize=(12, 6))
            fig.suptitle('maximum absolute difference projection along 3 axes', fontsize=12,
                         fontweight='bold')

            im, valmax = maxproj(imdiff, 0)

            plt.subplot(131)
            plt.imshow(im, cmap=cmap, vmax=maxproj_thrshl * valmax, vmin=-maxproj_thrshl * valmax)
            plt.colorbar()

            im, valmax = maxproj(imdiff, 1)

            plt.subplot(132)
            plt.imshow(im, cmap=cmap, vmax=maxproj_thrshl * valmax, vmin=-maxproj_thrshl * valmax)
            # plt.colorbar()

            im, valmax = maxproj(imdiff, 2)
            plt.subplot(133)
            plt.imshow(im, cmap=cmap, vmax=maxproj_thrshl * valmax, vmin=-maxproj_thrshl * valmax)
            # plt.colorbar()

            plt.show()

    if Nim > 1:
        out = {'mape': mape, 'mae': mae, 'mad': mad}
    else:
        out = {'mape': mape[0], 'mae': mae[0], 'mad': mad[0]}

    return out
