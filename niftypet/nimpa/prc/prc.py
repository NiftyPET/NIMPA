"""
NIMPA: functions for neuro image processing and analysis
including partial volume correction (PVC) and ROI extraction and analysis.
"""
import logging
import multiprocessing
import os
import pathlib
import re
import sys
from subprocess import run
from textwrap import dedent
from warnings import warn

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from pkg_resources import resource_filename
from tqdm.auto import trange

from . import imio, regseg
from .num import conv_separable

try:
    import SimpleITK as sitk
    sitk_flag = True
except ImportError:
    sitk_flag = False

log = logging.getLogger(__name__)

# possible extentions for DICOM files
dcmext = ('dcm', 'DCM', 'ima', 'IMA', 'img', 'IMG')
niiext = ('nii.gz', 'nii', 'img', 'hdr')


# ----------------------------------------------------------------------
def num(s):
    '''Converts the string to a float or integer number.'''
    try:
        return int(s)
    except ValueError:
        return float(s)


# ----------------------------------------------------------------------
def psf_gaussian(vx_size=(1, 1, 1), fwhm=(6, 5, 5), hradius=8):
    '''
    Separable kernels for Gaussian convolution executed on the GPU device
    The output kernels are in this order: z, y, x
    '''

    # if voxel size is given as scalar, interpret it as an isotropic
    # voxel size.
    if isinstance(vx_size, (float, int)):
        vx_size = [vx_size, vx_size, vx_size]

    # the same for the Gaussian kernel
    if isinstance(fwhm, (float, int)):
        fwhm = [fwhm, fwhm, fwhm]

    # avoid zeros in FWHM
    fwhm = [x + 1e-3 * (x <= 0) for x in fwhm]

    xSig = (fwhm[2] / vx_size[2]) / (2 * (2 * np.log(2))**.5)
    ySig = (fwhm[1] / vx_size[1]) / (2 * (2 * np.log(2))**.5)
    zSig = (fwhm[0] / vx_size[0]) / (2 * (2 * np.log(2))**.5)

    # get the separated kernels
    x = np.arange(-hradius, hradius + 1)
    xKrnl = np.exp(-(x**2 / (2 * xSig**2)))
    yKrnl = np.exp(-(x**2 / (2 * ySig**2)))
    zKrnl = np.exp(-(x**2 / (2 * zSig**2)))

    # normalise kernels
    xKrnl /= np.sum(xKrnl)
    yKrnl /= np.sum(yKrnl)
    zKrnl /= np.sum(zKrnl)

    krnl = np.array([zKrnl, yKrnl, xKrnl], dtype=np.float32)

    # for checking if the normalisation worked
    # np.prod( np.sum(krnl,axis=1) )

    # return all kernels together
    return krnl


# ----------------------------------------------------------------------
def psf_measured(scanner='mmr', scale=1):
    if scanner == 'mmr':
        # file name for the mMR's PSF and chosen scale
        fdat = resource_filename("niftypet.nimpa", f"auxdata/PSF-17_scl-{scale:d}.npy")
        # transaxial and axial PSF
        Hxy, Hz = np.load(fdat)
        return np.array([Hz, Hxy, Hxy], dtype=np.float32)
    raise NameError(f'Unsupported scanner ({scanner}):'
                    ' only Siemens mMR (mmr) is currently supported')


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# I M A G E   S M O O T H I N G
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def imsmooth(fim, fwhm=4, psf=None, voxsize=None, fout='', output='image', output_array=None,
             gpu=None, dev_id=0, sync=True, Cnt=None):
    '''
    Smooth image using Gaussian filter with either the PSF or FWHM given
    as an option.  By default FWHM = 4 is used with voxel size assumed 1 mm.
    Arguments:
    - fim:  can be a NIfTI image file or Numpy array
    - fwhm: the width at half max of the Gaussian kernel (z,y,x)
    - psf:  the point spread function for each direction (z,y,x) given as a
            Numpy matrix of 3x17 and used on the GPU as separable kernel
    - voxsize: size of the voxel (can be in mm or cm)
    - fout: the output file path
    - output: can be image as Numpy array or file or both.
    - dev_id: the ID of the CUDA device to try to use for computation
      (set to `False` to force disable GPU)
    - sync: whether to `cudaDeviceSynchronize()` after GPU operations
    - gpu: ignored
    '''
    if gpu is not None:
        warn("gpu is automatic", DeprecationWarning, stacklevel=2)
    if Cnt is not None and 'DEVID' in Cnt:
        dev_id = Cnt['DEVID']

    isfile = False
    if isinstance(fim, str) and os.path.isfile(fim):
        isfile = True
        imd = imio.getnii(fim, output='all')
        im = imd['im']
        voxsize = imd['voxsize']
        affine = imd['affine']
    elif isinstance(fim, dict) and 'voxsize' in fim:
        im = fim['im']
        voxsize = fim['voxsize']
        affine = fim['affine']
    elif isinstance(fim, (np.ndarray, np.generic)):
        im = fim
    else:
        raise ValueError(
            "incorrect image input.\nNIfTI file path, dictionary or Numpy array are accepted.")

    if psf is None:
        if voxsize is None and Cnt is not None and 'SO_VXZ' in Cnt:
            voxsize = [Cnt['SO_VXZ'], Cnt['SO_VXY'], Cnt['SO_VXX']]
        elif voxsize is None and Cnt is None:
            raise ValueError('the correct voxel size has to be provided')
        psf = psf_gaussian(vx_size=voxsize, fwhm=fwhm)
    imsmo = conv_separable(im, psf, output=output_array, dev_id=dev_id, sync=sync)

    # output dictionary
    dctout = {}
    dctout['im'] = imsmo
    dctout['fwhm'] = fwhm

    if isfile and fout == '':
        if fim.endswith('.nii.gz'):
            fout = fim.split('.nii.gz')[0] + '_smo' + str(fwhm).replace('.', '-') + '.nii.gz'
        else:
            fout = os.path.splitext(fim)[0] + '_smo' + str(fwhm).replace(
                '.', '-') + os.path.splitext(fim)[1]

    if isfile:
        dctout['affine'] = affine
        dctout['fim'] = fout

        imio.array2nii(
            imsmo, affine, fout, trnsp=(imd['transpose'].index(0), imd['transpose'].index(1),
                                        imd['transpose'].index(2)), flip=imd['flip'])

    if output == 'all':
        return dctout
    elif output == 'image':
        return imsmo
    elif output == 'file':
        return fout
    else:
        return None


# ==================================================================================================
# > get projections along 3D image axes
def im_project3(im):
    ''' project intensities on the 3 axes x, y ,z.
    '''

    if isinstance(im, str):
        img = imio.getnii(im)
    elif isinstance(im, (np.ndarray, np.generic)):
        img = im
    else:
        raise ValueError('unrecognised image input')

    qx = np.sum(img, axis=(0, 1))
    qy = np.sum(img, axis=(0, 2))
    qz = np.sum(img, axis=(1, 2))

    return qx, qy, qz


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# FUNCTIONS: TRIM &  PARTIAL VOLUME EFFECTS AND CORRECTION
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def imtrimup(fims, refim='', affine=None, scale=2, divdim=8**2, fmax=0.05, int_order=0, outpath='',
             fname='', fcomment='', fcomment_pfx='', store_avg=False, store_img_intrmd=False,
             store_img=False, imdtype=np.float32, memlim=False, verbose=False, Cnt=None):
    '''
    Trim and upsample PET image(s), e.g., for GPU execution,
    PVC correction, ROI sampling, etc.
    The input images 'fims' can be passed in multiple ways:
    1. as a string of the folder containing NIfTI files
    2. as a string of a NIfTI file path (this way a 4D image can be loaded).
    3. as a list of NIfTI file paths.
    4. as a 3D or 4D image
    Parameters:
    -----------
    refim:  Path to the reference image, which was already trimmed.
            Needs the trimming parameters stored in the NIfTI header in 'descrip'.
    affine: affine matrix, 4x4, for all the input images in case when images
            passed as a matrix (all have to have the same shape and data type)
    scale:  the scaling factor for upsampling has to be greater than 1.  It can be a scalar or
            a vector/tuple with elements for each image dimension.
    divdim: image divisor, i.e., the dimensions are a multiple of this number (default 64)
    int_order: interpolation order (0-nearest neighbour, 1-linear, as in scipy)
    fmax: fraction of the max image value used for finding image borders for trimming
    outpath: output folder path
    fname: file name when image given as a numpy matrix
    fcomment: part of the name of the output file, left for the user as a comment
    fcomment_pfx: part of the name of the output file at the start (prefix)
    store_img_intrmd: stores intermediate images with suffix '_i'
    store_avg: stores the average image (if multiple images are given)
    imdtype: data type for output images
    memlim: Ture for cases when memory is limited and takes more processing time instead.
    verbose: verbose mode [True/False]
    '''
    if Cnt is None:
        Cnt = {}

    # case when input folder is given
    if isinstance(fims, (str, pathlib.PurePath)) and os.path.isdir(fims):
        # list of input images (e.g., PET)
        fimlist = [os.path.join(fims, f) for f in os.listdir(fims) if f.endswith(niiext)]
        imdic = imio.niisort(fimlist, memlim=memlim)
        if not (imdic['N'] > 50 and memlim):
            imin = imdic['im']
        imshape = imdic['shape']
        affine = imdic['affine']
        fldrin = fims
        fnms = [os.path.basename(f).split('.nii')[0] for f in imdic['files'] if f is not None]
        # number of images/frames
        Nim = imdic['N']
        using_multiple_files = True

    # case when input file is a 3D or 4D NIfTI image
    elif isinstance(
            fims, (str, pathlib.PurePath)) and os.path.isfile(fims) and str(fims).endswith(niiext):
        imdic = imio.getnii(fims, output='all')
        imin = imdic['im']
        if imin.ndim == 3:
            imin.shape = (1, imin.shape[0], imin.shape[1], imin.shape[2])
        imdtype = imdic['dtype']
        imshape = imdic['shape'][-3:]
        affine = imdic['affine']
        fldrin = os.path.dirname(fims)
        fnms = imin.shape[0] * [os.path.basename(fims).split('.nii')[0]]
        # number of images/frames
        Nim = imin.shape[0]

    # case when a list of input files is given
    elif isinstance(fims, list) and all(map(os.path.isfile, fims)):
        imdic = imio.niisort(fims, memlim=memlim)
        if not (imdic['N'] > 50 and memlim):
            imin = imdic['im']
        imshape = imdic['shape']
        affine = imdic['affine']
        imdtype = imdic['dtype']
        fldrin = os.path.dirname(fims[0])
        fnms = [os.path.basename(f).split('.nii')[0] for f in imdic['files']]
        # number of images/frames
        Nim = imdic['N']
        using_multiple_files = True

    # case when an array [#frames, zdim, ydim, xdim].  Can be 3D or 4D
    elif isinstance(fims, (np.ndarray, np.generic)) and (fims.ndim == 4 or fims.ndim == 3):
        # check image affine
        if affine.shape != (4, 4):
            raise ValueError('Affine should be a 4x4 array.')
        # create a copy to avoid mutation when only one image (3D)
        imin = np.copy(fims)
        if fims.ndim == 3:
            imin.shape = (1, imin.shape[0], imin.shape[1], imin.shape[2])
        imshape = imin.shape[-3:]
        fldrin = os.path.join(os.path.expanduser('~'), 'NIMPA_output')
        if fname == '':
            fnms = imin.shape[0] * ['NIMPA']
        else:
            fnms = imin.shape[0] * [fname]
        # number of images/frames
        Nim = imin.shape[0]

    else:
        raise TypeError('Wrong data type input.')

    # > if using reference trimmed image:
    ref_flag = False
    if os.path.isfile(refim):
        refdct = imio.getnii(refim, output='all')
        nii_descrp = refdct['hdr']['descrip'].item().decode('utf-8')
        if 'trim' in nii_descrp:
            # > find all the numbers (int and floats)
            parstr = re.findall(r'-*\d+\.*\d*', nii_descrp)
            try:
                ix0, ix1, iy0, iy1, iz0, scale0, scale1, scale2, fmax = (num(s) for s in parstr)
                scale = [scale0, scale1, scale2]
            except ValueError:
                ix0, ix1, iy0, iy1, iz0, scale, fmax = (num(s) for s in parstr)
                scale = [scale, scale, scale]
            except Exception:
                log.error('the reference file does not have the trimming information.')

            ref_flag = True
            log.info(' using the trimming parameters of the reference image.')
        else:
            log.warning(' the reference image does not contain trimming info--using default.')

    # ------------------------------------------------------
    # store images in this folder
    if outpath == '':
        petudir = os.path.join(fldrin, 'trimmed')
    else:
        petudir = os.path.join(outpath, 'trimmed')
    imio.create_dir(petudir)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # scale is preferred to be integer
    try:
        scale = np.int8(scale)
    except ValueError:
        raise ValueError('scale has to be an integer or array of integers.')

    # > check if scale is given as scalar of array.
    # > if scalar, change it to an array.
    if isinstance(scale, np.int8):
        scale = scale * np.ones(3, dtype=np.int8)

    # > scale factor as the inverse of scale
    sf = 1 / np.float64(scale)
    log.debug(' upsampling scale {}, giving resolution scale factor {} for {} images.'.format(
        scale, sf, Nim))
    # ------------------------------------------------------

    # ------------------------------------------------------
    # scaled input image and get a sum image as the base for trimming
    if any(scale > 1):
        newshape = (scale[0] * imshape[0], scale[1] * imshape[1], scale[2] * imshape[2])
        imsum = np.zeros(newshape, dtype=imdtype)
        if not memlim:
            imscl = np.zeros((Nim,) + newshape, dtype=imdtype)
            with trange(Nim, desc="loading-scaling",
                        disable=log.getEffectiveLevel() > logging.INFO,
                        leave=log.getEffectiveLevel() <= logging.INFO) as pbar:
                for i in pbar:
                    imscl[i, :, :, :] = ndi.interpolation.zoom(imin[i, :, :, :], tuple(scale),
                                                               order=int_order)
                    imsum += imscl[i, :, :, :]
        else:
            with trange(Nim, desc="loading-scaling",
                        disable=log.getEffectiveLevel() > logging.INFO,
                        leave=log.getEffectiveLevel() <= logging.INFO) as pbar:
                for i in pbar:
                    if Nim > 50 and using_multiple_files:
                        imin_temp = imio.getnii(imdic['files'][i])
                        imsum += ndi.interpolation.zoom(imin_temp, tuple(scale), order=int_order)
                        log.debug(' image sum: read {}'.format(imdic['files'][i]))
                    else:
                        imsum += ndi.interpolation.zoom(imin[i, :, :, :], tuple(scale),
                                                        order=int_order)
    else:
        imscl = imin
        imsum = np.sum(imin, axis=0)

    # import pdb; pdb.set_trace()
    # ------------------------------------------------------

    if not ref_flag:
        # find the object bounding indexes in x, y and z axes, e.g., ix0-ix1 for the x axis
        qx, qy, qz = im_project3(imsum)

        ix0 = np.argmax(qx > (fmax * np.nanmax(qx)))
        ix1 = ix0 + np.argmin(qx[ix0:] > (fmax * np.nanmax(qx)))

        iy0 = np.argmax(qy > (fmax * np.nanmax(qy)))
        iy1 = iy0 + np.argmin(qy[iy0:] > (fmax * np.nanmax(qy)))

        iz0 = np.argmax(qz > (fmax * np.nanmax(qz)))

        # import pdb; pdb.set_trace()

        # find the maximum voxel range for x and y axes
        IX = ix1 - ix0 + 1
        IY = iy1 - iy0 + 1
        tmp = max(IX, IY)
        # > get the range such that it is divisible by
        # > divdim (64 by default) for GPU execution
        IXY = divdim * ((tmp+divdim-1) // divdim)
        div = (IXY-IX) // 2
        # x
        ix0 -= div
        ix1 += (IXY-IX) - div
        # y
        div = (IXY-IY) // 2
        iy0 -= div
        iy1 += (IXY-IY) - div
        # z
        tmp = (len(qz) - iz0 + 1)
        IZ = divdim * ((tmp+divdim-1) // divdim)
        iz0 -= IZ - tmp + 1

    # save the trimming parameters in a dic
    trimpar = {'x': (ix0, ix1), 'y': (iy0, iy1), 'z': (iz0), 'fmax': fmax, 'scale': scale}

    # new dims (z,y,x)
    newdims = (imsum.shape[0] - iz0, iy1 - iy0 + 1, ix1 - ix0 + 1)
    imtrim = np.zeros((Nim,) + newdims, dtype=imdtype)
    imsumt = np.zeros(newdims, dtype=imdtype)
    # in case of needed padding (negative indx resulting above)
    # the absolute values are supposed to work like padding in case the indx are negative
    iz0s, iy0s, ix0s = iz0, iy0, ix0
    iz0t, iy0t, ix0t = 0, 0, 0
    if iz0 < 0:
        iz0s = 0
        iz0t = abs(iz0)
        log.warning(
            dedent('''\
            -----------------------------------------------------------------
            Correcting for trimming outside the original image (z-axis)'
            -----------------------------------------------------------------'''))

    if iy0 < 0:
        iy0s = 0
        iy0t = abs(iy0)
        log.warning(
            dedent('''\
            -----------------------------------------------------------------
            Correcting for trimming outside the original image (y-axis)'
            -----------------------------------------------------------------'''))

    if ix0 < 0:
        ix0s = 0
        ix0t = abs(ix0)
        log.warning(
            dedent('''\
            -----------------------------------------------------------------
            Correcting for trimming outside the original image (x-axis)
            -----------------------------------------------------------------'''))

    # > in case the upper index goes beyond the scaled but untrimmed image
    iy1t = imsumt.shape[1]
    if iy1 >= imsum.shape[1]:
        iy1t -= iy1 + 1

    # > the same for x
    ix1t = imsumt.shape[2]
    if ix1 >= imsum.shape[2]:
        ix1t -= ix1 + 1

    # first trim the sum image
    imsumt[iz0t:, iy0t:iy1t, ix0t:ix1t] = imsum[iz0s:, iy0s:iy1 + 1, ix0s:ix1 + 1]

    # > new affine matrix for the upscaled and trimmed image
    # > use the scale factor reversely for consistency
    A = np.diag(np.append(sf[::-1], 1.) * np.diag(affine))

    # > note half of new voxel offset is used for the new centre of voxels
    A[0, 3] = affine[0, 3] + A[0, 0] * (ix0-0.5)
    A[1, 3] = affine[1, 3] + (affine[1, 1] * (imshape[1] - 1) - A[1, 1] * (iy1-0.5))
    A[2, 3] = affine[2, 3] - A[1, 1] * 0.5
    A[3, 3] = 1

    # output dictionary
    dctout = {'affine': A, 'trimpar': trimpar, 'imsum': imsumt}

    # NIfTI image description (to be stored in the header)
    niidescr = 'trim(x,y,z):' + str(trimpar['x']) + ',' + str(trimpar['y']) + ',' + str(
        (trimpar['z'],)) + ';scale=' + str(scale) + ';fmx=' + str(fmax)

    # > remove brackets and spaces from the file name
    scale_fnm = str(scale).replace('[', '').replace(']', '').replace(' ', '-')

    # store the sum image
    if store_avg and Nim > 1:
        fsum = os.path.join(
            petudir,
            fcomment_pfx + 'avg_trimmed-upsampled-scale-' + scale_fnm + fcomment + '.nii.gz')
        imio.array2nii(imsumt[::-1, ::-1, :], A, fsum, descrip=niidescr)
        log.debug('saved averaged image to: {}'.format(fsum))
        dctout['fsum'] = fsum

    # list of file names for the upsampled and trimmed images
    fpetu = []
    # > perform the trimming and save the intermediate images if requested
    with trange(Nim, desc="finalising trimming/scaling",
                disable=log.getEffectiveLevel() > logging.INFO,
                leave=log.getEffectiveLevel() <= logging.INFO) as pbar:

        for i in pbar:

            # memory saving option, second time doing interpolation
            if memlim:
                if Nim > 50 and using_multiple_files:
                    imin_temp = imio.getnii(imdic['files'][i])
                    im = ndi.interpolation.zoom(imin_temp, tuple(scale), order=int_order)
                    log.debug('image scaling: {}'.format(imdic['files'][i]))
                else:
                    im = ndi.interpolation.zoom(imin[i, :, :, :], tuple(scale), order=int_order)
            else:
                im = imscl[i, :, :, :]

            # trim the scaled image
            imtrim[i, iz0t:, iy0t:iy1t, ix0t:ix1t] = im[iz0s:, iy0s:iy1 + 1, ix0s:ix1 + 1]

            # save the up-sampled and trimmed PET images
            if store_img_intrmd:
                _frm = '_trmfrm' + str(i)
                _fstr = '_trimmed-upsampled-scale-' + scale_fnm + _frm * (Nim > 1) + fcomment
                fpetu.append(os.path.join(petudir, fnms[i] + _fstr + '_i.nii.gz'))
                imio.array2nii(imtrim[i, ::-1, ::-1, :], A, fpetu[i], descrip=niidescr)
                log.debug('saved upsampled PET image to: {}'.format(fpetu[i]))

    if store_img:
        _nfrm = '_nfrm' + str(Nim)
        fim = os.path.join(petudir, fcomment_pfx + 'trimmed-upsampled-scale-' +
                           scale_fnm) + _nfrm * (Nim > 1) + fcomment + '.nii.gz'
        log.info('storing image to:\n{}'.format(fim))

        imio.array2nii(np.squeeze(imtrim[:, ::-1, ::-1, :]), A, fim, descrip=niidescr)
        dctout['fim'] = fim

    # file names (with paths) for the intermediate PET images
    dctout['fimi'] = fpetu
    dctout['im'] = np.squeeze(imtrim)
    dctout['N'] = Nim
    dctout['affine'] = A

    return dctout


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def iyang(imgIn, krnl, imgSeg, Cnt, itr=5):
    '''
    Partial volume correction using iterative Yang method.
    Arguments:
        imgIn: input image which is blurred due to the PSF of the scanner
        krnl: shift invariant kernel of the PSF
        imgSeg: segmentation into regions starting with 0 (e.g., background)
          and then next integer numbers
        itr: number of iteration (default 5)
    '''
    dim = imgIn.shape
    m = np.int32(np.max(imgSeg))
    m_a = np.zeros((m + 1, itr), dtype=np.float32)

    for jr in range(0, m + 1):
        m_a[jr, 0] = np.mean(imgIn[imgSeg == jr])

    # init output image
    imgOut = np.copy(imgIn)

    # iterative Yang algorithm:
    for i in range(0, itr):
        log.debug('PVC Yang iteration = {}'.format(i))
        # piece-wise constant image
        imgPWC = imgOut
        imgPWC[imgPWC < 0] = 0
        for jr in range(0, m + 1):
            imgPWC[imgSeg == jr] = np.mean(imgPWC[imgSeg == jr])

        # blur the piece-wise constant image
        imgSmo = conv_separable(imgPWC, krnl, dev_id=Cnt['DEVID'])

        # correction factors
        imgCrr = np.ones(dim, dtype=np.float32)
        imgCrr[imgSmo > 0] = imgPWC[imgSmo > 0] / imgSmo[imgSmo > 0]
        imgOut = imgIn * imgCrr
        for jr in range(0, m + 1):
            m_a[jr, i] = np.mean(imgOut[imgSeg == jr])

    return imgOut, m_a


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# G E T   P A R C E L L A T I O N S   F O R   P V C   A N D   R O I   E X T R A C T I O N
# ------------------------------------------------------------------------------------------------------
def pvc_iyang(
    petin,
    mrin,
    Cnt,
    pvcroi,
    krnl,
    itr=5,
    tool='niftyreg',
    faff=None,
    outpath='',
    fcomment='',
    store_img=False,
    store_rois=False,
    matlab_eng_name='',
):
    ''' Perform partial volume (PVC) correction of PET data (petin) using MRI data (mrin).
        The PVC method uses iterative Yang method.
        GPU based convolution is the key routine of the PVC.
        Input:
        -------
        petin:  either a dictionary containing image data, file name and affine transform,
                or a string of the path to the NIfTI file of the PET data.
        mrin: a dictionary of MRI data, including the T1w image, which can be given
                in DICOM (field 'T1DCM') or NIfTI (field 'T1nii').  The T1w image data
                is needed for co-registration to PET if affine is not given in the text
                file with its path in faff.
        Cnt:    a dictionary of paths for third-party tools:
                * dcm2niix: Cnt['DCM2NIIX']
                * niftyreg, resample: Cnt['RESPATH']
                * niftyreg, rigid-reg: Cnt['REGPATH']
        pvcroi: list of regions (also a list) with number label to distinguish
                the parcellations.  The numbers correspond to the image values
                of the parcellated T1w image.  E.g.:
                pvcroi = [
                    [36], # ROI 1 (single parcellation region)
                    [35], # ROI 2
                    [39, 40, 72, 73, 74], # ROI 3 (multiple parcellation regions)
                    ...
                ]
        kernel: the point spread function (PSF) specific for the camera and the object.
                It is given as a 3x17 matrix, a 17-element kernel for each dimension (x,y,z).
                It is used in the GPU-based convolution using separable kernels.
        outpath:path to the output of the resulting PVC images
        faff:   a text file of the affine transformations needed to get the MRI into PET space.
                If not provided, it will be obtained from the performed rigid transformation.
                For this the MR T1w image is required.  If faff and T1w image are not provided,
                it will results in exception/error.
        fcomment:a string used in naming the produced files, helpful for distinguishing them.
        tool:   co-registration tool.  By default it is NiftyReg, but SPM is also
                possible (needs Matlab engine and more validation)
        itr:    number of iterations used by the PVC.  5-10 should be enough (5 default)
    '''
    # get all the input image properties
    if isinstance(petin, dict):
        im = petin['im']
        fpet = petin['fpet']
        B = petin['affine']
    elif isinstance(petin, str) and os.path.isfile(petin):
        imdct = imio.getnii(petin, output='all')
        im = imdct['im']
        B = imdct['affine']
        fpet = petin
    else:
        raise IOError('e> unrecognised input PET file')

    if im.ndim != 3:
        raise IndexError(
            'Only 3D images are expected in this method of partial volume correction.')

    # > avoid registration if the provided parcellation is already in PET space
    # > it is assumed so if the parcellation is given as a file path and no affine is given
    noreg = False
    if isinstance(mrin, str) and os.path.isfile(mrin):
        prcl_dir = os.path.dirname(mrin)
        tmpdct = imio.getnii(mrin, output='all')
        if faff is None and tmpdct['shape'] == imdct['shape']:
            fprcu = mrin
            fprc = mrin
            noreg = True
        elif faff is not None:
            fprc = mrin

    elif isinstance(mrin, dict) and os.path.isfile(mrin['T1lbl']):
        fprc = mrin['T1lbl']
        prcl_dir = os.path.dirname(fprc)

    else:
        raise NameError('e> missing or incorrect labels/parcellations')

    # establish the output folder
    if outpath == '':
        oprcl = os.path.join(prcl_dir, 'PVC-preprocessed')
        opvc = os.path.join(os.path.dirname(fpet), 'PVC')
    else:
        oprcl = os.path.join(outpath, 'PVC-preprocessed')
        opvc = os.path.join(outpath, 'PVC')

    # > create folders
    imio.create_dir(oprcl)
    imio.create_dir(opvc)
    if store_rois:
        orois = os.path.join(opvc, 'ROIs')
        imio.create_dir(orois)

    # > output dictionary
    outdct = {}

    # =================================================================
    # > if affine transformation (faff) is not given then register T1 to PET
    # and resample parcellations
    if not noreg and faff is None:
        ft1w = imio.pick_t1w(mrin)
        if tool == 'spm':
            regdct = regseg.coreg_spm(fpet, ft1w, matlab_eng_name=matlab_eng_name,
                                      fcomment=fcomment,
                                      outpath=os.path.join(outpath, 'PET', 'positioning'))
        elif tool == 'niftyreg':
            regdct = regseg.affine_niftyreg(
                fpet,
                ft1w,
                outpath=os.path.join(outpath, 'PET', 'positioning'),
                fcomment=fcomment,
                executable=Cnt['REGPATH'],
                omp=multiprocessing.cpu_count() / 2,
                rigOnly=True,
                affDirect=False,
                maxit=5,
                speed=True,
                pi=50,
                pv=50,
                smof=0,
                smor=0,
                rmsk=True,
                fmsk=True,
                rfwhm=15.,                                           # millilitres
                rthrsh=0.05,
                ffwhm=15.,                                           # millilitres
                fthrsh=0.05)
        faff = regdct['faff']

    # resample the T1/labels to upsampled PET
    # file name of the parcellation (e.g., GIF-based) upsampled to PET
    if faff is not None and os.path.isfile(faff):
        fprcu = os.path.join(
            oprcl,
            os.path.basename(fprc.split('.')[0] + '_registered_trimmed' + fcomment + '.nii.gz'))

        if tool == 'niftyreg':
            if os.path.isfile(Cnt['RESPATH']):
                cmd = [
                    Cnt['RESPATH'], '-ref', fpet, '-flo', fprc, '-trans', faff, '-res', fprcu,
                    '-inter', '0']
                if log.getEffectiveLevel() >= logging.INFO:
                    cmd.append('-voff')
                run(cmd)
            else:
                raise IOError('e> path to resampling executable is incorrect!')
        elif tool == 'spm':
            regseg.resample_spm(
                fpet,
                fprc,
                faff,
                fimout=fprcu,
                matlab_eng_name=matlab_eng_name,
                intrp=0.,
                del_ref_uncmpr=True,
                del_flo_uncmpr=True,
                del_out_uncmpr=True,
            )
    # =================================================================

    # > get the parcellation labels in the upsampled PET space
    prcdct = imio.getnii(fprcu, output='all')
    prcu = prcdct['im']

    # > path to parcellations in NIfTI format
    prcl_pth = os.path.split(fprc)

    # --------------------------------------------------------------------------
    # > get the parcellation specific for PVC based on the current parcellations
    imgroi = prcu.copy()
    imgroi[:] = 0

    # > number of segments, without the background
    nSeg = len(pvcroi)

    # > create the image of numbered parcellations
    for k in range(nSeg):
        for m in pvcroi[k]:
            imgroi[prcu == m] = k + 1

    # > save the PCV ROIs to a new NIfTI file
    if store_rois:
        froi = os.path.join(orois, prcl_pth[1].split('.nii')[0] + '_PVC-ROIs-inPET.nii.gz')
        imio.array2nii(
            imgroi, prcdct['affine'], froi,
            trnsp=(prcdct['transpose'].index(0), prcdct['transpose'].index(1),
                   prcdct['transpose'].index(2)), flip=prcdct['flip'])
        outdct['froi'] = froi
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # run iterative Yang PVC
    imgpvc, m_a = iyang(im, krnl, imgroi, Cnt, itr=itr)
    # --------------------------------------------------------------------------

    outdct['im'] = imgpvc
    outdct['imroi'] = imgroi
    outdct['fprc'] = fprcu
    outdct['imprc'] = prcu
    if not noreg:
        outdct['faff'] = faff

    if store_img:
        fpvc = os.path.join(
            opvc,
            os.path.split(fpet)[1].split('.nii')[0] + '_PVC' + fcomment + '.nii.gz')
        imio.array2nii(imgpvc[::-1, ::-1, :], B, fpvc, descrip='pvc=iY')
        outdct['fpet'] = fpvc

    return outdct


# =============================================================
# Convert CT units (HU) to PET mu-values
def ct2mu(im):
    '''HU units to 511keV PET mu-values
        https://link.springer.com/content/pdf/10.1007%2Fs00259-002-0796-3.pdf
        C. Burger, et al., PET attenuation coefficients from CT images,
    '''

    # convert nans to -1024 for the HU values only
    im[np.isnan(im)] = -1024
    # constants
    muwater = 0.096
    mubone = 0.172
    rhowater = 0.184
    rhobone = 0.428
    uim = np.zeros(im.shape, dtype=np.float32)
    uim[im <= 0] = muwater * (1 + im[im <= 0] * 1e-3)
    uim[im > 0] = muwater + im[im > 0] * (rhowater * (mubone-muwater) / (1e3 * (rhobone-rhowater)))
    # remove negative values
    uim[uim < 0] = 0
    return uim


# =============================================================


def centre_mass_img(img, output='mm'):
    """
    Calculate the centre of mass of an image along each axes (x,y,z), separately.
    Arguments:
      img: the NIfTI file or image dictionary with the image and header data.
        Outputs the list of the centre of mass for each axis.
    """

    # > check the input image
    if isinstance(img, (str, pathlib.Path)) and os.path.isfile(img):
        imdct = imio.getnii(img, output='all')
    elif isinstance(img, dict) and 'shape' in img:
        imdct = img
    else:
        raise ValueError('unrecognised input image')

    # > initialise centre of mass array in mm and in voxel indexes
    com = np.zeros(3, dtype=np.float32)
    icom = np.zeros(3, dtype=np.float32)
    # > total image sum
    imsum = np.sum(imdct['im'])

    for ind_ax in [-1, -2, -3]:
        # > list of axes
        axs = list(range(imdct['im'].ndim))
        del axs[ind_ax]
        # > indexed centre of mass
        icom[ind_ax] = np.sum(
            np.sum(imdct['im'], axis=tuple(axs)) * np.arange(imdct['shape'][ind_ax])) / imsum
        # > centre of mass in mm (zyx)
        com[ind_ax] = icom[ind_ax] * imdct['voxsize'][ind_ax]

    # > correct due to flipped indexing and world mm
    com[-1] = imdct['shape'][-1] * imdct['voxsize'][-1] - com[-1]
    com[-2] = imdct['shape'][-2] * imdct['voxsize'][-2] - com[-2]
    com[-3] = imdct['shape'][-3] * imdct['voxsize'][-3] - com[-3]

    if output == 'mm':
        return com
    elif output == 'vox':
        return icom
    else:
        raise ValueError('unrecognised output option')


# ==============================================================================


# ==============================================================================
def centre_mass_corr(img, Cnt=None, com=None, flip=None, outpath=None, fcomment='_com-modified',
                     fout=None):
    """
    Image centre of mass correction. The O point is in the middle of the
    image centre of voxel value mass (e.g, radio-activity).
    Arguments:
      img: input image as a NIfTI file or a dictionary of the input image as by
        `nimpa.getnii(path_im, output='all')`.
      com: applying the centre of mass already established.
      flip: flip the image along any dimension (given as tuple)
    """

    # > check the input image
    if isinstance(img, (str, pathlib.Path)) and os.path.isfile(img):
        imdct = imio.getnii(img, output='all')
    elif isinstance(img, dict) and 'shape' in img:
        imdct = img
    else:
        raise ValueError('unrecognised input image')

    # ------------------------------------------------------------------
    # [optional]
    # > applies if requested a radical correction of orientation by flipping
    if flip is not None:
        dimno = len(imdct['im'].shape)
        if dimno == 4:
            imdct['im'] = imdct['im'][:, ::flip[0], ::flip[1], ::flip[2]]
        elif dimno == 3:
            imdct['im'] = imdct['im'][::flip[0], ::flip[1], ::flip[2]]
    # ------------------------------------------------------------------

    # > check if the dictionary of constants is given
    if Cnt is None:
        Cnt = {}

    # > output the centre of mass if image radiodistribution in each dimension in mm.
    if com is None:
        com = centre_mass_img(imdct, output='mm')

    com = np.array(com)

    if not isinstance(com, np.ndarray):
        raise ValueError('The Centre of Mass is not a Numpy array!')

    # > initialise the list of relative NIfTI image CoMs
    com_nii = []

    # > modified affine for the centre of mass
    mA = imdct['affine'].copy()

    # > go through x, y and z
    for i in range(3):
        vox_size = max(imdct['affine'][i, :-1], key=abs)

        # > get the relative centre of mass for each axis (relative to the translation
        # > values in the affine matrix)
        if vox_size > 0:
            com_rel = com[2 - i] + imdct['affine'][i, -1]
        else:
            com_rel = com[2 - i] - abs(vox_size) * imdct['shape'][-i - 1] + imdct['affine'][i, -1]

        mA[i, -1] -= com_rel

        com_nii.append(com_rel)

    log.info('''
        \r relative CoM values are:
        \r {}
        '''.format(com_nii))

    # >------------------------------------------------------
    # > get the file name and path separated
    fsplt = os.path.split(imdct['fim'])

    # > get the output path
    if outpath is not None:
        opth = outpath
        imio.create_dir(opth)
    else:
        opth = fsplt[0]

    opth = pathlib.Path(opth)

    # > get the output file name
    if fout is not None:
        fimc = fout

        if isinstance(fimc, pathlib.Path) and (fimc.suffix != '.gz' or fimc.suffix != '.nii'):
            fimc.with_suffix('.nii.gz')

        elif isinstance(fimc, str) and not fimc.endswith(('.nii', 'nii.gz')):
            fimc = fimc + '.nii.gz'

        fimc = pathlib.Path(fimc)

        if not fimc.parent == '.':
            opth = fimc.parent
            fnm = fimc.name
        else:
            fnm = fimc
    else:
        fnm = fsplt[1].split('.nii')[0] + fcomment + '.nii.gz'

    # save to NIfTI
    innii = nib.load(imdct['fim'])
    # get a new NIfTI image for the perturbed MR
    imdata = innii.get_fdata()
    if flip is not None:
        imdata = imdata[::flip[2], ::flip[1], ::flip[0], ...]
    newnii = nib.Nifti1Image(imdata, mA, innii.header)

    fnew = os.path.join(opth, fnm)
    # save into a new file name for the T1w
    nib.save(newnii, fnew)
    return {'fim': fnew, 'com_rel': com_nii, 'com_abs': com}


# ==============================================================================


def nii_modify(nii, fimout='', outpath='', fcomment='', voxel_range=None):
    '''
    Modify the NIfTI image given either as a file path or a dictionary,
    obtained by nimpa.getnii(file_path).
    '''
    if voxel_range is None:
        voxel_range = []
    if isinstance(nii, str) and os.path.isfile(nii):
        dctnii = imio.getnii(nii, output='all')
        fnii = nii
    if isinstance(nii, dict) and 'im' in nii:
        dctnii = nii
        if 'fim' in dctnii:
            fnii = dctnii['fim']

    # --------------------------------------------------------------------------
    # > output path
    if outpath == '' and fimout != '' and '/' in fimout:
        opth = os.path.dirname(fimout)
        if opth == '' and isinstance(fnii, str) and os.path.isfile(fnii):
            opth = os.path.dirname(nii)
        fimout = os.path.basename(fimout)

    elif outpath == '' and isinstance(fnii, str) and os.path.isfile(fnii):
        opth = os.path.dirname(fnii)
    else:
        opth = outpath
    imio.create_dir(opth)

    # > output floating and affine file names
    if fimout == '':

        if fcomment == '':
            fcomment += '_nimpa-modified'

        fout = os.path.join(opth, os.path.basename(fnii).split('.nii')[0] + fcomment + '.nii.gz')
    else:
        fout = os.path.join(opth, fimout.split('.')[0] + '.nii.gz')
    # --------------------------------------------------------------------------

    # > reduce the max value to 255
    if voxel_range and len(voxel_range) == 1:
        im = voxel_range[0] * dctnii['im'] / np.max(dctnii['im'])
    elif voxel_range and len(voxel_range) == 2:
        # > normalise into range 0-1
        im = (dctnii['im'] - np.min(dctnii['im'])) / np.ptp(dctnii['im'])
        # > convert to voxel_range
        im = voxel_range[0] + im * (voxel_range[1] - voxel_range[0])
    else:
        return None

    # > output file name for the extra reference image
    imio.array2nii(
        im, dctnii['affine'], fout,
        trnsp=(dctnii['transpose'].index(0), dctnii['transpose'].index(1),
               dctnii['transpose'].index(2)), flip=dctnii['flip'])

    return {'fim': fout, 'im': im, 'affine': dctnii['affine']}


# ==============================================================================


# > used for cutting out part of image for face de-identification
def im_cut(im, i_cut, fout=None):
    ''' cut the part of image like the face for de-identification purposes.
        assumes that image is in the form of im[z,y,x].
    '''

    if fout is None:
        save_nii = False
    else:
        save_nii = True

    # > output dictionary
    out = {}

    if isinstance(im, str):
        imdct = imio.getnii(im, output='all')
        img = imdct['im']
        save_nii = True
        fout_s = os.path.split(im)
        fout = os.path.join(fout_s[0], fout_s[1].split('.nii')[0] + '_cut.nii.gz')
        out['fim'] = fout
    elif isinstance(im, (np.ndarray, np.generic)):
        img = im
    else:
        raise ValueError('unrecognised image input')

    # prj = im_project3(img)
    # plot(prj[0]); plot(prj[1]); plot(prj[2])

    img[:, :i_cut, :] = 0

    if save_nii:
        imio.array2nii(
            img, imdct['affine'], fout,
            trnsp=(imdct['transpose'].index(0), imdct['transpose'].index(1),
                   imdct['transpose'].index(2)), flip=imdct['flip'])
        out['fim'] = fout

    out['im'] = img

    return out


# ==============================================================================

#  ____________________________________________________________________________
# |                                                                            |
# |                 M R   B I A S   C O R R E C T I O N                        |
# |____________________________________________________________________________|


def bias_field_correction(fmr, fimout='', outpath='', fcomment='_N4bias', executable='',
                          exe_options=None, sitk_image_mask=True, verbose=False, Cnt=None):
    ''' Correct for bias field in MR image(s) given in <fmr> as a string
        (single file) or as a list of strings (multiple files).

        Output dictionary with the bias corrected file names.

        Options:
        - fimout:       The name (with path) of the output file.  It's
                        ignored when multiple files are given as input.  If
                        given for a single file name, the <outpath> and
                        <fcomment> options are ignored.
        - outpath:      Path to the output folder
        - fcomment:     A prefix comment to the file name
        - executable:   The path to the executable;  if 'sitk' is given instead
                        of the path, the Python module SimpleITK will be
                        used if it is available.
        - exe_options:  Options for the executable in the form of a list of
                        strings.
        - sitk_image_mask:  Image masking will be used if SimpleITK is
                            chosen.
    '''
    if exe_options is None:
        exe_options = []
    if Cnt is None:
        Cnt = {}

    if executable == 'sitk' and 'SimpleITK' not in sys.modules and not sitk_flag:
        raise ImportError(
            dedent('''\
            If SimpleITK module is required for bias correction, it needs to be
            first installed using this command:
            conda install -c simpleitk simpleitk
            or pip install SimpleITK'''))

    # --------------------------------------------------------------------------
    # INPUT
    # --------------------------------------------------------------------------
    # > path to a single file
    if isinstance(fmr, (str, pathlib.PurePath)) and os.path.isfile(fmr):
        fins = [fmr]

    # > list of file paths
    elif isinstance(fmr, list) and all(map(os.path.isfile, fmr)):
        fins = fmr
        log.info('multiple input files => ignoring the single output file name.')
        fimout = ''

    # > path to a folder
    elif isinstance(fmr, (str, pathlib.PurePath)) and os.path.isdir(fmr):
        fins = [os.path.join(fmr, f) for f in os.listdir(fmr) if f.endswith(('.nii', '.nii.gz'))]
        log.info('multiple input files from input folder.')
        fimout = ''

    else:
        raise ValueError('could not decode the input of floating images.')
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------------------------
    # > output path
    if outpath == '' and fimout != '':
        opth = os.path.dirname(fimout)
        if opth == '':
            opth = os.path.dirname(fmr)
            fimout = os.path.join(opth, fimout)
        n4opth = opth
        fcomment = ''
    elif outpath == '':
        opth = os.path.dirname(fmr)
        # > N4 bias correction specific folder
        n4opth = os.path.join(opth, 'N4bias')
    else:
        opth = outpath
        # > N4 bias correction specific folder
        n4opth = os.path.join(opth, 'N4bias')

    imio.create_dir(n4opth)

    outdct = {}
    # --------------------------------------------------------------------------

    for fin in fins:
        log.debug('input for bias correction:\n{}'.format(fin))

        if fimout == '':
            # split path
            fspl = os.path.split(fin)

            # N4 bias correction file output paths
            fn4 = os.path.join(n4opth, fspl[1].split('.nii')[0] + fcomment + '.nii.gz')
        else:
            fn4 = fimout

        if not os.path.exists(fn4):
            if executable == 'sitk':
                # =============================================
                # SimpleITK Bias field correction for T1 and T2
                # =============================================
                # > initialise the corrector
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                # numberFilltingLevels = 4

                # read input file
                im = sitk.ReadImage(str(fin))

                # > create a object specific mask
                fmsk = os.path.join(n4opth, fspl[1].split('.nii')[0] + '_sitk_mask.nii.gz')
                msk = sitk.OtsuThreshold(im, 0, 1, 200)
                sitk.WriteImage(msk, fmsk)

                # > cast to 32-bit float
                im = sitk.Cast(im, sitk.sitkFloat32)

                # ------------------------------------------
                log.info('correcting bias field for {}'.format(fin))
                n4out = corrector.Execute(im, msk)
                sitk.WriteImage(n4out, fn4)
                # ------------------------------------------
                if sitk_image_mask:
                    outdct.setdefault('fmsk', [])
                    outdct['fmsk'].append(fmsk)

            elif os.path.basename(executable) == 'N4BiasFieldCorrection' and os.path.isfile(
                    executable):
                cmd = [executable, '-i', fin, '-o', fn4]
                if verbose and os.path.basename(executable) == 'N4BiasFieldCorrection':
                    cmd.extend(['-v', '1'])
                cmd.extend(exe_options)
                run(cmd)
                if 'command' not in outdct:
                    outdct['command'] = []
                outdct['command'].append(cmd)
            elif os.path.isfile(executable):
                cmd = [executable]
                cmd.extend(exe_options)
                run(cmd)
                if 'command' not in outdct:
                    outdct['command'] = cmd
        else:
            log.info('N4 bias corrected file seems already existing.')

        # > output to dictionary
        outdct.setdefault('fim', [])
        outdct['fim'].append(fn4)

    if len(outdct['fim']) == 1:
        outdct['fim'] = outdct['fim'][0]
    return outdct


# ------------------------------------------------------------------------------------
def pet2pet_rigid(fref, fflo, Cnt, outpath='', rmsk=True, rfwhm=15., rthrsh=0.05, pi=50, pv=50,
                  smof=0, smor=0):

    # create a folder for PET images registered to ref PET
    if outpath == '':
        outpath = os.path.dirname(fflo)
    odir = os.path.join(outpath, 'PET2PET')
    imio.create_dir(odir)

    if rmsk:
        fimdir = os.path.join(odir, 'tmp')
        imio.create_dir(fimdir)
        fmsk = os.path.join(fimdir, 'rmask.nii.gz')
        imdct = imio.getnii(fref, output='all')
        smoim = ndi.filters.gaussian_filter(imdct['im'],
                                            imio.fwhm2sig(rfwhm, voxsize=imdct['affine'][0, 0]),
                                            mode='mirror')
        thrsh = rthrsh * smoim.max()
        immsk = np.int8(smoim > thrsh)
        for iz in range(immsk.shape[0]):
            for iy in range(immsk.shape[1]):
                ix0 = np.argmax(immsk[iz, iy, :] > 0)
                ix1 = immsk.shape[2] - np.argmax(immsk[iz, iy, ::-1] > 0)
                if (ix1 - ix0) > immsk.shape[2] - 10: continue
                immsk[iz, iy, ix0:ix1] = 1
        imio.array2nii(immsk[::-1, ::-1, :], imio.getnii_affine(fref), fmsk)

    # output in register with ref PET
    fout = os.path.join(odir, 'PET-r-to-' + os.path.basename(fref).split('.')[0] + '.nii.gz')
    # text file for the affine transform T1w->PET
    faff = os.path.join(odir, 'affine-PET-r-to-' + os.path.basename(fref).split('.')[0] + '.txt')
    # call the registration routine
    if os.path.isfile(Cnt['REGPATH']):
        cmd = [
            Cnt['REGPATH'], '-ref', fref, '-flo', fflo, '-rigOnly', '-speeeeed', '-aff', faff,
            '-pi',
            str(pi), '-pv',
            str(pv), '-smooF',
            str(smof), '-smooR',
            str(smor), '-res', fout]
        if rmsk:
            cmd.append('-rmask')
            cmd.append(fmsk)
        if log.getEffectiveLevel() >= logging.INFO:
            cmd.append('-voff')
        log.info('Executing command:\n{}'.format(cmd))
        run(cmd)
    else:
        log.error('path to registration executable is incorrect!')
        raise Exception('No registration executable found')

    return faff, fout


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def mr2pet_rigid(fpet, mridct, Cnt, outpath='', fcomment='', rmsk=True, rfwhm=15., rthrsh=0.05,
                 fmsk=True, ffwhm=15., fthrsh=0.05, pi=50, pv=50, smof=0, smor=0):
    # create output path if given
    if outpath != '':
        imio.create_dir(outpath)

    # --- MR T1w
    if 'T1nii' in mridct and os.path.isfile(mridct['T1nii']):
        ft1w = mridct['T1nii']
    elif 'T1bc' in mridct and os.path.isfile(mridct['T1bc']):
        ft1w = mridct['T1bc']
    elif 'T1DCM' in mridct and os.path.exists(mridct['MRT1W']):
        ft1w = imio.dcm2nii(mridct['T1nii'], 'converted', executable=Cnt.get('DCM2NIIX', None))
    else:
        raise ValueError('disaster: no T1w image!')

    # create a folder for MR images registered to PET
    if outpath != '':
        mrodir = os.path.join(outpath, 'T1w2PET')
        fimdir = os.path.join(outpath, os.path.join('T1w2PET', 'tmp'))
    else:
        mrodir = os.path.join(os.path.dirname(ft1w), 'mr2pet')
        fimdir = os.path.join(os.path.basename(ft1w), 'tmp')
    imio.create_dir(mrodir)
    imio.create_dir(fimdir)

    if rmsk:
        f_rmsk = os.path.join(fimdir, 'rmask.nii.gz')
        imdct = imio.getnii(fpet, output='all')
        smoim = ndi.filters.gaussian_filter(
            imdct['im'], imio.fwhm2sig(rfwhm, voxsize=abs(imdct['affine'][0, 0])), mode='mirror')
        thrsh = rthrsh * smoim.max()
        immsk = np.int8(smoim > thrsh)
        immsk = regseg.imfill(immsk)
        imio.array2nii(immsk[::-1, ::-1, :], imdct['affine'], f_rmsk)
    if fmsk:
        f_fmsk = os.path.join(fimdir, 'fmask.nii.gz')
        imdct = imio.getnii(ft1w, output='all')
        smoim = ndi.filters.gaussian_filter(
            imdct['im'], imio.fwhm2sig(ffwhm, voxsize=abs(imdct['affine'][0, 0])), mode='mirror')
        thrsh = fthrsh * smoim.max()
        immsk = np.int8(smoim > thrsh)
        immsk = regseg.imfill(immsk)
        imio.array2nii(immsk[::-1, ::-1, :], imdct['affine'], f_fmsk)

    # if provided, separate the comment with underscore
    if fcomment != '': fcomment = '_' + fcomment
    # output for the T1w in register with PET
    ft1out = os.path.join(
        mrodir, 'T1w-r-to-' + os.path.basename(fpet).split('.')[0] + fcomment + '.nii.gz')
    # text file for the affine transform T1w->PET
    faff = os.path.join(
        mrodir, 'affine-T1w-r-to-' + os.path.basename(fpet).split('.')[0] + fcomment + '.txt')
    # call the registration routine
    if os.path.isfile(Cnt['REGPATH']):
        cmd = [
            Cnt['REGPATH'], '-ref', fpet, '-flo', ft1w, '-rigOnly', '-speeeeed', '-aff', faff,
            '-pi',
            str(pi), '-pv',
            str(pv), '-smooF',
            str(smof), '-smooR',
            str(smor), '-res', ft1out]
        if rmsk:
            cmd.append('-rmask')
            cmd.append(f_rmsk)
        if fmsk:
            cmd.append('-fmask')
            cmd.append(f_fmsk)
        if not Cnt.get('VERBOSE', True):
            cmd.append('-voff')
        log.info('Executing command:\n{}'.format(cmd))
        run(cmd)
    else:
        raise IOError('path to registration executable is incorrect!')

    return faff
