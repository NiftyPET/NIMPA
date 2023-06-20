"""
Functions to help imaging with the PET/MR GE Signa scanner
"""

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2023"

import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from ..prc import imio

SZ_VXZ = 2.78      # Signa default axial voxel size


def pifa2nii(fpifa, fnii=None, outpath=None):
    ''' Convert the GE PIFA file format to NIfTI. For generating PIFA
        file in the same space as the PET reconstructed image, a NIfTI
        file of the reconstruction needs to be provided as `fnii`.

        The NIfTI file needs to represent the whole FOV of 60 cm.
    '''

    fpifa = Path(fpifa)
    if not fpifa.is_file():
        raise FileNotFoundError('PIFA file could not be found')

    if fnii is not None and Path(fnii).is_file():
        affine = imio.getnii(fnii, output='affine')
    else:
        affine = None

    # > output folder
    if outpath is None:
        pifadir = fpifa.parent
    else:
        pifadir = Path(outpath)
    imio.create_dir(pifadir)

    # > read the HDF5 file
    fh = h5py.File(fpifa, 'r')

    # > Diameter of the transaxial FOV
    DFOV = fh['HeaderData/ctacDfov'][0]

    # > PIFA x-y voxel size
    SZ_IMX = fh['HeaderData/xMatrix'][0]
    SZ_IMZ = fh['HeaderData/zMatrix'][0]
    # ZLOCAT = fh['HeaderData/tableLocation'][0] # << check if this is really correct
    SP_VXY = DFOV / SZ_IMX

    # > the mu-map in units of 1/mm
    mu_dat = np.array(fh['PifaData'])

    # > transpose the data for NIfTI output
    data = np.transpose(mu_dat[:, ::-1, :], (2, 1, 0))

    # > if affine is not provided through the NIfTI
    if affine is None:
        # > affine matrix
        A = np.eye(4)
        A[0, 0] = -SP_VXY
        A[1, 1] = SP_VXY
        A[2, 2] = SZ_VXZ

        A[0, 3] = DFOV/2 - SP_VXY/2
        A[1, 3] = -(DFOV/2 - SP_VXY/2)
        A[2, 3] = -(SZ_IMZ/2*SZ_VXZ - SZ_VXZ/2)
        A[3, 3] = 1
    else:
        # A = np.eye(4)
        # A[0,0] = -SP_VXY
        # A[1,1] = SP_VXY
        # A[2,2] = SZ_VXZ

        # A[0,3] = affine[0,3]
        # A[1,3] = affine[1,3]
        # A[2,3] = affine[2,3]
        # A[3,3] = 1
        A = affine

    img = nib.Nifti1Image(data, A)
    fout = pifadir / (fpifa.name.split('.pifa')[0] + '.nii.gz')
    img.to_filename(fout)

    return fout


def nii2pifa(fnii, fpifa, outpath=None, bed_mask_thresh=0.2):
    '''
    Convert a newly generated PIFA NIfTI `fnii` file to the
    GE PIFA file format using the original PIFA file `fpifa`.
    '''

    fpifa = Path(fpifa)
    if 'pifaIvv_' in fpifa.name:
        fpifa_ivv = fpifa
        fpifa = fpifa.parent / fpifa.name.replace('pifaIvv_', 'pifa_')
    elif 'pifa_' in fpifa.name:
        fpifa_ivv = fpifa.parent / fpifa.name.replace('pifa_', 'pifaIvv_')
    else:
        raise FileNotFoundError('PIFA files are unrecognised')

    # > output folder
    pifadir = fpifa.parent if outpath is None else Path(outpath)
    imio.create_dir(pifadir)

    # > new PIFA
    fpifa_n = pifadir / fpifa.name
    fpifa_ivv_n = pifadir / fpifa_ivv.name
    shutil.copyfile(fpifa, fpifa_n)
    shutil.copyfile(fpifa_ivv, fpifa_ivv_n)

    with h5py.File(fpifa_n, 'r+') as fh, h5py.File(fpifa_ivv_n, 'r+') as fh_ivv:
        # > get all the components of original PIFA
        pifa = np.array(fh['PifaData'])
        pifa_ivv = np.array(fh_ivv['PifaData'])
        bed = pifa - pifa_ivv
        msk_bed = bed > bed_mask_thresh * np.max(bed)

        # > make the new mu-map to PIFA
        newpifa_ivv = 0.1 * imio.getnii(fnii)
        # > flip the z-axis (in GE systems it's the other way round)
        # > and set to zero the voxels which belong to the bed/table
        newpifa_ivv = newpifa_ivv[::-1, ...] * ~msk_bed

        # > blend the bed and object
        newpifa = np.maximum(bed, newpifa_ivv)

        fh['PifaData'][...] = newpifa
        fh_ivv['PifaData'][...] = newpifa_ivv

    # > check if edited:
    with h5py.File(fpifa_n, 'r') as fcheck:
        dcheck = fcheck['PifaData'][...]
    if not np.allclose(dcheck, newpifa):
        raise ValueError('the CT modification did not work')

    with h5py.File(fpifa_ivv_n, 'r') as fcheck:
        dcheck = fcheck['PifaData'][...]
    if not np.allclose(dcheck, newpifa_ivv):
        raise ValueError('the CT modification did not work')

    return {'fpifa': fpifa_n, 'fpifa_ivv': fpifa_ivv_n, 'pifa': newpifa, 'opifa': pifa}
