#!/usr/bin/env python
"""initialise the NiftyPET NIMPA package"""
__author__ = "Pawel J. Markiewicz", "Casper O. da Costa-Luis"
__copyright__ = "Copyright 2021"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"
__all__ = [
    # gpu utils
    'cs', 'dev_info', 'gpuinfo',
    # utils
    'LOG_FORMAT', 'LogHandler',
    # config
    'path_resources', 'resources', 'cmake_prefix',
    # numcu
    'add', 'div', 'mul',
    # improc
    'conv_separable', 'isub', 'nlm', 'aff_dist', 'centre_mass_rel',
    # core
    'create_disk', 'get_cylinder', 'imdiff', 'imscroll', 'profile_points', 'imtrimup',
    'affine_fsl', 'affine_dipy', 'affine_niftyreg',
    'array2nii', 'bias_field_correction',
    'centre_mass_img', 'centre_mass_corr', 'coreg_spm', 'coreg_vinci',
    'create_dir', 'create_mask', 'ct2mu',
    'dcm2im', 'dcm2nii', 'dcmanonym', 'dcminfo', 'dcmsort', 'isdcm', 'dcmdir',
    'dice_coeff', 'dice_coeff_multiclass', 'fwhm2sig', 'getmgh', 'getnii', 'mgh2nii',
    'getnii_descr', 'im_cut', 'imfill', 'imsmooth', 'iyang', 'motion_reg', 'nii_gzip',
    'nii_modify', 'nii_ugzip', 'niisort', 'orientnii', 'pet2pet_rigid', 'pick_t1w',
    'psf_gaussian', 'psf_measured', 'pvc_iyang', 'realign_mltp_spm', 'resample_fsl',
    'resample_mltp_spm', 'resample_niftyreg', 'resample_spm', 'resample_vinci', 'resample_dipy',
    'time_stamp', 'rem_chars',
    # Signa
    'pifa2nii', 'nii2pifa',
    # ACR
    'acr'
    # 'get_params', 'get_paths', 'extract_reso_part', 'sampling_masks'
    # 'create_mumap_core', 'create_nac_core', 'create_reso', 'create_sampl_reso', 'create_sampl',
    # 'standard_analysis', 'estimate_fwhm'
    ] # yapf: disable

from os import fspath

try:          # py<3.9
    import importlib_resources as iresources
except ImportError:
    from importlib import resources as iresources

try:
    from numcu import add, div, mul
except ImportError:
    pass

from niftypet.ninst import cudasetup as cs
from niftypet.ninst.dinf import dev_info, gpuinfo
from niftypet.ninst.tools import LOG_FORMAT, LogHandler, path_resources, resources

from . import acr
from .img import create_disk, get_cylinder, imdiff, imscroll, nii2pifa, pifa2nii, profile_points
from .prc import imtrimup  # for backward compatibility
from .prc import (
    aff_dist,
    affine_dipy,
    affine_fsl,
    affine_niftyreg,
    array2nii,
    bias_field_correction,
    centre_mass_corr,
    centre_mass_img,
    centre_mass_rel,
    conv_separable,
    coreg_spm,
    coreg_vinci,
    create_dir,
    create_mask,
    ct2mu,
    dcm2im,
    dcm2nii,
    dcmanonym,
    dcmdir,
    dcminfo,
    dcmsort,
    dice_coeff,
    dice_coeff_multiclass,
    fwhm2sig,
    getmgh,
    getnii,
    getnii_descr,
    im_cut,
    imfill,
    imsmooth,
    isdcm,
    isub,
    iyang,
    mgh2nii,
    motion_reg,
    nii_gzip,
    nii_modify,
    nii_ugzip,
    niisort,
    nlm,
    orientnii,
    pet2pet_rigid,
    pick_t1w,
    psf_gaussian,
    psf_measured,
    pvc_iyang,
    realign_mltp_spm,
    rem_chars,
    resample_dipy,
    resample_fsl,
    resample_mltp_spm,
    resample_niftyreg,
    resample_spm,
    resample_vinci,
    time_stamp,
)

# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = fspath(iresources.files("niftypet.nimpa").resolve() / "cmake")
