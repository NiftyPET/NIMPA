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
    'conv_separable', 'isub', 'nlm',
    # core
    'create_disk', 'imdiff', 'imscroll', 'profile_points', 'imtrimup',
    'affine_fsl', 'affine_dipy', 'affine_niftyreg',
    'array2nii', 'bias_field_correction',
    'centre_mass_img', 'centre_mass_corr', 'coreg_spm', 'coreg_vinci',
    'create_dir', 'create_mask', 'ct2mu',
    'dcm2im', 'dcm2nii', 'dcmanonym', 'dcminfo', 'dcmsort',
    'dice_coeff', 'dice_coeff_multiclass', 'fwhm2sig', 'getnii',
    'getnii_descr', 'im_cut', 'imfill', 'imsmooth', 'iyang', 'motion_reg', 'nii_gzip',
    'nii_modify', 'nii_ugzip', 'niisort', 'orientnii', 'pet2pet_rigid', 'pick_t1w',
    'psf_gaussian', 'psf_measured', 'pvc_iyang', 'realign_mltp_spm', 'resample_fsl',
    'resample_mltp_spm', 'resample_niftyreg', 'resample_spm', 'resample_vinci', 'resample_dipy',
    'time_stamp'] # yapf: disable

from numcu import add, div, mul
from pkg_resources import resource_filename

from niftypet.ninst import cudasetup as cs
from niftypet.ninst.dinf import dev_info, gpuinfo
from niftypet.ninst.tools import LOG_FORMAT, LogHandler, path_resources, resources

from .img import create_disk, imdiff, imscroll, profile_points
from .prc import imtrimup  # for backward compatibility
from .prc import (
    affine_dipy,
    affine_fsl,
    affine_niftyreg,
    array2nii,
    bias_field_correction,
    centre_mass_corr,
    centre_mass_img,
    conv_separable,
    coreg_spm,
    coreg_vinci,
    create_dir,
    create_mask,
    ct2mu,
    dcm2im,
    dcm2nii,
    dcmanonym,
    dcminfo,
    dcmsort,
    dice_coeff,
    dice_coeff_multiclass,
    fwhm2sig,
    getnii,
    getnii_descr,
    im_cut,
    imfill,
    imsmooth,
    isub,
    iyang,
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
    resample_dipy,
    resample_fsl,
    resample_mltp_spm,
    resample_niftyreg,
    resample_spm,
    resample_vinci,
    time_stamp,
)

# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = resource_filename(__name__, "cmake")
