#!/usr/bin/env python
"""initialise the NiftyPET NIMPA package"""
__author__      = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__   = "Copyright 2020"

import logging
import os
import platform
import re
import sys
from textwrap import dedent

from pkg_resources import resource_filename
from tqdm.auto import tqdm

from niftypet.ninst import cudasetup as cs

# if getattr(resources, "CC_ARCH", "") and platform.system() in ['Linux', 'Windows']:
from niftypet.ninst.dinf import dev_info, gpuinfo
from niftypet.ninst.tools import LOG_FORMAT, LogHandler, path_resources, resources

from .img import create_disk, imdiff, imscroll, profile_points
from .prc import imtrimup  # for backward compatibility
from .prc import (
    affine_fsl,
    affine_niftyreg,
    array2nii,
    bias_field_correction,
    centre_mass_img,
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
    iyang,
    motion_reg,
    nii_gzip,
    nii_modify,
    nii_ugzip,
    niisort,
    orientnii,
    pet2pet_rigid,
    pick_t1w,
    psf_general,
    psf_measured,
    pvc_iyang,
    realign_mltp_spm,
    resample_fsl,
    resample_mltp_spm,
    resample_niftyreg,
    resample_spm,
    resample_vinci,
    time_stamp,
)

# log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# log.addHandler(LogHandler())  # do it anyway for convenience

# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = resource_filename(__name__, "cmake")
