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

from tqdm.auto import tqdm

from niftypet.ninst.tools import LogHandler
from niftypet.ninst import cudasetup as cs

log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# log.addHandler(LogHandler())  # do it anyway for convenience

path_resources = cs.path_niftypet_local()
resources = cs.get_resources()

# if getattr(resources, "CC_ARCH", "") and platform.system() in ['Linux', 'Windows']:
from niftypet.ninst.dinf import gpuinfo, dev_info

from .prc import imsmooth
from .prc import imtrimup
from .prc import imtrimup as trimim #for backward compatibility
from .prc import iyang, pvc_iyang, psf_general, psf_measured

from .prc import realign_mltp_spm, resample_mltp_spm
from .prc import coreg_spm, coreg_vinci, resample_spm, resample_vinci
from .prc import affine_fsl, resample_fsl
from .prc import affine_niftyreg, resample_niftyreg, pet2pet_rigid
from .prc import create_dir, time_stamp, fwhm2sig, getnii, getnii_descr, array2nii, dcm2im
from .prc import orientnii, nii_ugzip, nii_gzip, niisort, dcmsort, dcminfo, dcmanonym

from .prc import dice_coeff, dice_coeff_multiclass
from .prc import imfill, create_mask, centre_mass_img
from .prc import bias_field_correction
from .prc import pick_t1w

from .prc import motion_reg

from .prc import ct2mu
from .prc import nii_modify

from .prc import dcm2nii

from .prc import im_cut

from .img import create_disk, profile_points
from .img import imdiff, imscroll
