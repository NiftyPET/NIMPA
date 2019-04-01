#!/usr/bin/env python
"""initialise the NIMPA package (part of NiftyPET package)"""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018 Pawel Markiewicz @ University College London"
#------------------------------------------------------------------------------

import os
import sys
import platform

# if using conda put the resources in the folder with the environment name
if 'CONDA_DEFAULT_ENV' in os.environ:
	env = os.environ['CONDA_DEFAULT_ENV']
	print 'i> conda environment found:', env
else:
	env = ''
# create the path for the resources files according to the OS platform
if platform.system() in ['Linux', 'Darwin']:
	path_resources = os.path.join( os.path.join(os.path.expanduser('~'),   '.niftypet'), env )
elif platform.system() == 'Windows' :
	path_resources = os.path.join( os.path.join(os.getenv('LOCALAPPDATA'), '.niftypet'), env )
else:
	print 'e> unrecognised operating system!'
	
sys.path.append(path_resources)
try:
    import resources
except ImportError as ie:
    print '----------------------------'
    print 'e> Import Error: NiftyPET''s resources file <resources.py> could not be imported.  It should be in ''~/.niftypet/resources.py'' (Linux) or ''//Users//USERNAME//AppData//Local//niftypet//resources.py'' (Windows) but likely it does not exists.'
    print '----------------------------'
#===========================

if platform.system() in ['Linux', 'Windows']:
	from dinf import gpuinfo, dev_info
	
from prc import trimim, iyang, pvc_iyang, psf_general, psf_measured
from prc import coreg_spm, coreg_vinci, resample_spm, resample_vinci
from prc import affine_fsl, resample_fsl
from prc import affine_niftyreg, resample_niftyreg, pet2pet_rigid
from prc import create_dir, time_stamp, fwhm2sig, getnii, getnii_descr, array2nii, dcm2im
from prc import orientnii, nii_ugzip, nii_gzip, niisort, dcmsort, dcminfo, dcmanonym

from prc import dice_coeff, dice_coeff_multiclass
from prc import imfill, create_mask
from prc import correct_bias_n4, pick_t1w

from prc import motion_reg

from prc import ct2mu
from prc import nii_modify

from prc import dcm2nii

from img import create_disk, profile_points
