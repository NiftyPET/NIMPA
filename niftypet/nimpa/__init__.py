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


class LogHandler(logging.StreamHandler):
    """Custom formatting and tqdm-compatibility"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fmt = logging.Formatter(
            '%(levelname)s:%(asctime)s:%(name)s:%(funcName)s\n> %(message)s')
        self.setFormatter(fmt)

    def handleError(self, record):
        super().handleError(record)
        raise IOError(record)

    def emit(self, record):
        """Write to tqdm's stream so as to not break progress-bars"""
        try:
            msg = self.format(record)
            tqdm.write(
                msg, file=self.stream, end=getattr(self, "terminator", "\n"))
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


log = logging.getLogger(__name__)
# technically bad practice to add handlers
# https://docs.python.org/3/howto/logging.html#library-config
# but we'll do it anyway for convenience
log.addHandler(LogHandler())

# if using conda put the resources in the folder with the environment name
if 'CONDA_DEFAULT_ENV' in os.environ:
    try:
        env = re.findall('envs/(.*)/bin/python', sys.executable)[0]
    except IndexError:
        env = os.environ['CONDA_DEFAULT_ENV']
    log.info('conda environment found:' + env)
else:
    env = ''

# create the path for the resources files according to the OS platform
if platform.system() in ['Linux', 'Darwin']:
    path_resources = os.path.join( os.path.join(os.path.expanduser('~'),   '.niftypet'), env )
elif platform.system() == 'Windows':
    path_resources = os.path.join( os.path.join(os.getenv('LOCALAPPDATA'), '.niftypet'), env )
else:
    log.error('unrecognised operating system!')

sys.path.insert(1,path_resources)
try:
    import resources
except ImportError as ie:
    raise ImportError(dedent('''\
        --------------------------------------------------------------------------
        NiftyPET resources file <resources.py> could not be imported.
        It should be in ~/.niftypet/resources.py (Linux) or
        in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
        but likely it does not exists.
        --------------------------------------------------------------------------'''))

if resources.CC_ARCH != '' and platform.system() in ['Linux', 'Windows']:
    from .dinf import gpuinfo, dev_info

from .prc import trimim, iyang, pvc_iyang, psf_general, psf_measured
from .prc import smoothim

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
