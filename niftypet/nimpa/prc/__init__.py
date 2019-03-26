# initialise the module folder

from prc import trimim, iyang, pvc_iyang, psf_general, psf_measured
from prc import ct2mu
from prc import nii_modify
from prc import correct_bias_n4

from imio import create_dir, time_stamp, fwhm2sig, getnii, getnii_descr, array2nii
from imio import orientnii, nii_ugzip, nii_gzip, dcmsort, niisort, dcm2im, dcminfo
from imio import dcmanonym, pick_t1w, dcm2nii

from regseg import coreg_spm, resample_spm, affine_niftyreg, resample_niftyreg
from regseg import coreg_vinci, resample_vinci
from regseg import motion_reg
from regseg import dice_coeff, dice_coeff_multiclass
from regseg import imfill, create_mask

from regseg import affine_fsl, resample_fsl

# will be depreciated
from prc import pet2pet_rigid
