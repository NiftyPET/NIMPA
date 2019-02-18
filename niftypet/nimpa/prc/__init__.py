# initialise the module folder

from prc import trimim, iyang, pvc_iyang, psf_general, psf_measured
from prc import ct2mu
from prc import nii_modify

from imio import create_dir, time_stamp, fwhm2sig, getnii, getnii_descr, array2nii
from imio import orientnii, nii_ugzip, nii_gzip, dcmsort, niisort, dcm2im, dcminfo, dcmanonym

from regseg import coreg_spm, resample_spm, affine_niftyreg, resample_niftyreg
from regseg import coreg_vinci, resample_vinci
from regseg import motion_reg
from regseg import dice_coeff, dice_coeff_multiclass
from regseg import imfill, create_mask
from regseg import correct_bias_n4

# will be depreciated
from prc import affine_fsl, resample_fsl
from prc import reg_mr2pet, pet2pet_rigid
