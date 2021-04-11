# initialise the module folder
__all__ = [
    # imio
    'array2nii', 'create_dir', 'dcm2im', 'dcm2nii', 'dcmanonym', 'dcminfo',
    'dcmsort', 'fwhm2sig', 'getnii', 'getnii_descr', 'nii_gzip', 'nii_ugzip',
    'niisort', 'orientnii', 'pick_t1w', 'time_stamp',
    # prc
    'bias_field_correction', 'centre_mass_img', 'ct2mu', 'im_cut', 'imsmooth',
    'imtrimup', 'iyang', 'nii_modify', 'pet2pet_rigid', 'psf_gaussian',
    'psf_measured', 'pvc_iyang',
    # regseg
    'affine_fsl', 'affine_niftyreg', 'coreg_spm', 'coreg_vinci', 'create_mask',
    'dice_coeff', 'dice_coeff_multiclass', 'imfill', 'motion_reg',
    'realign_mltp_spm', 'resample_fsl', 'resample_mltp_spm',
    'resample_niftyreg', 'resample_spm', 'resample_vinci'] # yapf: disable

from .imio import (
    array2nii,
    create_dir,
    dcm2im,
    dcm2nii,
    dcmanonym,
    dcminfo,
    dcmsort,
    fwhm2sig,
    getnii,
    getnii_descr,
    nii_gzip,
    nii_ugzip,
    niisort,
    orientnii,
    pick_t1w,
    time_stamp,
)

# will be deprecated
from .prc import (
    bias_field_correction,
    centre_mass_img,
    ct2mu,
    im_cut,
    imsmooth,
    imtrimup,
    iyang,
    nii_modify,
    pet2pet_rigid,
    psf_gaussian,
    psf_measured,
    pvc_iyang,
)
from .regseg import (
    affine_fsl,
    affine_niftyreg,
    coreg_spm,
    coreg_vinci,
    create_mask,
    dice_coeff,
    dice_coeff_multiclass,
    imfill,
    motion_reg,
    realign_mltp_spm,
    resample_fsl,
    resample_mltp_spm,
    resample_niftyreg,
    resample_spm,
    resample_vinci,
)
