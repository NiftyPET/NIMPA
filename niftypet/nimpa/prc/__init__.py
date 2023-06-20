# initialise the module folder
__all__ = [
    # imio
    'array2nii', 'create_dir', 'dcm2im', 'dcm2nii', 'dcmanonym', 'dcminfo', 'dcmsort', 'fwhm2sig',
    'mgh2nii', 'getmgh', 'getnii', 'getnii_descr', 'nii_gzip', 'nii_ugzip', 'niisort', 'orientnii',
    'pick_t1w', 'time_stamp', 'rem_chars', 'isdcm', 'dcmdir',
    # prc
    'bias_field_correction', 'centre_mass_img', 'centre_mass_rel', 'centre_mass_corr', 'ct2mu',
    'im_cut', 'imsmooth', 'imtrimup',
    'iyang', 'nii_modify', 'pet2pet_rigid', 'psf_gaussian', 'psf_measured', 'pvc_iyang',
    # num
    'conv_separable', 'isub', 'nlm',
    # regseg
    'aff_dist', 'affine_dipy', 'affine_fsl', 'affine_niftyreg',
    'coreg_spm', 'coreg_vinci', 'create_mask', 'dice_coeff', 'dice_coeff_multiclass',
    'imfill', 'motion_reg', 'realign_mltp_spm', 'resample_fsl', 'resample_dipy',
    'resample_mltp_spm', 'resample_niftyreg', 'resample_spm',
    'resample_vinci'] # yapf: disable

from .imio import (
    array2nii,
    create_dir,
    dcm2im,
    dcm2nii,
    dcmanonym,
    dcmdir,
    dcminfo,
    dcmsort,
    fwhm2sig,
    getmgh,
    getnii,
    getnii_descr,
    isdcm,
    mgh2nii,
    nii_gzip,
    nii_ugzip,
    niisort,
    orientnii,
    pick_t1w,
    rem_chars,
    time_stamp,
)
from .num import conv_separable, isub, nlm

# will be deprecated
from .prc import (
    bias_field_correction,
    centre_mass_corr,
    centre_mass_img,
    centre_mass_rel,
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
    aff_dist,
    affine_dipy,
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
    resample_dipy,
    resample_fsl,
    resample_mltp_spm,
    resample_niftyreg,
    resample_spm,
    resample_vinci,
)
