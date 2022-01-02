# initialise the module folder
__all__ = [
    # imio
    'array2nii', 'create_dir', 'dcm2im', 'dcm2nii', 'dcmanonym', 'dcminfo', 'dcmsort', 'fwhm2sig',
    'getnii', 'getnii_descr', 'nii_gzip', 'nii_ugzip', 'niisort', 'orientnii', 'pick_t1w',
    'time_stamp',
    # prc
    'bias_field_correction', 'centre_mass_img', 'centre_mass_corr', 'ct2mu', 'im_cut',
    'imsmooth', 'imtrimup',
    'iyang', 'nii_modify', 'pet2pet_rigid', 'psf_gaussian', 'psf_measured', 'pvc_iyang',
    # improc
    'conv_separable', 'div', 'isub', 'mul', 'nlm',
    # regseg
    'affine_fsl', 'affine_niftyreg', 'coreg_spm', 'coreg_vinci', 'create_mask', 'dice_coeff',
    'dice_coeff_multiclass', 'imfill', 'motion_reg', 'realign_mltp_spm', 'resample_fsl',
    'resample_dipy', 'resample_mltp_spm', 'resample_niftyreg', 'resample_spm',
    'resample_vinci'] # yapf: disable

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
    centre_mass_corr,
    centre_mass_img,
    conv_separable,
    ct2mu,
    div,
    im_cut,
    imsmooth,
    imtrimup,
    isub,
    iyang,
    mul,
    nii_modify,
    nlm,
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
    resample_dipy,
    resample_fsl,
    resample_mltp_spm,
    resample_niftyreg,
    resample_spm,
    resample_vinci,
)
