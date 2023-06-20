"""ACR/Jaszczak PET phantom design parameters"""
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2021-23"

import os
from pathlib import Path

import numpy as np

try:          # py<3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources

# CONSTANTS/PARAMTERS

# smoothing of the templates before registration
fwhm_tmpl = 3.

# smoothing kernel FWHM for the NAC and QNT PET
fwhm_nac = 3.
fwhm_qnt = 3.

# number of iterations for NAC and QNT PET reconstruction
itr_nac = 1
itr_qnt = 4
itr_qnt2 = [4, 8, 16]

# image trimming upsampling scaling factor
sclt = 4

# and interpolation (1: linear)
interp = 1

# transaxial voxel size of the PNG images: 300dpi => 0.08467 mm
vxysz = 25.4 / 300
# target voxel size
vxsz = 0.2

# scaling factor for images from PNG image resolution
scl = vxysz / vxsz

# second level image scaling factor
scld = 2

# interpolation order (0-NN, 1-trilinear)
intord = 0

# padding for the design arrays (gives more breathing space for smoothing in transaxial views)
pad = 4

# angle of rotations for the initial position of resolution rods (cold)
rods_rotate = 0

# > buffer size for the activity image for registration purposes
buff_size = 16

# mu-value for the perspex attenuation
mu_prspx = 0.1036
mu_water = 0.096
mu_bone = 0.148
mu_screw = 0.117

# PNG values attenuation
water = 209
prspx = 167
scrws = 129
boney = 113

# activity values for inserts and background and NAC edges
ains = 200
abck = 100
aedg = 150

# PNG values for activity
png_ains = 129
png_abck = 167
png_acap = 129
png_aedg = 167

# ACR ring sample distancing
# background circle radii:
rbckgs = np.arange(12, 220, 12) / 2
# sampling ticks
sbckgs = rbckgs - rbckgs[0] / 2

# insert sampling
rinsrt = np.array([4, 8, 12, 16, 20, 25, 30, 34, 38, 42, 46, 50]) / 2
sinsrt = rinsrt - rinsrt[0] / 2
sinsrt[0] = 0.

# sampling resolution rods
# centres of sampling rings
srngs = np.array([2.52, 4.53, 6.54, 8.56, 10.65, 12.68, 14.71, 16.74, 18.8, 20.8, 24]) / 2
rods_rngc = np.concatenate(([0], srngs[0:-1] + (srngs[1:] - srngs[0:-1]) / 2))

# resolution rod radii
rods_rad = np.array([12.6, 11.02, 9.46, 7.84, 6.27, 4.66]) / 2

# rods nominal diameters
rods_nom = np.array(['12.7', '11.1', '9.5', '7.9', '6.4', '4.8'])

# number of sampling rings for each rod
rods_nrngs = np.array([11, 10, 9, 7, 6, 4])

# sampling offset for each rod pie-piece
rods_off = np.array([60, 50, 40, 30, 20, 10])

# DIPY rigid body registration parameters
dipy_itrs = [10000, 1000, 200]
dipy_sgms = [3.0, 1.0, 0.0]
dipy_fcts = [4, 2, 1]

# for rods more iterations
dipy_rods_itrs = [10000, 1000, 300]


def get_params(cpath=None):
    """
    Get all the parameters and file names for ACR template generation.
    Arguments:
      cpath  : path of custom ACR design files
    """
    if cpath is None:
        f_core_main_comp = resources.files(
            "niftypet.nimpa").resolve() / "acr_design" / "core_mumap" / "acr-main-compartment.png"
        if not f_core_main_comp.is_file():
            raise IOError('Design ACR phantom files are missing in the NIMPA package')
        mfldr = f_core_main_comp.parent.parent

    elif os.path.isdir(cpath) and os.path.isfile(
            os.path.join(cpath, 'core_mumap', 'acr-main-compartment.png')):
        mfldr = Path(cpath)
    else:
        raise ValueError('The input folder is not recognised')

    # design PDF/PNG path for sampling, mu-map, NAC distribution
    dmpth = mfldr / 'core_mumap'
    dnpth = mfldr / 'core_nac'
    drpth = mfldr / 'rods'
    dspth = mfldr / 'sampling'

    # dictionary of design constant
    Cntd = {
        'dspth': dspth, 'dmpth': dmpth, 'vxysz': vxysz, 'vxsz': vxsz, 'sclt': sclt,
        'interp': interp, 'scld': scld, 'scl': scl, 'intord': intord, 'rods_rotate': rods_rotate,
        'buff_rods_size': buff_size, 'itr_nac': itr_nac, 'itr_qnt': itr_qnt, 'itr_qnt2': itr_qnt2,
        'fwhm_nac': fwhm_nac, 'fwhm_qnt': fwhm_qnt}
    # mu-values
    Cntd.update(mu_prspx=mu_prspx, mu_water=mu_water, mu_bone=mu_bone, mu_screw=mu_screw)

    # activity values
    Cntd.update(ains=ains, abck=aedg)
    # PNG mu-values
    Cntd.update(png_water=water, png_prspx=prspx, png_scrwy=boney)
    # PNG activity values
    Cntd.update(png_ains=png_ains, png_abck=png_abck, png_acap=png_acap, png_aedg=png_aedg,
                dpad=pad, fwhm_tmpl=fwhm_tmpl, dipy_itrs=dipy_itrs, dipy_sgms=dipy_sgms,
                dipy_fcts_itrs=dipy_rods_itrs)
    # ACR core mu-map designs
    Cntd.update(fmcap0=dmpth / 'acr-main-cap-0.png', fmcap1=dmpth / 'acr-main-cap-1.png',
                fmcap2=dmpth / 'acr-main-cap-2.png', fbscrw=dmpth / 'acr-bone-screw.png',
                fscrws=dmpth / 'acr-screws.png', flid0=dmpth / 'acr-lid0.png',
                flid1=dmpth / 'acr-lid1.png', finsrt=dmpth / 'acr-inserts.png',
                fbinsrt=dmpth / 'acr-inserts-bottoms.png', fmain=dmpth / 'acr-bottom.png')
    # ACR core activity (NAC) designs
    Cntd.update(fcap=dnpth / 'acr-cap.png', fins=dnpth / 'acr-inserts.png',
                fbig=dnpth / 'acr-rng.png')
    # ACR rods
    Cntd.update(frespng=drpth / 'acr-rods.png', frenpng=drpth / 'acr-rods-ends.png',
                fresWpng=drpth / 'acr-rods-ends-water.png')
    # ACR sampling designs
    Cntd.update(fs_rods=dspth / 'acr-rods-sampling.png', fs_bckg=dspth / 'acr-bckg-sampling.png',
                fs_ibckg=dspth / 'acr-insrt-bckg-sampling.png',
                fs_air=dspth / 'acr-air-sampling.png', fs_h2o=dspth / 'acr-h2o-sampling.png',
                fs_bone=dspth / 'acr-bone-sampling.png', fs_hot1=dspth / 'acr-hot1-sampling.png',
                fs_hot2=dspth / 'acr-hot2-sampling.png', fs_hot3=dspth / 'acr-hot4-sampling.png')
    # parameters for plotting rods results
    Cntd.update(rods_rngc=rods_rngc, rods_rad=rods_rad, rods_nom=rods_nom, rods_nrn=rods_off)
    # intensity offsetsfor the front phantom sampling
    Cntd.update(soff_rods=None, soff_bckg=300, soff_ibckg=200, soff_air=70, soff_h2o=50,
                soff_bone=90, soff_hot1=10, soff_hot2=20, soff_hot3=30, soff_hot4=40,
                rbckgs=rbckgs, sbckgs=sbckgs, rinsrt=sinsrt)
    # AXIAL DEMNSIONS, extension in mm
    Cntd.update(k_rods=83.40, k_rodsend=3.0) # rods

    # CORE PHANTOM
    Cntd.update(
        k_mcapA=19.2,   # main cap top
        k_mcapB=5.3,    # main cap
        k_mcapC=6.7,    # main cap
        k_mcapD=12.3,   # main cap bottom
        k_lidA=5.8,     # lid top
        k_lidB=5.8,     # lid bottom
        k_insrtA=38.0,  # inserts
        k_insrtB=1.5,   # insert bottoms
        k_contA=147.5,  # main container
        k_unfrm=58.101, # main container uniform part
        k_contB=11.6,   # container bottom
    )

    return Cntd
