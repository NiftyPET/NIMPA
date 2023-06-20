"""ACR/Jaszczak PET phantom design, image reconstruction and analysis"""
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2021-3"

from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import curve_fit

from ..prc import imio, prc


def erf(x, a, u, k, b):
    """error function"""
    return a * scipy.special.erf(k * (x-u)) + b


def derf(x, a, u, k):
    """derivative of error function"""
    return a * k * 2 / (np.pi**.5) * np.exp(-(k * (x-u))**2)


def standard_analysis(
        fim,
        vois,
        fwhm=4.,
        patient_weight=70,  # kg
        simulated_dose=220, # MBq
        width_mm=10,
        zoffset=0):
    """
    Perform the standard ACR analysis.
    Arguments:
      fim:        the input NIfTI image in high resolution.
      vois:       the masks for VOIs to perform the analysis.
      fwhm:       the FWHM of the smoothing Gaussian kernel.
      zoffset:    offset from the middle of the axial extension
                  of the inserts.  By default the slice for
                  the analysis is in the middle of the insert's
                  axial extension.
      patient_does: patient simulated does in MBq
      patient_weight: patient weight in kg
    """

    fim = Path(fim)
    if not fim.is_file():
        raise IOError('unrecognised input NIfTI file')

    imd = imio.getnii(fim, output='all')

    # > for SUV calculations
    dose2wght = simulated_dose * 1e6 / (patient_weight*1e3)

    im_smo = prc.imsmooth(imd['im'], fwhm=fwhm, voxsize=imd['voxsize'])

    im_suv = im_smo / dose2wght

    # > axial width for standard analysis
    width_vox = int(np.round(width_mm / imd['voxsize'][0]))

    # > axial voxel range
    zrng = vois['r_insrt']

    z_start_idx = int(np.mean(zrng) - width_vox/2)

    msk = np.zeros(vois['s_i1'].shape, dtype=bool)

    zi = z_start_idx + zoffset
    msk[zi:zi + width_vox, ...] = True

    # > (A) Contrast
    h1 = np.float32(np.max(im_suv[vois['s_i1'] * msk]))
    h2 = np.max(im_suv[vois['s_i2'] * msk])
    h3 = np.max(im_suv[vois['s_i3'] * msk])
    h4 = np.max(im_suv[vois['s_i4'] * msk])

    # > (B) Scatter/Attenuation
    bckg_avg = np.float32(np.mean(im_suv[vois['s_bckg'] * msk]))
    bone_avg = np.mean(im_suv[vois['s_b'] * msk])
    h2o_avg = np.mean(im_suv[vois['s_w'] * msk])
    air_avg = np.mean(im_suv[vois['s_a'] * msk])

    bckg_min = np.min(im_suv[vois['s_bckg'] * msk])
    bone_min = np.min(im_suv[vois['s_b'] * msk])
    h2o_min = np.min(im_suv[vois['s_w'] * msk])
    air_min = np.min(im_suv[vois['s_a'] * msk])

    # > Ratio Calculations
    h1_bckg = h1 / bckg_avg
    h2_bckg = h2 / bckg_avg
    h3_bckg = h3 / bckg_avg
    h4_bckg = h4 / bckg_avg

    h2_h1 = np.float32(h2 / h1)
    h3_h1 = h3 / h1
    h4_h1 = h4 / h1

    air_bone = air_min / bone_min
    h2o_bone = h2o_min / bone_min

    out = {
        'h1max': h1, 'h2max': h2, 'h3max': h3, 'h4max': h4, 'bckg_avg': bckg_avg,
        'bone_avg': bone_avg, 'h2o_avg': h2o_avg, 'air_avg': air_avg, 'bckg_min': bckg_min,
        'bone_min': bone_min, 'h2o_min': h2o_min, 'air_min': air_min, 'h1_bckg': h1_bckg,
        'h2_bckg': h2_bckg, 'h3_bckg': h3_bckg, 'h4_bckg': h4_bckg, 'h2_h1': h2_h1, 'h3_h1': h3_h1,
        'h4_h1': h4_h1, 'air_bone': air_bone, 'h2o_bone': h2o_bone}

    for k in out:
        out[k] = np.float32(out[k])

    test1 = bckg_avg > 0.85 and bckg_avg < 1.15
    test2 = h1 > 1.8 and h1 < 2.8
    test3 = h2_h1 > 0.7

    out['test_bckg_avg'] = 'PASS'*test1 + 'FAIL' * ~test1 + ': ' + str(round(bckg_avg,
                                                                             2)) + ' (0.85-1.15)'
    out['test_25mm_insert'] = 'PASS'*test2 + 'FAIL' * ~test2 + ': ' + str(round(h1,
                                                                                2)) + ' (1.8-2.8)'
    out['test_16/25_insert'] = 'PASS'*test3 + 'FAIL' * ~test3 + ': ' + str(round(h2_h1,
                                                                                 2)) + ' (>0.7)'

    return out


def estimate_fwhm(fim, vois, Cntd, insert='water'):
    ''' Estimate the effective image resolution
        for any given ACR cylindrical insert
    '''

    from matplotlib import pyplot as plt

    if isinstance(fim, (str, Path)) and Path(fim).is_file():
        im = imio.getnii(fim)
    elif isinstance(fim, np.ndarray) and fim.ndim == 3:
        im = fim
    else:
        raise IOError('unrecognised input file or Numpy array')

    if im.shape != vois['fst_insrt'].shape:
        raise ValueError('the VOIs shape is incompatible with the image')

    ins = insert

    # > ring index ranges
    RIR = {
        'bone': [90, 102], # bone
        'air': [70, 82],   # air
        'water': [50, 62], # water
        'hot1': [10, 20],  # from biggest to smallest hot insert
        'hot2': [20, 30],
        'hot3': [30, 40],
        'hot4': [40, 50]}

    if ins not in RIR:
        raise ValueError('unrecognised insert name')

    # > pick what template to use (reduced axially or full)
    tmplt = vois['fst_insrt']
    tmplt3 = vois['fst_insrt3']

    # > sampling for analytical edge functions (transaxial)
    x = np.linspace(-1, np.max(Cntd['sinsrt']) + 1)

    # > extracted ring index values
    riv = np.zeros((RIR[ins][1] - RIR[ins][0]), dtype=np.float64)

    res = {
        'y': np.zeros(len(x), dtype=np.float64), 'dy': np.zeros(len(x), dtype=np.float64),
        'fwhm': 0, 'peak': 0, 'parg': 0, 'mnmx': 0}

    irngs = range(RIR[ins][0], RIR[ins][1])
    nrng = len(irngs)

    # > extract rings
    for i, l in enumerate(irngs):
        if ins == 'hot3':
            msk = tmplt3 == l
        else:
            msk = tmplt == l
        riv[i] = np.sum(im[msk]) / np.sum(msk)

    r = Cntd['sinsrt'][range(nrng)]
    re = x

    # ------- ERF fitting and derivative --------
    pe0 = [np.max(riv), np.median(r), .1, np.min(riv)]

    pe, pcov = curve_fit(erf, r, riv, pe0)

    ye = erf(re, *pe)
    res['y'] = ye

    dye = derf(re, *pe[:-1])
    res['dy'] = dye

    sgm = 1 / (np.sqrt(2) * pe[2])
    res['fwhm'] = 2 * np.sqrt(2 * np.log(2)) * sgm
    res['peak'] = 2 / (np.pi**.5) * pe[2] * pe[0]

    # > inflection point (should be around the insert border/wall)
    res['parg'] = pe[1]

    if ins[0] == 'h':
        res['mnmx'] = np.max(riv)
        res['dy'] *= -1
    else:
        res['mnmx'] = np.min(riv)
    res['r'] = r

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(res['r'], riv, 'o')
    ax[0].plot(x, res['y'])
    ax[1].plot(x, res['dy'])
    ax[0].set_title('Insert {}: FWHM = {} mm'.format(ins, round(res['fwhm'], 2)))
    ax[0].set_ylabel('Bq/ML')
    ax[1].set_ylabel('Bq/ML')
    ax[1].set_xlabel('distance from insert centre [mm]')

    res['r_values'] = riv
    return res
