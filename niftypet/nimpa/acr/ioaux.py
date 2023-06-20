"""ACR/Jaszczak PET phantom I/O and auxiliary functions"""
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2021-23"
import os
from pathlib import Path, PurePath

import dipy.align as align
import nibabel as nib
import numpy as np

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
except ImportError:
    pass

from ..prc import imio

# INPUT/OUTPUT


def get_paths(Cntd, outpath=None):
    '''
    Get all the paths for generating mu-maps and sampling templates.
    Arguments:
    Cntd    - Dictionary of constants and parameters
    outpath - output folder/path
    '''

    fimup = None
    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        fimup = Path(Cntd['fqntup'])

    elif 'fnacup' in Cntd and Path(Cntd['fnacup']).is_file():
        fimup = Path(Cntd['fnacup'])

    else:
        raise ValueError(
            'NAC or QNT PET image has to be identified in the constants dictionary Cntd')

    if not fimup:
        return None

    # > output path
    if isinstance(outpath, (str, PurePath)):
        opth = outpath
        Cntd['opth'] = opth
    elif 'opth' in Cntd:
        opth = Cntd['opth']
    else:
        raise ValueError('unrecognised output path')

    imio.create_dir(opth)

    # folder with templates, which will be registered to PET images
    tfldr = os.path.join(opth, 'templates')
    imio.create_dir(tfldr)
    t_acr_core_flddr = os.path.join(tfldr, 'ACR-core')
    imio.create_dir(t_acr_core_flddr)
    t_acr_reso_flddr = os.path.join(tfldr, 'ACR-rods')
    imio.create_dir(t_acr_reso_flddr)
    t_acr_smpl_flddr = os.path.join(tfldr, 'ACR-smpl')
    imio.create_dir(t_acr_smpl_flddr)

    # target rods output folder path
    rpth = os.path.join(opth, 'ACR-rods')
    imio.create_dir(rpth)
    # target core output folder path
    mpth = os.path.join(opth, 'ACR-core')
    imio.create_dir(mpth)

    # -----------------------------------------------------------
    # Core ACR and NAC PET (for registration)
    # -----------------------------------------------------------
    # rigid transformation affine output file name
    # for the NAC main/core part
    faff = os.path.join(mpth, 'affine-acr-dipy.npy')

    # output file name for the ACR NAC activity for registration
    facrad = os.path.join(
        t_acr_core_flddr,
        'acr-activity-' + str(Cntd['vxsz'] * Cntd['scld']).replace('.', '-') + 'mm.nii.gz')

    # output file for the ACR mu-map without resolution rod insert
    facrmu = os.path.join(t_acr_core_flddr,
                          'acr-umap-' + str(Cntd['vxsz']).replace('.', '-') + 'mm.nii.gz')

    # the same as above but with lower resolution (double voxel size)
    facrdmu = os.path.join(
        t_acr_core_flddr,
        'acr-umap-' + str(Cntd['vxsz'] * Cntd['scld']).replace('.', '-') + 'mm.nii.gz')

    # output mu-map for the original PET image reconstruction
    fmuo = os.path.join(mpth, 'acr-mumap-dipy.nii.gz')

    # output mu-map for the original PET image reconstruction
    # (high resolution, i.e., smaller voxel size)
    fmuo_hires = os.path.join(mpth, 'acr-mumap-hires-dipy.nii.gz')

    # full mu-map (combined core and resolution parts) - the endpoint result
    fmuf = os.path.join(opth, 'acr-complete-umap.nii.gz')

    # -----------------------------------------------------------
    # RESOLUTION RODS
    # -----------------------------------------------------------
    # output file name for the resolution rods mu-map
    fresomu = os.path.join(t_acr_reso_flddr,
                           'acr-reso-umap-' + str(Cntd['vxsz']).replace('.', '-') + 'mm.nii.gz')

    # the same but double voxel size (as defined by scld)
    fresdmu = os.path.join(
        t_acr_reso_flddr,
        'acr-reso-umap-' + str(Cntd['vxsz'] * Cntd['scld']).replace('.', '-') + 'mm.nii.gz')

    # the resolution rods mu-map with water background (used for the phantom)
    fresdWmu = os.path.join(
        t_acr_reso_flddr,
        'acr-reso-water-umap-' + str(Cntd['vxsz'] * Cntd['scld']).replace('.', '-') + 'mm.nii.gz')

    # the resolution active QNT for alternative registration of the rods to PET
    fresdQmu = os.path.join(
        t_acr_reso_flddr,
        'acr-reso-active-qnt-' + str(Cntd['vxsz'] * Cntd['scld']).replace('.', '-') + 'mm.nii.gz')

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        # output file for the resolution rods part only
        fpet_res = os.path.join(
            rpth,
            os.path.basename(fimup).split('-scale-' + str(Cntd['sclt']) + '-')[0] + '_rods.nii.gz')

        # output file for rigid body transformation affine
        faff_res = os.path.join(rpth, 'affine-dipy-acr-reso.npy')
        # if os.path.isfile(faff_res):
        #     txaff = np.load(faff_res)

        fmur = os.path.join(rpth, 'acr-reso-water-mumap-dipy.nii.gz')
    else:
        fpet_res = ''
        faff_res = ''
        fmur = ''

    # -----------------------------------------------------------
    # SAMPLING
    # -----------------------------------------------------------
    # sampling ring template for the rods
    fst_res = os.path.join(t_acr_smpl_flddr,
                           'acr-res-sampling-' + str(Cntd['vxsz']).replace('.', '-') + 'mm.nii.gz')
    # sampling template for the inserts
    fst_insrt = os.path.join(
        t_acr_smpl_flddr, 'acr-all-sampling-' + str(Cntd['vxsz']).replace('.', '-') + 'mm.nii.gz')
    # sampling template for the 3rd hot insert
    fst_insrt3 = os.path.join(
        t_acr_smpl_flddr,
        'acr-insrt3-sampling-' + str(Cntd['vxsz']).replace('.', '-') + 'mm.nii.gz')
    # sampling template for the insert background
    fst_ibckg = os.path.join(
        t_acr_smpl_flddr,
        'acr-ibckg-sampling-' + str(Cntd['vxsz']).replace('.', '-') + 'mm.nii.gz')

    # get the output file names into the dictionary
    Cntd['out'] = {
        'faff_res': faff_res, 'faff': faff, 'fpet_res': fpet_res, 'fresomu': fresomu,
        'fresdmu': fresdmu, 'fresdWmu': fresdWmu, 'fresdQmu': fresdQmu, 'fmur': fmur,
        'facrad': facrad, 'facrmu': facrmu, 'facrdmu': facrdmu, 'fmuo': fmuo,
        'fmuo_hires': fmuo_hires, 'fmuf': fmuf, 'fst_res': fst_res, 'fst_insrt': fst_insrt,
        'fst_insrt3': fst_insrt3, 'fst_ibckg': fst_ibckg}

    return Cntd


def extract_reso_part(Cntd, offset=15, forced=False):
    '''
    extract resolution part for registration
    Arguments:
    offset:     the extend of the extra extension from rods to the uniform
                region.
    forced:     force the extraction even if detected to be already done.
    '''

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = imio.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed QNT ACR PET image cannot be found')

    if forced or not os.path.isfile(Cntd['out']['fpet_res']):
        # pick the axial cut-off position based on the axial summed profile
        axprf = np.sum(imupd['im'], axis=(1, 2))

        thrshld = np.max(axprf) * 0.33
        ids = np.where(axprf > thrshld)
        i0 = np.min(ids)
        i1 = axprf.argmax()

        tmp = axprf[i0:i1]
        wndw = 7
        dip = np.array([(tmp[i + wndw] - tmp[i]) < -4e6 for i in range(i1 - i0 - wndw)])

        cutoff = np.max(np.where(dip)) + i0 + offset
        '''
        figure(); plot(axprf, '.-')
        figure(); plot(tmp, '.-')
        '''

        rodsim = imupd['im'].copy()
        rodsim[cutoff:, ...] = 0

        imio.array2nii(
            rodsim, imupd['affine'], Cntd['out']['fpet_res'],
            trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
                   imupd['transpose'].index(2)), flip=imupd['flip'])

        return tmp

    else:
        print('i> extraction has already been performed')
        return None


def sampling_masks(Cntd, use_stored=False):
    ''' get the sampling masks for analysis of the ACR phantom
    '''

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        fimup = str(Cntd['fqntup'])
    else:
        raise ValueError('Upscaled and trimmed ACR PET image cannot be found')

    # prepare output folder
    smpl_dir = os.path.join(os.path.dirname(Cntd['out']['fmuf']), 'sampling_masks')
    imio.create_dir(smpl_dir)
    print(f'i> using sampling folder: {smpl_dir}')

    # > dictionary of VOIs for sampling PET images,
    # > made by registering the templates to PET space.
    vois = {}

    for t in (t for t in Cntd['out'] if t[:4] == 'fst_'):
        fpth = Cntd['out'][t]

        print(fpth)

        if not os.path.isfile(fpth):
            raise ValueError('The sampling template file does not exists.')

        # template in PET space
        fvois = os.path.join(smpl_dir, os.path.basename(fpth).split('.nii.gz')[0] + '_dipy.nii.gz')

        # if the resampled template does not exist, resample it
        if not use_stored or not os.path.isfile(fvois):
            if t == 'fst_res':
                faff = Cntd['out']['faff_res']
            else:
                faff = Cntd['out']['faff']

            print('i> using this affine for resampling:\n   ', faff)

            affine = np.load(faff)

            static, static_affine, moving, moving_affine, between_affine = (
                align._public._handle_pipeline_inputs(fpth, fimup, moving_affine=None,
                                                      static_affine=None, starting_affine=affine))

            affine_map = align._public.AffineMap(between_affine, static.shape, static_affine,
                                                 moving.shape, moving_affine)

            rsmpl = affine_map.transform(moving, interpolation='nearest')

            smpl_nii = nib.Nifti1Image(np.int32(rsmpl), static_affine)
            nib.save(smpl_nii, fvois)

            del moving, static

        vois[t] = imio.getnii(fvois)

        # > masks/vois for standard analysis of the ACR phantom
        # > also get the axial range of each concentric VOIs

        # > 6 cm diameter ROI for ACR background
        if t == 'fst_ibckg':
            # > get the range for the uniform part
            zax = np.sum(vois[t] == 317, axis=(1, 2))
            zax = np.where(zax > 0)[0]
            vois['r_bckg'] = [zax[0], zax[-1]]

            # > get the range for the faceplate inserts
            zax = np.sum(vois[t] == 217, axis=(1, 2))
            zax = np.where(zax > 0)[0]
            vois['r_ibckg'] = [zax[0], zax[-1]]

            # > get the standard ACR background VOI
            vois['s_bckg'] = (vois[t] >= 200) & (vois[t] < 205)

        elif t == 'fst_insrt':
            # > get the range:
            zax = np.sum(vois[t] == 100, axis=(1, 2))
            zax = np.where(zax > 0)[0]
            vois['r_insrt'] = [zax[0], zax[-1]]

            # > get the VOI for the inserts
            vois['s_i1'] = (vois[t] <= 14) & (vois[t] >= 10)
            vois['s_i2'] = (vois[t] <= 24) & (vois[t] >= 20)
            vois['s_i4'] = (vois[t] <= 44) & (vois[t] >= 40)
            vois['s_w'] = (vois[t] <= 54) & (vois[t] >= 50)
            vois['s_a'] = (vois[t] <= 74) & (vois[t] >= 70)
            vois['s_b'] = (vois[t] <= 94) & (vois[t] >= 90)

        elif t == 'fst_insrt3':
            # > get the range:
            zax = np.sum(vois[t] == 38, axis=(1, 2))
            zax = np.where(zax > 0)[0]
            vois['r_insrt3'] = [zax[0], zax[-1]]

            # > get the VOI for the insert
            vois['s_i3'] = (vois[t] <= 34) & (vois[t] >= 30)

        elif t == 'fst_res':
            # > get the range:
            zax = np.sum(vois[t] == 70, axis=(1, 2))
            zax = np.where(zax > 0)[0]
    return vois


def zmask(masks, key, Cntd, axial_offset=8, width_mm=10, z_start_idx=None, level=None):
    '''
    Obtain a sub-mask for the standard ACR analysis using 1 cm slice.
    Arguments:
        width_mm: axial width of the ROI mask (in mm)
    '''

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = imio.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed ACR PET image cannot be found')

    if level is None:
        zax = np.sum(masks[key], axis=(1, 2))
    else:
        zax = np.sum(masks[key] == level, axis=(1, 2))
    zax = np.where(zax > 0)[0]

    # width of axial voxel extension
    width_vox = int(np.round(width_mm / imupd['voxsize'][0]))

    # the range
    z0 = zax[0] + axial_offset
    z1 = zax[-1] - axial_offset # wip: maybe -width_vox

    msk = np.zeros(masks[key].shape, dtype=bool)

    if z_start_idx is None:
        z_start_idx = z0

    msk[z_start_idx:z_start_idx + width_vox, ...] = True

    zmasks = {
        'z0': z0, 'z1': z1, 'zax': zax, 'width_vox': width_vox, 'z_start_idx': z_start_idx,
        'msk': msk}
    zmasks[key] = masks[key] * msk

    if key == 'fst_insrt':
        # 6 cm diameter ROI for ACR background
        msk_bck = (masks['fst_ibckg'] >= 200) & (masks['fst_ibckg'] < 205) * zmasks['msk']
        # all the inserts
        msk_i1 = (masks['fst_insrt'] <= 14) & (masks['fst_insrt'] >= 10) * zmasks['msk']
        msk_i2 = (masks['fst_insrt'] <= 24) & (masks['fst_insrt'] >= 20) * zmasks['msk']
        msk_i3 = (masks['fst_insrt3'] <= 34) & (masks['fst_insrt3'] >= 30) * zmasks['msk']
        msk_i4 = (masks['fst_insrt'] <= 44) & (masks['fst_insrt'] >= 40) * zmasks['msk']
        msk_w = (masks['fst_insrt'] <= 54) & (masks['fst_insrt'] >= 50) * zmasks['msk']
        msk_a = (masks['fst_insrt'] <= 74) & (masks['fst_insrt'] >= 70) * zmasks['msk']
        msk_b = (masks['fst_insrt'] <= 94) & (masks['fst_insrt'] >= 90) * zmasks['msk']

        zmasks['msk_bckg'] = msk_bck
        zmasks['msk_i1'] = msk_i1
        zmasks['msk_i2'] = msk_i2
        zmasks['msk_i3'] = msk_i3
        zmasks['msk_i4'] = msk_i4
        zmasks['msk_w'] = msk_w
        zmasks['msk_a'] = msk_a
        zmasks['msk_b'] = msk_b

        zmasks['fst_ibckg'] = masks['fst_ibckg'] * msk

        zmasks[key + '3'] = masks[key + '3'] * msk

    return zmasks


def extract_rings(img, tmpl, l0=None, l1=None):
    """
    extract average ROI ring values from PET image using the high resolution templates.
    """

    tmpl = np.int32(tmpl)

    # unique ring labels (id values)
    urs = np.unique(tmpl)

    if l0 is None:
        l0 = np.min(urs)
    if l1 is None:
        l1 = np.max(urs) + 1

    # rings ids (labels)
    irngs = range(l0, l1)

    # number of rings (labels)
    nrng = len(irngs)

    # initialise array for storing average voxel values for any label
    vrngs = np.zeros(nrng, dtype=np.float32)

    for i, l in enumerate(irngs):
        msk = tmpl == l
        vrngs[i] = np.sum(img[msk]) / np.sum(msk)

    return vrngs


def plot_hotins(im, masks, Cntd, inserts=None, axes=None, ylim=None, colour='k', marker_sz=3.5,
                draw_bckg=True, legend=True):
    """
    plot sampling rings in hot insert regions of size 25, 16, 12 and 8mm.
    Arguments:
      im:     input image for sampling (high resolution of around 0.5 mm)
      masks:  the concentric VOIs for sampling inserts
      inserts: the list of selected inserts for plotting
              (1: the biggest, 4: the smallest)
      draw_back: if True, draws insert walls in the background
    """
    if inserts is None:
        inserts = [1, 2, 3, 4]

    rinsrt = [12.5, 8, 6, 4]

    if axes is None:
        _, ax = plt.subplots(1)
    else:
        ax = axes

    hots = np.array([])
    # extract ring values
    if 1 in inserts:
        rh1 = extract_rings(im, masks['fst_insrt'], l0=10, l1=20)
        hots = np.concatenate((rh1, hots), axis=0)
        plt.plot(Cntd['sinsrt'][:-2], rh1, 'o-', color=colour, ms=marker_sz, label='25-mm')

    if 2 in inserts:
        rh2 = extract_rings(im, masks['fst_insrt'], l0=20, l1=30)
        hots = np.concatenate((rh2, hots), axis=0)
        plt.plot(Cntd['sinsrt'][:-2], rh2, 'p-', color=colour, ms=marker_sz, label='16-mm')

    if 3 in inserts:
        rh3 = extract_rings(im, masks['fst_insrt3'], l0=30, l1=40)
        hots = np.concatenate((rh3, hots), axis=0)
        plt.plot(Cntd['sinsrt'][:-2], rh3, 'x-', color=colour, ms=marker_sz, label='12-mm')

    if 4 in inserts:
        rh4 = extract_rings(im, masks['fst_insrt'], l0=40, l1=50)
        hots = np.concatenate((rh4, hots), axis=0)
        plt.plot(Cntd['sinsrt'][:-2], rh4, 'v-', color=colour, ms=marker_sz, label='8-mm')

    # get the global y-limits for all axes
    if ylim is None:
        ylim = [np.floor(np.amin(hots) / 100) * 100, np.ceil(np.amax(hots) / 100) * 100]

    if draw_bckg:
        for i in inserts:
            rect = patches.Rectangle((rinsrt[i - 1], ylim[0]), 1.5, ylim[1], linewidth=0,
                                     edgecolor='None', facecolor='0.85')
            ax.add_patch(rect)

    ax.set_ylim(ylim)
    plt.ylabel('Bq/mL')
    plt.xlabel('sampling ring radii [mm]')
    plt.title('hot inserts')
    if legend:
        plt.legend()
    return ax


def plot_coldins(im, Cntd, masks, ylim=None, line_style='.-', colour='k', axes=None,
                 draw_bckg=True):
    """plot sampling rings in cold insert regions of water, air and bone"""
    if axes is None:
        _, axs = plt.subplots(1, 3)
    else:
        axs = axes

    colds = np.array([])

    r_h2o = extract_rings(im, masks['fst_insrt'], l0=50, l1=62)
    r_air = extract_rings(im, masks['fst_insrt'], l0=70, l1=82)
    r_bone = extract_rings(im, masks['fst_insrt'], l0=90, l1=102)
    colds = np.concatenate((colds, r_h2o, r_air, r_bone), axis=0)

    # get the global y-limits for all axes
    if ylim is None:
        ylim = [np.floor(np.amin(colds) / 100) * 100, np.ceil(np.amax(colds) / 100) * 100]

    # water
    axs[0].set_ylabel('Bq/mL')
    axs[0].plot(Cntd['sinsrt'], r_h2o, line_style, color=colour)
    axs[0].set_ylim(ylim)
    axs[0].set_xlim([0, 25])
    axs[0].set_title('water')

    if draw_bckg:
        rect = patches.Rectangle((0, 0), 12.5, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor='b', alpha=0.1)
        # add the patch to the plot
        axs[0].add_patch(rect)
        rect = patches.Rectangle((12.5, 0), 1.5, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor=str(0.8))
        # add the patch to the plot
        axs[0].add_patch(rect)
        rect = patches.Rectangle((14, 0), 11, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor='red', alpha=0.1)
        # add the patch to the plot
        axs[0].add_patch(rect)

    # air
    axs[1].plot(Cntd['sinsrt'], r_air, line_style, color=colour)
    axs[1].set_ylim(ylim)
    axs[1].set_xlim([0, 25])
    axs[1].get_yaxis().set_ticks([])
    axs[1].set_title('air')
    axs[1].set_xlabel('ring radii [mm]')

    if draw_bckg:
        rect = patches.Rectangle((0, 0), 12.5, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor='None')
        # add the patch to the plot
        axs[1].add_patch(rect)
        rect = patches.Rectangle((12.5, 0), 1.5, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor=str(0.8))
        # add the patch to the plot
        axs[1].add_patch(rect)
        rect = patches.Rectangle((14, 0), 11, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor='red', alpha=0.1)
        # add the patch to the plot
        axs[1].add_patch(rect)

    # bone
    axs[2].plot(Cntd['sinsrt'], r_bone, line_style, color=colour)
    axs[2].set_ylim(ylim)
    axs[2].set_xlim([0, 25])
    axs[2].get_yaxis().set_ticks([])
    axs[2].set_title('bone')

    if draw_bckg:
        rect = patches.Rectangle((0, 0), 12.5, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor='0.85')
        axs[2].add_patch(rect)
        rect = patches.Rectangle((12.5, 0), 12.5, ylim[1], linewidth=0, edgecolor='None',
                                 facecolor='red', alpha=0.1)
        axs[2].add_patch(rect)

    return axs


def plot_uniformity(im, Cntd, masks, inserts_area=True, uniform_area=True, ylim=None,
                    ring_draw_ymin=None, ring_drw_ymax=None, grid=False, axis=None):
    """plot the uniformity regions using 18 sampling rings"""
    if axis is None:
        _, ax = plt.subplots(1)
    else:
        ax = axis

    # background around inserts
    if inserts_area:
        r_bi = extract_rings(im, masks['fst_ibckg'], l0=200, l1=218)
        plt.plot(Cntd['sbckgs'], r_bi, '.-', color='k', label='insert area')

    # background in the middle uniform part
    if uniform_area:
        r_b = extract_rings(im, masks['fst_insrt'], l0=300, l1=318)
        plt.plot(Cntd['sbckgs'], r_b, '.--', color='k', label='uniform area')

    plt.legend()

    if ylim is not None:
        plt.ylim(ylim)

    plt.ylabel('Bq/mL')
    plt.xlabel('sampling ring radii [mm]')
    plt.title('background rings')

    if ring_drw_ymax is not None and ring_draw_ymin is not None:
        for k in range(len(Cntd['rbckgs'])):
            rect = patches.Rectangle((Cntd['rbckgs'][k] - 6, ring_draw_ymin), 6, ring_drw_ymax,
                                     linewidth=0, edgecolor='None',
                                     facecolor=str(0.8 + (k%2) * 0.1))
            # add the patch to the plot
            ax.add_patch(rect)

    if grid: plt.grid('on')
    plt.show()


def plot_rods(Cntd, masks, im, color='k', ylim=None, pick_rods=None, draw_rods=True,
              contrast_ref=None, line_style='.-', out_raw=False):
    """
    Calculate the resolution curves for each set of rod diameters.
    Return the contrast for each rod set.
    """

    # pick the rods for which to plot the recovery
    if pick_rods is None:
        pick_rods = range(0, len(Cntd['rods_nrngs']))

    # contrast and ratio values for output
    cntrst = np.zeros(len(pick_rods))
    ratios = np.zeros(len(pick_rods))

    # raw values for each ring and selected rod
    rawval = np.zeros((len(pick_rods), np.max(Cntd['rods_nrngs'][pick_rods])), dtype=np.float32)

    # sampling
    for k, (nrng, off, rad) in enumerate(
            zip(Cntd['rods_nrngs'][pick_rods], Cntd['rods_off'][pick_rods],
                Cntd['rods_rad'][pick_rods])):

        print(f'# sampling rings {nrng} with offset {off} and radius {rad}')

        vrng = np.zeros(nrng, dtype=np.float32)
        for i in range(nrng):
            msk = masks['fst_res'] == i + off
            vrng[i] = np.sum(im[msk]) / np.sum(msk)

        rawval[k, :nrng] = vrng

        ratios[k] = vrng[-1] / vrng[0]
        if contrast_ref is None:
            cntrst[k] = (vrng[-1] - vrng[0]) / vrng[-1]
        else:
            cntrst[k] = (vrng[-1] - vrng[0]) / contrast_ref

        plt.plot(Cntd['rods_rngc'][:nrng], vrng, line_style, color=color)

    if draw_rods:
        for rx in Cntd['rods_rad'][pick_rods]:
            plt.axvline(x=rx, color='0.8', linestyle='-', lw=1)

    plt.ylabel('Bq/mL')
    plt.xlabel('sampling ring radii [mm]')
    plt.title('ACR resolution rods')
    plt.show()

    return rawval if out_raw else (cntrst, ratios)
