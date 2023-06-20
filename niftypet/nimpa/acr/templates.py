"""ACR/Jaszczak PET phantom templates"""
__author__ = ("Pawel Markiewicz", "Casper da Costa-Luis")
__copyright__ = "Copyright 2021-23"
import os
from pathlib import Path

import imageio
import numpy as np
import scipy.ndimage as ndi

from niftypet import nimpa

# MU-MAPs


def create_mumap_core(Cntd, return_raw=False):
    """ACR core mu-map generation without the resolution rods"""
    if all(os.path.isfile(Cntd['out'][f]) for f in ('facrmu', 'facrdmu')):
        print('i> found already generated mu-map at:\n{}\nand\n{}'.format(
            Cntd['out']['facrmu'], Cntd['out']['facrdmu']))
        return None

    if 'fnacup' in Cntd and Path(Cntd['fnacup']).is_file():
        imupd = nimpa.getnii(Cntd['fnacup'], output='all')
    elif 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = nimpa.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed ACR PET image cannot be found')

    # > main cap
    mcap0_png = imageio.imread(Cntd['fmcap0'])
    mcap1_png = imageio.imread(Cntd['fmcap1'])
    mcap2_png = imageio.imread(Cntd['fmcap2'])

    # > bone and other screws
    bscrw_png = imageio.imread(Cntd['fbscrw'])
    scrws_png = imageio.imread(Cntd['fscrws'])

    # > lid
    lid0_png = imageio.imread(Cntd['flid0'])
    lid1_png = imageio.imread(Cntd['flid1'])

    # > cylindrical inserts attached to the lid
    insrt_png = imageio.imread(Cntd['finsrt'])
    binsrt_png = imageio.imread(Cntd['fbinsrt'])

    # > the main phantom container with the bottom
    main_png = imageio.imread(Cntd['fmain'])
    bttm_png = imageio.imread(Cntd['fbttm'])

    # > convert image slices to mu-map values
    # > main cap, part 0 (A)
    mcap0 = np.float32((mcap0_png[..., 0] == Cntd['png_scrws']) * Cntd['mu_screw'])
    mcap0 = ndi.zoom(mcap0, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(mcap0, cmap='bone', vmin=0.07, vmax=0.14)
    # > main cap, part 1 (B)
    mcap1 = np.float32((mcap1_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
        (mcap1_png[..., 0] == Cntd['png_scrws']) * Cntd['mu_screw'])
    mcap1 = ndi.zoom(mcap1, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(mcap1, cmap='bone', vmin=0.07, vmax=0.14)
    # > main cap, part 2 (C)
    mcap2 = np.float32((mcap2_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
        (mcap2_png[..., 0] == Cntd['png_water']) * Cntd['mu_water'])
    # prnt = mcap2
    mcap2 = ndi.zoom(mcap2, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(mcap2, cmap='bone', vmin=0.07, vmax=0.14)
    # > normal screws
    ascrws = np.float32((scrws_png[..., 0] == Cntd['png_scrws']) * Cntd['mu_screw'])
    # prnt += ascrws
    ascrws = ndi.zoom(ascrws, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(ascrws, cmap='bone', vmin=0.07, vmax=0.14)
    # > bone screw
    bscrws = np.float32((bscrw_png[..., 0] == Cntd['png_scrws']) * Cntd['mu_screw'])
    # prnt += bscrws
    bscrws = ndi.zoom(bscrws, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(ascrws+bscrws, cmap='bone', vmin=0.07, vmax=0.14)
    # matshow(prnt, cmap='bone', vmin=0.07, vmax=0.14)

    # > multiple slices according to the thickness of different part (axially)

    # > main cap with the three parts
    k = int(np.round(Cntd['k_mcapA'] / Cntd['vxsz']))
    mcapA = np.repeat(mcap0[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_mcapB'] / Cntd['vxsz']))
    mcapB = np.repeat(mcap1[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_mcapC'] / Cntd['vxsz']))
    tmp = (ascrws + mcap1)
    mcapC = np.repeat(tmp[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_mcapD'] / Cntd['vxsz']))
    tmp = (ascrws + mcap2 + bscrws)
    mcapD = np.repeat(tmp[None, ...], k, axis=0)

    # matshow(mcapD[0,...], cmap='bone')

    # > lid in two parts
    lid0 = np.float32((lid0_png[..., 0] == Cntd['png_scrws']) * Cntd['mu_screw']) + np.float32(
        (lid0_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
            (lid0_png[..., 0] == Cntd['png_water']) * Cntd['mu_water'])
    lid0 = ndi.zoom(lid0, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(lid0, cmap='bone', vmin=0.07, vmax=0.14)
    lid1 = np.float32((lid1_png[..., 0] == Cntd['png_scrws']) * Cntd['mu_screw']) + np.float32(
        (lid1_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
            (lid1_png[..., 0] == Cntd['png_water']) * Cntd['mu_water'])
    lid1 = ndi.zoom(lid1, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(lid1, cmap='bone', vmin=0.07, vmax=0.14)

    k = int(np.round(Cntd['k_lidA'] / Cntd['vxsz']))
    lidA = np.repeat(lid0[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_lidB'] / Cntd['vxsz']))
    lidB = np.repeat(lid1[None, ...], k, axis=0)

    # > cylindrical inserts
    insrt = np.float32((insrt_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
        (insrt_png[..., 0] == Cntd['png_boney']) * Cntd['mu_bone']) + np.float32(
            (insrt_png[..., 0] == Cntd['png_water']) * Cntd['mu_water'])
    insrt = ndi.zoom(insrt, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(insrt, cmap='bone', vmin=0.07, vmax=0.14)
    # > bone insert
    insrtb = np.float32((binsrt_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
        (binsrt_png[..., 0] == Cntd['png_water']) * Cntd['mu_water'])
    insrtb = ndi.zoom(insrtb, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(insrtb, cmap='bone', vmin=0.07, vmax=0.14)

    k = int(np.round(Cntd['k_insrtA'] / Cntd['vxsz']))
    insrtA = np.repeat(insrt[None, ...], k, axis=0)
    '''
    matshow(insrtA[0,...], cmap='bone', vmin=0.08, vmax=0.12)
    '''

    k = int(np.round(Cntd['k_insrtB'] / Cntd['vxsz']))
    insrtB = np.repeat(insrtb[None, ...], k, axis=0)

    # > main compartment
    main = np.float32((main_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']) + np.float32(
        (main_png[..., 0] == Cntd['png_water']) * Cntd['mu_water'])
    main = ndi.zoom(main, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(main, cmap='bone', vmin=0.07, vmax=0.14)
    # > bottom of the phantom
    bttm = np.float32((bttm_png[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx'])
    bttm = ndi.zoom(bttm, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    # matshow(bttm, cmap='bone', vmin=0.07, vmax=0.14)

    k = int(np.round(Cntd['k_contA'] / Cntd['vxsz']))
    contA = np.repeat(main[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_contB'] / Cntd['vxsz']))
    contB = np.repeat(bttm[None, ...], k, axis=0)

    # > ASSEMBLE ALL PARTS
    # > whole container - concatenate all the bits together
    acr = np.concatenate((mcapA, mcapB, mcapC, mcapD, lidA, lidB, insrtA, insrtB, contA, contB),
                         axis=0)

    # > pad the image to make the dims even
    acr = np.pad(acr, ((0, 0), (Cntd['dpad'], Cntd['dpad'] + 1), (Cntd['dpad'], Cntd['dpad'] + 1)),
                 constant_values=((0, 0), (0, 0), (0, 0)))

    # > make it up-side down (as it is scanned)
    acr = acr[::-1, :, :]

    # > get the affine and the save to NIfTI
    imxys = acr.shape[2]
    imzys = acr.shape[0]
    affine = np.array([[-Cntd['vxsz'], 0., 0., .5 * imxys * Cntd['vxsz']],
                       [0., Cntd['vxsz'], 0., -.5 * imxys * Cntd['vxsz']],
                       [0., 0., Cntd['vxsz'], -.5 * imzys * Cntd['vxsz']], [0., 0., 0., 1.]])

    nimpa.array2nii(
        acr, affine, Cntd['out']['facrmu'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])
    '''
    nimpa.imscroll(facrmu, view='s', vmin=0.08, vmax=0.12, cmap='bone')
    nimpa.imscroll(acr, view='s', vmin=0.07, vmax=0.14, cmap='bone')
    matshow(acr[...,600])
    matshow(imup['im'][...,300])
    '''

    # > scale down the image; used for reducing the registration time

    acrd = ndi.zoom(acr, 1 / Cntd['scld'], output=None, order=1, mode='constant')

    # > affine
    imxys = acrd.shape[2]
    imzys = acrd.shape[0]
    affined = np.array(
        [[-Cntd['vxsz'] * Cntd['scld'], 0., 0., .5 * imxys * Cntd['vxsz'] * Cntd['scld']],
         [0., Cntd['vxsz'] * Cntd['scld'], 0., -.5 * imxys * Cntd['vxsz'] * Cntd['scld']],
         [0., 0., Cntd['vxsz'] * Cntd['scld'], -.5 * imzys * Cntd['vxsz'] * Cntd['scld']],
         [0., 0., 0., 1.]])

    nimpa.array2nii(
        acrd, affined, Cntd['out']['facrdmu'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    # nimpa.imscroll(acrd, view='s', vmin=0.08, vmax=0.12, cmap='bone')
    # del insrt, insrtb, insrtA, insrtB, contA, contB, main
    # del lidA, lidB, lid0, lid1, mcapA, mcapB, mcapC, mcapD

    # facr = os.path.join(os.path.dirname(imup['fim']), 'acr.npz')
    # np.savez(facr, acrd=acrd, affined=affined)

    # > PRINT:
    '''
    p = np.load(os.path.join(os.path.dirname(imup['fim']), 'print-resw.npy'))
    acr[p>0] = p[p>0]
    np.save(os.path.join(os.path.dirname(imup['fim']), 'print-acr.npy'), acr)
    a = np.load(os.path.join(os.path.dirname(imup['fim']), 'print-acr.npy'))
    nimpa.imscroll(a, view='s', vmin=0.07, vmax=0.14, cmap='bone')
    matshow(a[...,550], vmin=0.07, vmax=0.14, cmap='bone')
    matshow(a[...,430], vmin=0.07, vmax=0.14, cmap='bone')
    '''
    return {'acrd': acrd, 'acr': acr} if return_raw else None


def create_nac_core(Cntd, return_raw=False):
    """simulate NAC PET image for the main/core registration"""
    if os.path.isfile(Cntd['out']['facrad']):
        return None

    # > dictionary of upsampled and trimmed NAC pet image
    if 'fnacup' in Cntd and Path(Cntd['fnacup']).is_file():
        imupd = nimpa.getnii(Cntd['fnacup'], output='all')
    elif 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = nimpa.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed ACR PET image cannot be found')

    cap_png = imageio.imread(Cntd['fcap'])
    ins_png = imageio.imread(Cntd['fins'])
    big_png = imageio.imread(Cntd['fbig'])
    rng_png = imageio.imread(Cntd['frng'])

    cap = np.float32((cap_png[..., 0] == Cntd['png_acap']) * Cntd['abck'])
    cap = ndi.zoom(cap, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    ins = np.float32((ins_png[..., 0] == Cntd['png_abck']) * Cntd['abck']) + np.float32(
        (ins_png[..., 0] == Cntd['png_ains']) * Cntd['ains'])
    ins = ndi.zoom(ins, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    big = np.float32((big_png[..., 0] == Cntd['png_abck']) * Cntd['abck'])
    big = ndi.zoom(big, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    rng = np.float32((rng_png[..., 0] == Cntd['png_aedg']) * Cntd['aedg'])
    rng = ndi.zoom(rng, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    ins[rng == Cntd['aedg']] = Cntd['aedg']
    big[rng == Cntd['aedg']] = Cntd['aedg']

    # > for the gap, lower intensity
    rng[rng == Cntd['aedg']] = Cntd['aedg'] // 3
    '''
    matshow(cap, cmap='gray_r',vmax=200)
    matshow(ins, cmap='gray_r',vmax=200)
    matshow(big, cmap='gray_r',vmax=200)
    matshow(rng, cmap='gray_r',vmax=200)
    '''

    # > put unique slices together with k number representing the thickness
    # > variables with underscore act as zero-fillers
    k = int(np.round(Cntd['k_mcapA'] / Cntd['vxsz'])) + int(
        np.round(Cntd['k_mcapB'] / Cntd['vxsz'])) + int(np.round(Cntd['k_mcapC'] / Cntd['vxsz']))
    cap_ = np.zeros((k,) + cap.shape, dtype=np.float32)

    k = int(np.round(Cntd['k_mcapD'] / Cntd['vxsz']))
    capa = np.repeat(cap[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_lidA'] / Cntd['vxsz'])) + int(np.round(Cntd['k_lidB'] / Cntd['vxsz']))
    lid_ = np.zeros((k,) + cap.shape, dtype=np.float32)

    k = int(np.round(Cntd['k_insrtA'] / Cntd['vxsz']))
    insa = np.repeat(ins[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_unfrm'] / Cntd['vxsz'])) + int(
        np.round(Cntd['k_insrtB'] / Cntd['vxsz']))
    biga = np.repeat(big[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_rodsend'] / Cntd['vxsz']))
    gapa = np.repeat(rng[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_rods'] / Cntd['vxsz']))
    resa = np.repeat(big[None, ...], k, axis=0)

    k = int(np.round(Cntd['k_contB'] / Cntd['vxsz']))
    btm_ = np.zeros((k,) + cap.shape, dtype=np.float32)

    # > 3D assembly
    acra = np.concatenate((cap_, capa, lid_, insa, biga, gapa, resa, gapa, btm_), axis=0)

    # > pad the image to make the dims even
    acra = np.pad(acra,
                  ((0, 0), (Cntd['dpad'], Cntd['dpad'] + 1), (Cntd['dpad'], Cntd['dpad'] + 1)),
                  constant_values=((0, 0), (0, 0), (0, 0)))

    # > make it up-side down (as it is scanned)
    acra = acra[::-1, :, :]

    # > scale down
    acrad = ndi.zoom(acra, 1 / Cntd['scld'], output=None, order=1, mode='constant')

    # > affine
    imxys = acrad.shape[2]
    imzys = acrad.shape[0]
    affined = np.array(
        [[-Cntd['vxsz'] * Cntd['scld'], 0., 0., .5 * imxys * Cntd['vxsz'] * Cntd['scld']],
         [0., Cntd['vxsz'] * Cntd['scld'], 0., -.5 * imxys * Cntd['vxsz'] * Cntd['scld']],
         [0., 0., Cntd['vxsz'] * Cntd['scld'], -.5 * imzys * Cntd['vxsz'] * Cntd['scld']],
         [0., 0., 0., 1.]])

    nimpa.array2nii(
        acrad, affined, Cntd['out']['facrad'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    # matshow(acrad[420,...], cmap='magma')
    # matshow(imup['im'][420,...], cmap='magma')
    '''
    #PRINT
    matshow(acra[...,555], cmap='gray_r', vmax=200)
    '''

    if return_raw:
        return acrad
    else:
        return None


def create_reso(Cntd, return_raw=False):
    """Create the resolution rods for the mu-map and for registration."""
    if all(os.path.isfile(Cntd['out'][f]) for f in ('fresomu', 'fresdmu', 'fresdWmu', 'fresdQmu')):
        return None

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = nimpa.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed ACR QNT PET image cannot be found')

    # ------------------------------------------------------
    # > form 3D digital resolution part of the phantom for registration
    # > and attenuation purposes
    renpng = imageio.imread(Cntd['frenpng'])
    respng = imageio.imread(Cntd['frespng'])

    # > these are rods mixed with water
    renWpng = imageio.imread(Cntd['frenWpng'])
    resWpng = imageio.imread(Cntd['fresWpng'])
    # ------------------------------------------------------

    ren = np.float32((renpng[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx'])
    res = np.float32((respng[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx'])
    '''
    matshow(res, cmap='bone', vmin=0.07, vmax=0.14)
    matshow(ren, cmap='bone', vmin=0.07, vmax=0.14)
    '''

    renW = (renWpng[..., 0] == Cntd['png_water']) * Cntd['mu_water'] + (
        renWpng[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']
    resW = (resWpng[..., 0] == Cntd['png_water']) * Cntd['mu_water'] + (
        resWpng[..., 0] == Cntd['png_prspx']) * Cntd['mu_prspx']
    renW = np.float32(renW)
    resW = np.float32(resW)

    # --PRINT--
    '''
    matshow(ren, cmap='gray_r', vmax= 0.14)
    matshow(res, cmap='gray_r', vmax= 0.14)
    matshow(renW, cmap='bone', vmin=0.07, vmax=0.14)
    matshow(resW, cmap='bone', vmin=0.07, vmax=0.14)
    '''
    # --END-PRINT--

    if Cntd['rods_rotate'] != 0:
        ren = ndi.rotate(ren, Cntd['rods_rotate'], reshape=False, order=Cntd['intord'],
                         mode='constant')
        res = ndi.rotate(res, Cntd['rods_rotate'], reshape=False, order=Cntd['intord'],
                         mode='constant')

        renW = ndi.rotate(renW, Cntd['rods_rotate'], reshape=False, order=Cntd['intord'],
                          mode='constant')
        resW = ndi.rotate(resW, Cntd['rods_rotate'], reshape=False, order=Cntd['intord'],
                          mode='constant')

    ren = ndi.zoom(ren, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    res = ndi.zoom(res, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    renW = ndi.zoom(renW, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    resW = ndi.zoom(resW, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    # > rod ends
    k = int(np.round(Cntd['k_rodsend'] / Cntd['vxsz']))
    ren = np.repeat(ren[None, ...], k, axis=0)
    renW = np.repeat(renW[None, ...], k, axis=0)
    '''
    matshow(ren[0,...])
    matshow(renW[0,...])
    '''

    # > rods themselves
    k = int(np.round(Cntd['k_rods'] / Cntd['vxsz']))
    res = np.repeat(res[None, ...], k, axis=0)
    resW = np.repeat(resW[None, ...], k, axis=0)

    # > create a buffer for registration purposes
    # > as a margin between the rods end and the uniform part
    bsz = Cntd['buff_rods_size']
    buff = resW.copy()
    buff = buff[:bsz, ...]
    buff[:] = 0

    # > put all together
    reso = np.concatenate((ren, res, ren, buff), axis=0)
    # reso = np.pad( reso, ((1,0), (0,1),(0,1)), constant_values=((0,0), (0,0), (0, 0)) )
    reso = np.pad(reso,
                  ((1, 0), (Cntd['dpad'], Cntd['dpad'] + 1), (Cntd['dpad'], Cntd['dpad'] + 1)),
                  constant_values=((0, 0), (0, 0), (0, 0)))

    # nimpa.imscroll(res, view='c')

    imxys = reso.shape[2]
    imzys = reso.shape[0]
    affine = np.array([[-Cntd['vxsz'], 0., 0., .5 * imxys * Cntd['vxsz']],
                       [0., Cntd['vxsz'], 0., -.5 * imxys * Cntd['vxsz']],
                       [0., 0., Cntd['vxsz'], -.5 * imzys * Cntd['vxsz']], [0., 0., 0., 1.]])

    nimpa.array2nii(
        reso,                                                            # [::-1, :, :],
        affine,
        Cntd['out']['fresomu'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)),
        flip=imupd['flip'])

    # nimpa.imscroll(fresomu, view='t')

    # > scale down
    resd = ndi.zoom(reso, 1 / Cntd['scld'], output=None, order=1, mode='constant')

    # > affine
    imxys = resd.shape[2]
    imzys = resd.shape[0]
    affined = np.array(
        [[-Cntd['vxsz'] * Cntd['scld'], 0., 0., .5 * imxys * Cntd['vxsz'] * Cntd['scld']],
         [0., Cntd['vxsz'] * Cntd['scld'], 0., -.5 * imxys * Cntd['vxsz'] * Cntd['scld']],
         [0., 0., Cntd['vxsz'] * Cntd['scld'], -.5 * imzys * Cntd['vxsz'] * Cntd['scld']],
         [0., 0., 0., 1.]])

    nimpa.array2nii(
        resd, affined, Cntd['out']['fresdmu'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    # > joining bits together for rods in water
    resoW = np.concatenate((renW, resW, renW, buff), axis=0)
    # > padding for even dims
    # resoW = np.pad( resoW, ((1,0), (0,1),(0,1)), constant_values=((0,0), (0,0), (0, 0)) )
    resoW = np.pad(resoW,
                   ((1, 0), (Cntd['dpad'], Cntd['dpad'] + 1), (Cntd['dpad'], Cntd['dpad'] + 1)),
                   constant_values=((0, 0), (0, 0), (0, 0)))

    # > scale down
    resdW = ndi.zoom(resoW, 1 / Cntd['scld'], output=None, order=1, mode='constant')

    nimpa.array2nii(
        resdW, affined, Cntd['out']['fresdWmu'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    # > FOR registration using QNT reconstruction

    # > the extra part of the background at the bottom of the resolution bit after truncation.
    renB = renW[:bsz, ...].copy()
    renB[renB > 0] = 100

    renW[renW > Cntd['mu_water']] = 0
    renW[renW == Cntd['mu_water']] = 100

    resW[resW > Cntd['mu_water']] = 0
    resW[resW == Cntd['mu_water']] = 100

    # > joining bits together for rods in water
    resoW = np.concatenate((renW, resW, renW, renB), axis=0)
    # resoW = np.concatenate((renW, resW, renW), axis=0)

    # > padding for even dims
    resoW = np.pad(resoW,
                   ((1, 0), (Cntd['dpad'], Cntd['dpad'] + 1), (Cntd['dpad'], Cntd['dpad'] + 1)),
                   constant_values=((0, 0), (0, 0), (0, 0)))

    # > scale down
    resdW = ndi.zoom(resoW, 1 / Cntd['scld'], output=None, order=1, mode='constant')

    nimpa.array2nii(
        resdW, affined, Cntd['out']['fresdQmu'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    if return_raw:
        return {'resoW': resoW, 'resdW': resdW, 'resd': resd, 'reso': reso}


# =======================================================================
# >  S A M P L I N G   T E M P L A T E S
# =======================================================================


def create_sampl_reso(Cntd, return_raw=False):
    '''
    Create template of sampling rings for the resolution rods.
    Arguments:

    Cntd  - dictionary of constants, e.g.:
            spth  - output path of the folder used for storing the sampling
            templates
            dpth  - input path of the folder with the design in PNG format
            vxsz  - target voxel size
            scl   - scale factor from the original PNG voxel size to the target
            intord- interpolation order (0-nearest neighbour, etc.).
            rods_rotate - the initial rotation needed for accurately registering
                    the rods as they were placed during the phantom scan.
    imupd - dictionary of the trimmed and upsampled PET image used
            as references.
    '''

    if os.path.isfile(Cntd['out']['fst_res']):
        return None

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = nimpa.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed ACR QNT PET image cannot be found')

    rods_png = imageio.imread(Cntd['fs_rods'])

    gamma = 1.
    y = np.int16(0.2126 * rods_png[..., 0]**gamma + 0.7152 * rods_png[..., 1]**gamma +
                 0.0722 * rods_png[..., 2]**gamma)

    # > unique values 11,10,9,7,6,4
    uy = np.unique(y)

    # > levels of rings for each rod diameter
    o = 6
    k = o
    lvls = []
    lvls.append(uy[:k])
    lvls.append(uy[k:k + o])
    k += o
    lvls.append(uy[k:k + o])
    k += o
    lvls.append(uy[k:k + o])
    k += o
    o = 5
    lvls.append(uy[k:k + o])
    k += o
    lvls.append(uy[k:k + o])
    k += o
    o = 4
    lvls.append(uy[k:k + o])
    k += o
    o = 3
    lvls.append(uy[k:k + o])
    k += o
    lvls.append(uy[k:k + o])
    k += o
    o = 2
    lvls.append(uy[k:k + o])
    k += o
    o = 1
    lvls.append(uy[k:k + o])
    k += o
    lvls.append(uy[k:k + o])
    k += o

    g = -1 * np.ones((6, len(lvls) - 1), dtype=np.int16)
    for i in range(len(lvls) - 1):
        lvls[i]
        # > go through 6 rod diameters
        g[:len(lvls[i]), i] = lvls[i]

    g = g[::-1, :]

    srods = np.zeros(rods_png[..., 0].shape, dtype=np.uint16)

    for i, off in enumerate([10, 20, 30, 40, 50, 60]):
        for j, lvl in enumerate(g[i, g[i, :] > 0]):
            srods[y == lvl] = off + j

    if Cntd['rods_rotate'] != 0:
        srods = ndi.rotate(srods, Cntd['rods_rotate'], reshape=False, order=Cntd['intord'],
                           mode='constant')

    srods = ndi.zoom(srods, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    # > rod ends
    k = int(np.round(Cntd['k_rodsend'] / Cntd['vxsz']))
    srend = np.zeros((k,) + srods.shape, dtype=np.uint8)

    # > rods themselves
    k = int(np.round(Cntd['k_rods'] / Cntd['vxsz']))
    srods = np.repeat(srods[None, ...], k, axis=0)

    sres = np.concatenate((srend, srods, srend), axis=0)
    sres = np.pad(sres, ((1, 0), (0, 1), (0, 1)), constant_values=((0, 0), (0, 0), (0, 0)))

    imxys = sres.shape[2]
    imzys = sres.shape[0]
    affine = np.array([[-Cntd['vxsz'], 0., 0., .5 * imxys * Cntd['vxsz']],
                       [0., Cntd['vxsz'], 0., -.5 * imxys * Cntd['vxsz']],
                       [0., 0., Cntd['vxsz'], -.5 * imzys * Cntd['vxsz']], [0., 0., 0., 1.]])

    nimpa.array2nii(
        sres, affine, Cntd['out']['fst_res'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    if return_raw:
        del srods, srend
        return sres
    else:
        del srods, sres, srend
        return None


def create_sampl(Cntd, return_raw=False):
    """Create template of sampling rings for the inserts."""
    if all(os.path.isfile(Cntd['out'][f] for f in ('fst_insrt', 'fst_insrt3', 'fst_ibckg'))):
        if return_raw:
            return {
                'allsmplng': Cntd['out']['fst_insrt'], 'insrt3smplng': Cntd['out']['fst_insrt3'],
                'ibckgsmplng': Cntd['out']['fst_ibckg']}
        return None

    if 'fqntup' in Cntd and Path(Cntd['fqntup']).is_file():
        imupd = nimpa.getnii(Cntd['fqntup'], output='all')
    else:
        raise ValueError('Upscaled and trimmed ACR QNT PET image cannot be found')

    # Unused for now:
    # bckg_png = imageio.imread(Cntd['fs_bckg'])
    # ibckg_png = imageio.imread(Cntd['fs_ibckg'])
    # air_png = imageio.imread(Cntd['fs_air'])
    # h2o_png = imageio.imread(Cntd['fs_h2o'])
    # bone_png = imageio.imread(Cntd['fs_bone'])
    # hot1_png = imageio.imread(Cntd['fs_hot1'])
    # hot2_png = imageio.imread(Cntd['fs_hot2'])
    # hot3_png = imageio.imread(Cntd['fs_hot3'])
    # hot4_png = imageio.imread(Cntd['fs_hot4'])

    # > converted and combined in fewer templates
    insrts = None

    tmpl = (k for k in Cntd if k[:3] == 'fs_')
    for k in tmpl:
        off = Cntd['soff_' + k[3:]]
        if off is None: continue
        print('file: {} >> offset = {}'.format(k, off))

        impng = imageio.imread(Cntd[k])

        # > array for conversion of labels
        arr = np.zeros(impng[..., 0].shape, dtype=np.uint16)

        for i, v in enumerate(np.unique(impng[..., 0])):
            if v < 255:
                arr += np.uint16((impng[..., 0] == v) * (off+i))

        # > assemble all the templates together apart from hot3 which is
        # > overlapping, hence separate
        if k[3:] == 'hot3':
            insrt3 = arr
        elif k[3:] == 'bckg':
            bckg = arr
        elif k[3:] == 'ibckg':
            ibckg = arr
        else:
            if insrts is None:
                insrts = arr
            else:
                insrts += arr

    # > AXIAL SPACING
    # > upper padding
    ku = int(
        sum(
            np.round(Cntd[k] / Cntd['vxsz'])
            for k in ('k_mcapA', 'k_mcapB', 'k_mcapC', 'k_mcapD', 'k_lidA', 'k_lidB')))
    ki = int(np.round(Cntd['k_insrtA'] / Cntd['vxsz']))

    # > insert bottoms
    kb = int(np.round(Cntd['k_insrtB'] / Cntd['vxsz']))

    # > uniform bit region
    kuni = int(np.round(Cntd['k_unfrm'] / Cntd['vxsz']))

    # > resolution region
    kr = int(np.round((Cntd['k_rods'] + 2 * Cntd['k_rodsend']) / Cntd['vxsz']))

    # > bottom fixed lid
    kl = int(np.round(Cntd['k_contB'] / Cntd['vxsz']))

    # > scale down the original PNG images
    insrts = ndi.zoom(insrts, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    insrt3 = ndi.zoom(insrt3, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    ibckg = ndi.zoom(ibckg, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')
    bckg = ndi.zoom(bckg, Cntd['scl'], output=None, order=Cntd['intord'], mode='constant')

    # > padding with zeros
    up_ = np.zeros((ku,) + insrts.shape, dtype=np.uint16)
    ibtm_ = np.zeros((kb,) + insrts.shape, dtype=np.uint16)
    btm_ = np.zeros((kl + kr,) + insrts.shape, dtype=np.uint16)

    # > repeat according to the axial dimensions
    insrts = np.repeat(insrts[None, ...], ki, axis=0)
    insrt3 = np.repeat(insrt3[None, ...], ki, axis=0)
    ibckg = np.repeat(ibckg[None, ...], ki, axis=0)
    bckg = np.repeat(bckg[None, ...], kuni, axis=0)

    allsmplng = np.concatenate((up_, insrts, ibtm_, bckg, btm_), axis=0)

    insrt3smplng = np.concatenate((up_, insrt3, ibtm_, bckg, btm_), axis=0)

    ibckgsmplng = np.concatenate((up_, ibckg, ibtm_, bckg, btm_), axis=0)

    # > pad the image to make the dims even
    allsmplng = np.pad(allsmplng, ((0, 0), (0, 1), (0, 1)),
                       constant_values=((0, 0), (0, 0), (0, 0)))
    insrt3smplng = np.pad(insrt3smplng, ((0, 0), (0, 1), (0, 1)),
                          constant_values=((0, 0), (0, 0), (0, 0)))
    ibckgsmplng = np.pad(ibckgsmplng, ((0, 0), (0, 1), (0, 1)),
                         constant_values=((0, 0), (0, 0), (0, 0)))

    # > make it up-side down (as it is scanned)
    allsmplng = allsmplng[::-1, :, :]
    insrt3smplng = insrt3smplng[::-1, :, :]
    ibckgsmplng = ibckgsmplng[::-1, :, :]

    # > get the affine and save to NIfTI
    imxys = ibckgsmplng.shape[2]
    imzys = ibckgsmplng.shape[0]
    affine = np.array([[-Cntd['vxsz'], 0., 0., .5 * imxys * Cntd['vxsz']],
                       [0., Cntd['vxsz'], 0., -.5 * imxys * Cntd['vxsz']],
                       [0., 0., Cntd['vxsz'], -.5 * imzys * Cntd['vxsz']], [0., 0., 0., 1.]])

    nimpa.array2nii(
        allsmplng, affine, Cntd['out']['fst_insrt'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    nimpa.array2nii(
        insrt3smplng, affine, Cntd['out']['fst_insrt3'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    nimpa.array2nii(
        ibckgsmplng, affine, Cntd['out']['fst_ibckg'],
        trnsp=(imupd['transpose'].index(0), imupd['transpose'].index(1),
               imupd['transpose'].index(2)), flip=imupd['flip'])

    if return_raw:
        return {'allsmplng': allsmplng, 'insrt3smplng': insrt3smplng, 'ibckgsmplng': ibckgsmplng}
