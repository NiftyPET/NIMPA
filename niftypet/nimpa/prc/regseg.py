"""
NIMPA: functions for neuro image processing and analysis.
Includes functions relating to image registration/segmentation.
"""
import glob
import logging
import os
import shutil
import sys
from subprocess import call
from textwrap import dedent

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from dipy.align import _public as align
from dipy.align import affine_registration, center_of_mass, rigid, translation
from miutil.fdio import hasext
from spm12.regseg import resample_spm  # NOQA: F401 yapf: disable
from spm12.regseg import coreg_spm

from .. import resources as rs
from . import imio, prc

log = logging.getLogger(__name__)


def imfill(immsk):
    '''fill the empty patches of image mask 'immsk' '''
    for iz in range(immsk.shape[0]):
        for iy in range(immsk.shape[1]):
            ix0 = np.argmax(immsk[iz, iy, :] > 0)
            ix1 = immsk.shape[2] - np.argmax(immsk[iz, iy, ::-1] > 0)
            if (ix1 - ix0) > immsk.shape[2] - 10: continue
            immsk[iz, iy, ix0:ix1] = 1
    return immsk


# ------------------------------------------------------------------------------
# Create object mask for the input image
# ------------------------------------------------------------------------------


def create_mask(
    fnii,
    fimout='',
    outpath='',
    fill=1,
    dtype_fill=np.uint8,
    thrsh=0.,
    fwhm=0.,
):
    '''create mask over the whole image or over the threshold area'''
    # > output path
    if outpath == '' and fimout != '':
        opth = os.path.dirname(fimout)
        if opth == '':
            opth = os.path.dirname(fnii)
            fimout = os.path.join(opth, fimout)

    elif outpath == '':
        opth = os.path.dirname(fnii)

    else:
        opth = outpath

    # > output file name if not given
    if fimout == '':
        fniis = os.path.split(fnii)
        fimout = os.path.join(opth, fniis[1].split('.nii')[0] + '_mask.nii.gz')

    niidct = imio.getnii(fnii, output='all')
    im = niidct['im']
    hdr = niidct['hdr']

    if im.ndim > 3:
        raise ValueError('The masking function only accepts 3-D images.')

    # > generate output image
    if thrsh > 0.:
        smoim = ndi.filters.gaussian_filter(im, imio.fwhm2sig(fwhm, voxsize=abs(hdr['pixdim'][1])),
                                            mode='mirror')
        thrsh = thrsh * smoim.max()
        immsk = np.int8(smoim > thrsh)
        immsk = imfill(immsk)

        # > output image
        imo = fill * immsk.astype(dtype_fill)
    else:
        imo = fill * np.ones(im.shape, dtype=dtype_fill)

    # > save output image
    imio.array2nii(
        imo, niidct['affine'], fimout,
        trnsp=(niidct['transpose'].index(0), niidct['transpose'].index(1),
               niidct['transpose'].index(2)), flip=niidct['flip'])

    return {'fim': fimout, 'im': imo}


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# I M A G E   R E G I S T R A T I O N
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def affine_dipy(
    fref,
    fflo,
    nbins=32,
    metric='MI',
    level_iters=None,
    sigmas=None,
    factors=None,
    outpath=None,
    faffine=None,
    pickname='ref',
    fcomment='',
    rfwhm=15.,
    ffwhm=15.,
    verbose=True,
):
    """
    Perform affine 3-stage image registration using DIPY.

    Arguments:
      fref: file path to the reference image to which the floating image
        will be registered.
      fflo: file path to the floating image to be registered to the
        reference image.
      outpath: folder path for the output
    """
    if level_iters is None:
        level_iters = [10000, 1000, 200]
    if sigmas is None:
        sigmas = [3.0, 1.0, 0.0]
    if factors is None:
        factors = [4, 2, 1]
    # create a folder for images registered to ref
    if outpath is None:
        odir = os.path.join(os.path.dirname(fflo), 'affine-dipy')
    else:
        odir = os.path.join(outpath, 'affine-dipy')
    imio.create_dir(odir)

    # output in register with ref and text file for the affine transform
    if faffine is not None:
        faff = faffine
    else:
        if pickname == 'ref':
            faff = os.path.join(
                odir,
                'affine_dipy_ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.npy')
        elif pickname == 'flo':
            faff = os.path.join(
                odir,
                'affine_dipy_flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.npy')

    # > smoothing if needed:
    if rfwhm > 0.:
        fstatic = os.path.basename(fref).split('.nii')[0] + '_smth' + str(rfwhm).replace(
            '.', '-') + 'mm.nii.gz'
        fstatic = os.path.join(odir, fstatic)
        _ = prc.imsmooth(fref, fwhm=rfwhm, fout=fstatic)
    else:
        fstatic = fref

    if ffwhm > 0.:
        fmoving = os.path.basename(fflo).split('.nii')[0] + '_smth' + str(ffwhm).replace(
            '.', '-') + 'mm.nii.gz'
        fmoving = os.path.join(odir, fmoving)
        _ = prc.imsmooth(fflo, fwhm=ffwhm, fout=fmoving)
    else:
        fmoving = fflo

    # ------------------------------------------------------------------
    pipeline = [center_of_mass, translation, rigid]

    txim, txaff = affine_registration(fmoving, fstatic, nbins=nbins, metric=metric,
                                      pipeline=pipeline, level_iters=level_iters, sigmas=sigmas,
                                      factors=factors)
    # ------------------------------------------------------------------
    np.save(faff, txaff)
    return {'affine': txaff, 'faff': faff}


def resample_dipy(
    fref,
    fflo,
    faff=None,
    outpath=None,
    fimout=None,
    fcomment='',
    pickname='ref',
    intrp=1,
    dtype_nifti=np.float32,
    verbose=True,
):
    """
    Resample image fflo to the reference fref. Image needs to be given as NIfTI files.
    Arguments:
      faff: Numpy array or Numpy file (*.npy) for the affine transformation.
      intrp: interpolation: 0-NN, 1-linear
    """

    # > OUTPUT
    # ------------------------------------------------------------------
    # > output path
    if outpath is None and fimout is not None:
        opth = os.path.dirname(fimout)
        if opth == '':
            opth = os.path.dirname(fflo)
    elif outpath is None or not os.path.isdir(outpath):
        opth = os.path.dirname(fflo)
    else:
        opth = outpath

    imio.create_dir(opth)

    # > the output naming
    if os.path.dirname(fimout) != '':
        fout = fimout
    elif fimout is not None and not os.path.isfile(fimout):
        fout = os.path.join(opth, fimout)
    elif pickname == 'ref':
        fout = os.path.join(
            opth, 'resampled-dipy_to_ref-' + os.path.basename(fref).split('.nii')[0] + fcomment +
            '.nii.gz')
    elif pickname == 'flo':
        fout = os.path.join(
            opth, 'resampled-dipy_to_flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment +
            '.nii.gz')
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    if faff is not None:
        if faff.endswith('.npy'):
            affine = np.load(faff)
        elif isinstance(faff, np.ndarray):
            affine = faff
        else:
            raise ValueError('e> unrecognised affine matrix input')
    else:
        affine = None
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    if intrp == 0:
        interpolation = 'nearest'
    elif intrp == 1:
        interpolation = 'linear'
    else:
        raise ValueError('e> unrecognised interpolation input')
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # > DIPY RESAMPLING
    # ------------------------------------------------------------------
    static, static_affine, moving, moving_affine, between_affine = align._handle_pipeline_inputs(
        fflo, fref, moving_affine=None, static_affine=None, starting_affine=affine)
    affine_map = align.AffineMap(between_affine, static.shape, static_affine, moving.shape,
                                 moving_affine)
    rsmpl = affine_map.transform(moving, interpolation=interpolation)

    del moving, static

    # > save to NIfTI
    smpl_nii = nib.Nifti1Image(rsmpl.astype(dtype_nifti), static_affine)
    nib.save(smpl_nii, fout)
    return {'fnii': fout, 'im': rsmpl}


def affine_niftyreg(
    fref,
    fflo,
    outpath='',
    fname_aff='',
    pickname='ref',
    fcomment='',
    executable='',
    omp=1,
    rigOnly=False,
    affDirect=False,
    maxit=5,
    speed=True,
    pi=50,
    pv=50,
    smof=0,
    smor=0,
    rmsk=True,
    fmsk=True,
    rfwhm=15.,
    rthrsh=0.05,
    ffwhm=15.,
    fthrsh=0.05,
    verbose=True,
):

    # check if the executable exists:
    if not os.path.isfile(executable):
        raise IOError('Incorrect path to executable file for registration.')

    # create a folder for images registered to ref
    if outpath != '':
        odir = os.path.join(outpath, 'affine-niftyreg')
        fimdir = os.path.join(outpath, os.path.join('affine-niftyreg', 'mask'))
    else:
        odir = os.path.join(os.path.dirname(fflo), 'affine-niftyreg')
        fimdir = os.path.join(os.path.dirname(fflo), 'affine-niftyreg', 'mask')
    imio.create_dir(odir)
    imio.create_dir(fimdir)

    if rmsk:
        f_rmsk = os.path.join(fimdir,
                              'rmask_' + os.path.basename(fref).split('.nii')[0] + '.nii.gz')
        create_mask(fref, fimout=f_rmsk, thrsh=rthrsh, fwhm=rfwhm)

    if fmsk:
        f_fmsk = os.path.join(fimdir,
                              'fmask_' + os.path.basename(fflo).split('.nii')[0] + '.nii.gz')
        create_mask(fflo, fimout=f_fmsk, thrsh=fthrsh, fwhm=ffwhm)

    # output in register with ref and text file for the affine transform
    if fname_aff != '':
        fout = os.path.join(odir, fname_aff.split('.')[0] + '.nii.gz')
        faff = os.path.join(odir, fname_aff)
    else:
        if pickname == 'ref':
            fout = os.path.join(
                odir,
                'affine_ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.nii.gz')
            faff = os.path.join(
                odir, 'affine_ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.txt')
        elif pickname == 'flo':
            fout = os.path.join(
                odir,
                'affine_flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.nii.gz')
            faff = os.path.join(
                odir, 'affine_flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.txt')

    # call the registration routine
    cmd = [
        executable, '-ref', fref, '-flo', fflo, '-aff', faff, '-pi',
        str(pi), '-pv',
        str(pv), '-smooF',
        str(smof), '-smooR',
        str(smor), '-maxit', '10', '-omp',
        str(omp), '-res', fout]
    if speed:
        cmd.append('-speeeeed')
    if rigOnly:
        cmd.append('-rigOnly')
    if affDirect:
        cmd.append('affDirect')
    if rmsk:
        cmd.append('-rmask')
        cmd.append(f_rmsk)
    if fmsk:
        cmd.append('-fmask')
        cmd.append(f_fmsk)
    if not verbose:
        cmd.append('-voff')

    call(cmd)

    # > affine to Numpy array
    aff = np.loadtxt(faff)

    return {'affine': aff, 'faff': faff, 'fim': fout} # faff, fout


def resample_niftyreg(
    fref,
    fflo,
    faff,
    outpath='',
    fimout='',
    fcomment='',
    pickname='ref',
    intrp=1,
    executable='',
    verbose=True,
):

    # check if the executable exists:
    # if executable=='' and 'RESPATH' in Cnt and os.path.isfile(Cnt['RESPATH']):
    #     executable = Cnt['RESPATH']

    if not os.path.isfile(executable):
        raise IOError('Incorrect path to executable file for registration.')

    # > output path
    if outpath == '' and fimout != '':
        opth = os.path.dirname(fimout)
        if opth == '':
            opth = os.path.dirname(fflo)
    elif outpath == '':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath

    imio.create_dir(opth)

    # > the output naming
    if '/' in fimout:
        fout = fimout
    elif fimout != '' and not os.path.isfile(fimout):
        fout = os.path.join(opth, fimout)
    elif pickname == 'ref':
        fout = os.path.join(
            opth, 'affine_ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.nii.gz')
    elif pickname == 'flo':
        fout = os.path.join(
            opth, 'affine_flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.nii.gz')

    if isinstance(intrp, int):
        intrp = str(intrp)

    cmd = [executable, '-ref', fref, '-flo', fflo, '-trans', faff, '-res', fout, '-inter', intrp]
    if not verbose:
        cmd.append('-voff')
    call(cmd)

    return fout


# ==============================================================================
# S P M realignment of multiple images through m-scripting (dynamic PET)
# ==============================================================================


def realign_mltp_spm(
    fims,
    quality=1.0,
    fwhm=6,
    sep=4,
    rtm=1,
    interp=2,
    graph=0,
    outpath='',
    fcomment='',
    niicopy=False,
    niisort=False,
):
    '''
    Arguments:
        fims:   has to be a list of at least two files with the first one acting
                as a reference.
    '''
    # > input folder
    inpath = os.path.dirname(fims[0])

    # > output folder
    if outpath == '':
        outpath = os.path.join(inpath, 'align')
    else:
        outpath = os.path.join(outpath, 'align')

    if fims[0][-3:] == '.gz' or niicopy:
        tmpth = outpath # os.path.join(outpath, 'tmp')
        rpth = tmpth
    else:
        tmpth = outpath
        rpth = inpath

    imio.create_dir(tmpth)

    # > uncompress for SPM
    fungz = []
    for f in fims:
        if f[-3:] == '.gz':
            fun = imio.nii_ugzip(f, outpath=tmpth)
        elif os.path.isfile(f) and f.endswith('nii'):
            if niicopy:
                fun = os.path.join(tmpth, os.path.basename(f).split('.nii')[0] + '_copy.nii')
                shutil.copyfile(f, fun)
            else:
                fun = f
        else:
            log.warning('omitting file/folder: {}'.format(f))
        fungz.append(fun)

    if niisort:
        niisrt = imio.niisort([
            os.path.join(tmpth, f) for f in os.listdir(tmpth)
            if os.path.isfile(os.path.join(tmpth, f))])
        fsrt = niisrt['files']
    else:
        fsrt = fungz

    P_input = [f + ',1' for f in fsrt if f.endswith('nii') and f[0] != 'r' and 'mean' not in f]

    # > maximal number of characters per line (for Matlab array)
    Pinmx = max(map(len, P_input))

    # > equalise the input char array
    Pineq = [f.ljust(Pinmx) for f in P_input]

    # --------------------------------------------------------------------------
    # > MATLAB realign flags for SPM
    flgs = []
    pw = '\'\''

    flgs.append('flgs.quality = ' + str(quality))
    flgs.append('flgs.fwhm = ' + str(fwhm))
    flgs.append('flgs.sep = ' + str(sep))
    flgs.append('flgs.rtm = ' + str(rtm))
    flgs.append('flgs.pw  = ' + str(pw))
    flgs.append('flgs.interp = ' + str(interp))
    flgs.append('flgs.graphics = ' + str(graph))
    flgs.append('flgs.wrap = [0 0 0]')

    flgs.append('disp(flgs)')
    # --------------------------------------------------------------------------

    fscript = os.path.join(outpath, 'pyauto_script_realign.m')

    with open(fscript, 'w') as f:
        f.write('% AUTOMATICALLY GENERATED MATLAB SCRIPT FOR SPM PET REALIGNMENT\n\n')

        f.write('%> the following flags for PET image alignment are used:\n')
        f.write("disp('m> SPM realign flags:');\n")
        for item in flgs:
            f.write("%s;\n" % item)

        f.write('\n%> the following PET images will be aligned:\n')
        f.write("disp('m> PET images for SPM realignment:');\n")

        f.write('P{1,1} = [...\n')
        for item in Pineq:
            f.write("'%s'\n" % item)
        f.write('];\n\n')
        f.write('disp(P{1,1});\n')

        f.write('spm_realign(P, flgs);')

    cmd = ['matlab', '-nodisplay', '-nodesktop', '-r', 'run(' + '\'' + fscript + '\'' + '); exit']
    call(cmd)

    fres = glob.glob(os.path.join(rpth, 'rp*.txt'))[0]
    res = np.loadtxt(fres)

    return {'fout': fres, 'outarray': res, 'P': Pineq, 'fims': fsrt}


# ==============================================================================
# S P M resampling of multiple images through m-scripting (dynamic PET)
# ==============================================================================


def resample_mltp_spm(
    fims,
    ftr,
    interp=1.,
    which=1,
    mask=0,
    mean=0,
    graph=0,
    niisort=False,
    prefix='r_',
    outpath='',
    fcomment='',
    pickname='ref',
    copy_input=False,
    del_in_uncmpr=False,
    del_out_uncmpr=False,
):
    '''
    Arguments:
        fims:   has to be a list of at least two files with the first one acting
                as a reference.
    '''
    if not isinstance(fims, list) and not isinstance(fims[0], str):
        raise ValueError('e> unrecognised list of input images')

    if not os.path.isfile(ftr):
        raise IOError('e> cannot open the file with translations and rotations')

    # > output path
    if outpath == '':
        opth = os.path.dirname(fims[0])
    else:
        opth = outpath

    imio.create_dir(opth)

    # > working file names (not touching the original ones)
    _fims = []

    # > decompress if necessary
    for f in fims:
        if not os.path.isfile(f) and not (f.endswith('nii') or f.endswith('nii.gz')):
            raise IOError('e> could not open file:', f)

        if f[-3:] == '.gz':
            fugz = imio.nii_ugzip(f, outpath=os.path.join(opth, 'copy'))
        elif copy_input:
            fnm = os.path.basename(f).split('.nii')[0] + '_copy.nii'
            fugz = os.path.join(opth, 'copy', fnm)
            shutil.copyfile(f, fugz)
        else:
            fugz = f

        _fims.append(fugz)

    if niisort:
        niisrt = imio.niisort(_fims)
        _fims = niisrt['files']

    # > maximal number of characters per line (for Matlab array)
    Pinmx = max(map(len, _fims))

    # > equalise the input char array
    Pineq = [f.ljust(Pinmx) for f in _fims]

    # --------------------------------------------------------------------------
    # > SPM reslicing (resampling) flags
    flgs = []

    flgs.append('flgs.mask = ' + str(mask))
    flgs.append('flgs.mean = ' + str(mean))

    flgs.append('flgs.which = ' + str(which))
    flgs.append('flgs.interp = ' + str(interp))
    flgs.append('flgs.graphics = ' + str(graph))
    flgs.append('flgs.wrap = [0 0 0]')
    flgs.append('flgs.prefix = ' + "'" + prefix + "'")

    flgs.append('disp(flgs)')
    # --------------------------------------------------------------------------

    fsrpt = os.path.join(opth, 'pyauto_script_resampling.m')
    with open(fsrpt, 'w') as f:
        f.write('% AUTOMATICALLY GENERATED MATLAB SCRIPT FOR SPM RESAMPLING PET IMAGES\n\n')

        f.write('%> the following flags for image reslicing are used:\n')
        f.write("disp('m> SPM reslicing flags:');\n")
        for item in flgs:
            f.write("%s;\n" % item)

        f.write('\n%> the following PET images will be aligned:\n')
        f.write("disp('m> PET images for SPM reslicing:');\n")

        f.write('P{1,1} = [...\n')
        for item in Pineq:
            f.write("'%s'\n" % item)
        f.write('];\n\n')
        f.write('disp(P{1,1});\n')

        # f.write('\n%> the following PET images will be aligned using'
        #         ' the translations and rotations in X:\n')
        # f.write("X = dlmread('"+ftr+"');\n")
        # f.write('for fi = 2:'+str(len(fims))+'\n')
        # f.write("    VF = strcat(P{1,1}(fi,:),',1');\n")
        # f.write('    M_ = zeros(4,4);\n')
        # f.write('    M_(:,:) = spm_get_space(VF);\n')
        # f.write('    M = spm_matrix(X(fi,:));\n')
        # f.write('    spm_get_space(VF, M\M_(:,:));\n')
        # f.write('end\n\n')

        f.write('spm_reslice(P, flgs);\n')

    cmd = ['matlab', '-nodisplay', '-nodesktop', '-r', 'run(' + '\'' + fsrpt + '\'' + '); exit']
    call(cmd)

    if del_in_uncmpr:
        for fpth in _fims:
            os.remove(fpth)


# ==============================================================================
# V I N C I  registration and resampling
# ==============================================================================


def coreg_vinci(
    fref,
    fflo,
    vc='',
    con='',
    vincipy_path='',
    scheme_xml='',
    outpath='',
    fname_aff='',
    pickname='ref',
    fcomment='',
    flo_colourmap='Green',
    close_vinci=False,
    close_buff=True,
    cleanup=True,
    save_res=False,
):
    if scheme_xml == '':
        raise IOError(
            dedent("""\
            the Vinci schema *.xml file is not provided.
            please add the schema file in the call: scheme_xml=..."""))

    # --------------------------------------------------------------------------
    # > output path
    if outpath == '' and fname_aff != '' and '/' in fname_aff:
        opth = os.path.dirname(fname_aff)
        if opth == '':
            opth = os.path.dirname(fflo)
        fname_aff = os.path.basename(fname_aff)

    elif outpath == '':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    imio.create_dir(os.path.join(opth, 'affine-vinci'))

    # > output floating and affine file names
    if fname_aff == '':
        if pickname == 'ref':

            faff = os.path.join(
                opth, 'affine-vinci',
                'affine-ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.xml')

            fout = os.path.join(
                opth, 'affine-vinci',
                'affine-ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.nii.gz')

        else:
            faff = os.path.join(
                opth, 'affine-vinci',
                'affine-flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.xml')

            fout = os.path.join(
                opth, 'affine-vinci',
                'affine-flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.nii.gz')

    else:

        # > add '.xml' extension if not in the affine output file name
        if not fname_aff.endswith('.xml'):
            fname_aff += '.xml'

        faff = os.path.join(opth, 'affine-vinci', fname_aff)

        fout = os.path.join(opth, fname_aff.split('.')[0] + '.nii.gz')
    # --------------------------------------------------------------------------

    if vincipy_path == '':
        try:
            vincipy_path = rs.VINCIPATH
        except AttributeError:
            raise AttributeError('e> could not find variable VINCIPATH in resources.py')

    sys.path.append(vincipy_path)

    try:
        from VinciPy import Vinci_Bin, Vinci_Connect, Vinci_Core, Vinci_ImageT, Vinci_XML
    except ImportError:
        raise ImportError(
            dedent("""\
            could not import Vinci:
            check the variable VINCIPATH (path to Vinci) in resources.py"""))

    # > start Vinci core engine if it is not running/given
    if vc == '' or con == '':
        # > adopted from the Vinci's example RunMMMJob.py
        bin = Vinci_Bin.Vinci_Bin()
        con = Vinci_Connect.Vinci_Connect(bin)
        con.StartMyVinci()

        vc = Vinci_Core.Vinci_CoreCalc(con)
        vc.StdProject()

    # > read the registration schema file
    f = open(scheme_xml, 'rb')
    reg_scheme = f.read()
    f.close()

    # > check scheme data file and remove possible XML comments at its beginning
    root = Vinci_XML.ElementTree.fromstring(reg_scheme)
    if root.tag != "MMM":
        sys.exit("scheme data file %s does not contain tag MMM\n" % scheme_xml)
    reg_scheme = Vinci_XML.ElementTree.tostring(root)
    # bytes -> str
    reg_scheme = reg_scheme.decode("utf-8")

    # > pick reference and floating images
    ref = Vinci_ImageT.newTemporary(vc, szFileName=fref)
    rsl = Vinci_ImageT.newTemporary(vc, szFileName=fflo)

    # > set the colour map for the floating image
    rsl.setColorSettings(CTable=flo_colourmap)

    # --------------------------------------------------------------------------
    # > perform the registration
    rsl.alignToRef(ref, reg_scheme, szRegistrationSummaryFile=faff)
    # --------------------------------------------------------------------------

    # > find the weird folder name created and to be disposed of.
    if cleanup:
        # > get the folder name (going to rubbish)
        del_dir = os.path.normpath(faff).split(os.sep)[1]
        del_dir = os.path.join(os.path.split(fflo)[0], os.path.normpath(faff).split(os.sep)[1])
        if os.path.isdir(del_dir):
            shutil.rmtree(del_dir)

    # > save the registered image
    if save_res:
        rsl.saveYourselfAs(bUseOffsetRotation=True, szFullFileName=fout)

    # > close image buffers for reference and floating
    if close_buff:
        rsl.killYourself()
        ref.killYourself()

    if close_vinci: con.CloseVinci(True)

    return {'faff': faff, 'fim': fout, 'vinci_con': con, 'vinci_vc': vc}


# ------------------------------------------------------------------------------
# VINCI RESAMPLE
# ------------------------------------------------------------------------------


def resample_vinci(
    fref,
    fflo,
    faff,
    intrp=0,
    fimout='',
    fcomment='',
    outpath='',
    pickname='ref',
    vc='',
    con='',
    vincipy_path='',
    atlas_resample=False,
    atlas_ref_make=False,
    atlas_ref_del=False,
    close_vinci=False,
    close_buff=True,
):
    '''
    Resample the floating image <fflo> into the geometry of <fref>,
    using the Vinci transformation output <faff> (an *.xml file).
    Output the NIfTI file path of the resampled/resliced image.
    '''
    # > output path
    if outpath == '' and fimout != '' and '/' in fimout:
        opth = os.path.dirname(fimout)
        if opth == '':
            opth = os.path.dirname(fflo)
        fimout = os.path.basename(fimout)

    elif outpath == '':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    # > output floating and affine file names
    if fimout == '':
        if pickname == 'ref':
            fout = os.path.join(
                opth,
                'affine-ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.nii.gz')
        else:
            fout = os.path.join(
                opth,
                'affine-flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.nii.gz')
    else:
        fout = os.path.join(opth, fimout.split('.')[0] + '.nii.gz')
    # --------------------------------------------------------------------------

    if vincipy_path == '':
        try:
            vincipy_path = rs.VINCIPATH
        except AttributeError:
            raise AttributeError('e> could not find variable VINCIPATH in resources.py')

    sys.path.append(vincipy_path)

    try:
        from VinciPy import Vinci_Bin, Vinci_Connect, Vinci_Core, Vinci_ImageT
    except ImportError:
        raise ImportError(
            dedent("""\
            could not import Vinci:
            check the variable VINCIPATH (path to Vinci) in resources.py"""))

    # --------------------------------------------------------------------------
    # > start Vinci core engine if it is not running/given
    if vc == '' or con == '':
        # > adopted from the Vinci's example RunMMMJob.py
        bin = Vinci_Bin.Vinci_Bin()
        con = Vinci_Connect.Vinci_Connect(bin)
        con.StartMyVinci()

        vc = Vinci_Core.Vinci_CoreCalc(con)
        vc.StdProject()
    # --------------------------------------------------------------------------

    if int(intrp) == 0:
        interp_nn = True
    else:
        interp_nn = False

    # --------------------------------------------------------------------------
    # > change the reference image to be loaded as atlas
    if atlas_resample and atlas_ref_make:

        # > output file name for the extra reference image
        fextref = os.path.join(opth, 'reference-as-atlas.nii.gz')

        prc.nii_modify(fref, fimout=fextref, voxel_range=[0., 255.])

        ref = Vinci_ImageT.newTemporary(vc, szFileName=fextref, bLoadAsAtlas=atlas_resample)

        if atlas_ref_del:
            os.remove(fextref)

    else:
        ref = Vinci_ImageT.newTemporary(vc, szFileName=fref, bLoadAsAtlas=atlas_resample)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # > reapply rigid transformation to new image

    flo = Vinci_ImageT.newTemporary(vc, szFileName=fflo, bLoadAsAtlas=atlas_resample)

    rsl = flo.reapplyMMMTransform(faff, refImage=ref, IsComputed=True)

    rsl.saveYourselfAs(bUseOffsetRotation=True, bUseNextNeighbourInterp=interp_nn,
                       szFullFileName=fout)
    # --------------------------------------------------------------------------

    # > close image buffers for reference and floating
    if close_buff:
        ref.killYourself()
        flo.killYourself()
        rsl.killYourself()

    if close_vinci: con.CloseVinci(True)

    return fout


def affine_fsl(
    fref,
    fflo,
    outpath='',
    fname_aff='',
    pickname='ref',
    fcomment='',
    executable='',
    costfun='normmi',
    dof=6,
    hstbins=256,
    verbose=True,
):
    """
    FSL-FLIRT REGISTRATION (AFFINE/RIGID)
    """
    if executable == '':
        if 'FSLDIR' not in os.environ:
            raise IOError('e> no FSL executable provided!')
        else:
            executable = os.path.join(os.environ['FSLDIR'], 'bin', 'flirt')

    if not os.path.isfile(executable) or not call([executable, '-version']) == 0:
        raise IOError('e> no valid FSL executable provided!')

    # --------------------------------------------------------------------------
    # > output path
    if outpath == '' and fname_aff != '' and os.path.isfile(fname_aff):
        opth = os.path.dirname(fname_aff)
        if opth == '':
            opth = os.path.dirname(fflo)
        fname_aff = os.path.basename(fname_aff)

    elif outpath == '':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    imio.create_dir(os.path.join(opth, 'affine-fsl'))

    # > output floating and affine file names
    if fname_aff == '':

        if pickname == 'ref':
            faff = os.path.join(
                opth, 'affine-fsl',
                'affine-ref-' + os.path.basename(fref).split('.nii')[0] + fcomment + '.txt')

        else:
            faff = os.path.join(
                opth, 'affine-fsl',
                'affine-flo-' + os.path.basename(fflo).split('.nii')[0] + fcomment + '.txt')

    else:

        # > add '.xml' extension if not in the affine output file name
        if not fname_aff.endswith('.txt'):
            fname_aff += '.txt'

        faff = os.path.join(opth, 'affine-vinci', fname_aff)
    # --------------------------------------------------------------------------

    # > command with options for FSL-FLIRT registration
    cmd = [
        executable, '-cost', costfun, '-dof',
        str(dof), '-omat', faff, '-in', fflo, '-ref', fref, '-bins',
        str(hstbins)]

    # > add verbose mode if requested
    if isinstance(verbose, bool): verbose = int(verbose)
    if verbose > 0: cmd.extend(['-verbose', str(verbose)])

    # > execute FSL-FLIRT command with the above options
    call(cmd)

    # convert to Numpy float
    aff = np.loadtxt(faff)

    return {'affine': aff, 'faff': faff}


# ------------------------------------------------------------------------------
# FSL RESAMPLING
# ------------------------------------------------------------------------------


def resample_fsl(
    imref,
    imflo,
    faff,
    outpath='',
    fimout='',
    pickname='ref',
    fcomment='',
    intrp=1,
    executable='',
):
    if executable == '':
        if 'FSLDIR' not in os.environ:
            raise IOError('e> no FSL executable provided!')
        else:
            executable = os.path.join(os.environ['FSLDIR'], 'bin', 'flirt')

    if not os.path.isfile(executable) or not call([executable, '-version']) == 0:
        raise IOError('e> no valid FSL executable provided!')

    log.info(
        dedent('''\
        ===================================================================
        F S L  resampling...
        ==================================================================='''))

    # > output path
    if outpath == '' and fimout != '':
        opth = os.path.dirname(fimout)
        if opth == '':
            opth = os.path.dirname(imflo)

    elif outpath == '':
        opth = os.path.dirname(imflo)
    elif outpath != '':
        opth = outpath

    imio.create_dir(opth)

    # > the output naming
    if '/' in fimout:
        fout = fimout
    elif fimout != '' and not os.path.isfile(fimout):
        fout = os.path.join(opth, fimout)
    elif pickname == 'ref':
        fout = os.path.join(
            opth, 'affine_ref-' + os.path.basename(imref).split('.nii')[0] + fcomment + '.nii.gz')
    elif pickname == 'flo':
        fout = os.path.join(
            opth, 'affine_flo-' + os.path.basename(imflo).split('.nii')[0] + fcomment + '.nii.gz')

    intrp = int(intrp)
    if isinstance(intrp, int): intrp = str(intrp)
    if intrp == '1':
        interpolation = 'trilinear'
    elif intrp == '0':
        interpolation = 'nearestneighbour'

    cmd = [
        executable, '-in', imflo, '-ref', imref, '-out', fout, '-init', faff, '-applyxfm',
        '-interp', interpolation]
    call(cmd)

    log.info(
        dedent('''\
        ===================================================================
        F S L  resampling done.
        ==================================================================='''))

    return fout


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# M O T I O N
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def motion_reg(
    ref,
    flo,
    fcomment='',
    pickname='ref',
    outpath='',
    fname_aff='',
    matlab_eng_name='',
    rot_thresh=1.,
    trn_thresh=1.,
):
    if isinstance(flo, str):
        flolst = [flo]
    elif isinstance(flo, list) and all(os.path.isfile(f) for f in flo):
        flolst = flo
    else:
        raise OSError('could not decode the input of floating images.')

    # dctout = {}
    lstout = []

    motion_rot = False
    motion_trn = False

    for i in range(len(flolst)):
        M = coreg_spm(ref, flolst[i], fcomment=fcomment, pickname=pickname, fname_aff=fname_aff,
                      matlab_eng_name=matlab_eng_name, outpath=outpath, visual=0, del_uncmpr=True)

        if any((np.abs(M['rotations']) * 180 / np.pi) > rot_thresh):
            log.warning(
                dedent('''\
                at least one rotation is above the threshold of {}, and is:
                {}''').format(rot_thresh, M['rotations'] * 180 / np.pi))
            motion_rot = True
        if any(np.abs(M['translations']) > trn_thresh):
            log.warning(
                dedent('''\
                at least one translation is above the threshold of {}, and is:
                {}''').format(trn_thresh, M['translations']))
            motion_trn = True

        lstout.append({'regout': M, 'trans_mo': motion_trn, 'rotat_mo': motion_rot})

    return lstout


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# I M A G E   S I M I L A R I T Y
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


def dice_coeff(im1, im2, val=1):
    '''
    Calculate Dice score for parcellation images <im1> and <im2> and ROI value <val>.
    Input images can be given as:
        1. paths to NIfTI image files or as
        2. Numpy arrays.
    The ROI value can be given as:
        1. a single integer representing one ROI out of many in the parcellation
           images (<im1> and <im2>) or as
        2. a list of integers to form a composite ROI used for the association test.
    Outputs a float number representing the Dice score.
    '''
    if all(
            isinstance(i, str) and os.path.isfile(i) and hasext(i, ('nii', 'nii.gz'))
            for i in [im1, im2]):
        imn1 = imio.getnii(im1, output='image')
        imn2 = imio.getnii(im2, output='image')
    elif isinstance(im1, (np.ndarray, np.generic)) and isinstance(im1, (np.ndarray, np.generic)):
        imn1 = im1
        imn2 = im2
    else:
        raise TypeError('Unrecognised or Mismatched Images.')

    # a single value corresponding to one ROI
    if isinstance(val, int):
        imv1 = (imn1 == val)
        imv2 = (imn2 == val)
    # multiple values in list corresponding to a composite ROI
    elif isinstance(val, list) and all(isinstance(v, int) for v in val):
        imv1 = (imn1 == val[0])
        imv2 = (imn2 == val[0])
        for v in val[1:]:
            # boolean addition to form a composite ROI
            imv1 += (imn1 == v)
            imv2 += (imn2 == v)
    else:
        raise TypeError('ROI Values have to be integer (single or in a list).')

    if imv1.shape != imv2.shape:
        raise ValueError('Shape Mismatch: Input images must have the same shape.')

    # compute Dice coefficient
    intrsctn = np.logical_and(imv1, imv2)

    return 2. * intrsctn.sum() / (imv1.sum() + imv2.sum())


def dice_coeff_multiclass(im1, im2, roi2ind):
    '''
    Calculate Dice score for parcellation images <im1> and <im2> and ROI value <val>.
    Input images can be given as:
        1. paths to NIfTI image files or as
        2. Numpy arrays.
    The ROI value must be given as a dictionary of lists of indexes for each ROI
    Outputs a float number representing the Dice score.
    '''
    if all(
            isinstance(i, str) and os.path.isfile(i) and hasext(i, ('nii', 'nii.gz'))
            for i in [im1, im2]):
        imn1 = imio.getnii(im1, output='image')
        imn2 = imio.getnii(im2, output='image')
    elif isinstance(im1, (np.ndarray, np.generic)) and isinstance(im1, (np.ndarray, np.generic)):
        imn1 = im1
        imn2 = im2
    else:
        raise TypeError('Unrecognised or Mismatched Images.')

    if imn1.shape != imn2.shape:
        raise ValueError('Shape Mismatch: Input images must have the same shape.')

    out = {}
    for k in roi2ind:

        # multiple values in list corresponding to a composite ROI
        imv1 = (imn1 == roi2ind[k][0])
        imv2 = (imn2 == roi2ind[k][0])
        for v in roi2ind[k][1:]:
            # boolean addition to form a composite ROI
            imv1 += (imn1 == v)
            imv2 += (imn2 == v)

        # compute Dice coefficient
        intrsctn = np.logical_and(imv1, imv2)
        out[k] = 2. * intrsctn.sum() / (imv1.sum() + imv2.sum())

    return out
