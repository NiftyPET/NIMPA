"""ACR/Jaszczak PET phantom I/O and auxiliary functions"""
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2021-23"

import os
from itertools import chain
from pathlib import Path, PurePath
from subprocess import run

import dcm2niix
from miutil.fdio import hasext

from ..prc import imio, prc


def preproc(indat, Cntd, smooth=True, reftrim='', outpath=None, mode='nac'):
    """Convert to NIfTI (if DICOM), smooth using the Gaussian and trim/scale up."""
    opth = Path(indat).parent if outpath is None else Path(outpath)

    if mode == 'nac':
        outdir = opth / mode.upper()
    elif mode[:3] == 'qnt':
        outdir = opth / mode.upper()
    else:
        raise ValueError('unrecognised mode')

    imio.create_dir(outdir)

    if isinstance(indat, (str, PurePath)) and Path(indat).is_dir():
        # CONVERT TO NIfTI
        if not imio.dcmdir(indat):
            raise IOError('the provided folder does not contain DICOM files')
        for f in chain(outdir.glob('*.nii*'), outdir.glob('*.json')):
            # remove previous files
            os.remove(f)
        run([dcm2niix.bin, '-i', 'y', '-v', 'n', '-o', outdir, '-f', '%f_%s', str(indat)])
        fnii = list(outdir.glob('*offline3D*.nii*'))
        if len(fnii) == 1:
            fnii = fnii[0]
        else:
            raise ValueError('Confusing or missing NIfTI output')
    elif isinstance(indat, (str, PurePath)) and Path(indat).is_file() and hasext(
            indat, ('nii', 'nii.gz')):
        fnii = Path(indat)
    else:
        raise ValueError('the input NIfTI file or DICOM folder do not exist')

    # > Gaussian smooth image data if needed
    if smooth:
        if Cntd['fwhm_' + mode[:3]] > 0:
            smostr = '_smo-' + str(Cntd['fwhm_' + mode]).replace('.', '-') + 'mm'
            fnii = prc.imsmooth(
                fnii, fwhm=Cntd['fwhm_' + mode[:3]], fout=outdir /
                (mode.upper() + '_' + fnii.name.split('.nii')[0] + smostr + '.nii.gz'),
                output='file')

    Cntd['f' + mode] = fnii

    # > trim and upsample the PET
    imup = prc.imtrimup(
        fnii,
        refim=reftrim,
        scale=Cntd['sclt'],
        int_order=Cntd['interp'],
        fmax=0.1,                                       # controls how much trimming there is
        fcomment_pfx=fnii.name.split('.nii')[0] + '__',
        store_img=True)

    Cntd[f'f{mode}up'] = Path(imup['fim'])

    return Cntd
