"""image input/output functionalities."""
import datetime
import logging
import numbers
import os
import pathlib
import re
import shutil
from pathlib import Path
from subprocess import run
from textwrap import dedent

import nibabel as nib
import numpy as np
import pydicom as dcm
from miutil.imio.nii import array2nii  # NOQA: F401 # yapf: disable
from miutil.imio.nii import getnii  # NOQA: F401 # yapf: disable
from miutil.imio.nii import nii_gzip  # NOQA: F401 # yapf: disable
from miutil.imio.nii import nii_ugzip  # NOQA: F401 # yapf: disable
from miutil.imio.nii import niisort  # NOQA: F401 # yapf: disable

# > NiftyPET resources
from .. import resources as rs

log = logging.getLogger(__name__)

# > possible extensions for DICOM files
dcmext = ('dcm', 'DCM', 'ima', 'IMA', 'img', 'IMG')

# > remove characters unwanted in file/folder names
avoid_chars = '{}[]!@#$%^&*.()+=:;~ '

# > DICOM coding of PET isotopes
istp_code = {
    'C-111A1': 'F18', 'C-105A1': 'C11', 'C-B1038': 'O15', 'C-128A2': 'Ge68', 'C-131A3': 'Ga68'}


def create_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)


def time_stamp(simple_ascii=False):
    now = datetime.datetime.now()
    if simple_ascii:
        nowstr = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '_' + str(
            now.hour) + 'h' + str(now.minute)
    else:
        nowstr = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + ' ' + str(
            now.hour) + ':' + str(now.minute)
    return nowstr


def fwhm2sig(fwhm, voxsize=2.0):
    return (fwhm/voxsize) / (2 * (2 * np.log(2))**.5)


def rem_chars(txt, replacement_char='_'):
    ''' remove disallowed or inconvenient characters
        (as def. in `avoid_chars`) from file/folder names.
    '''
    for c in avoid_chars:
        txt = txt.replace(c, '_')

    while txt[0] == replacement_char:
        txt = txt[1:]

    return txt


def mgh2nii(fim, fout=None, output=None):
    ''' Convert `*.mgh` or `*.mgz` FreeSurfer image to NIfTI.
        Arguments:
            fim: path to the input MGH file
            fout: path to the output NIfTI file, if None then
                  creates based on `fim`
            output: if not None and an applicable string it will
            output a dictionary or an array (see below)
        Return:
            None: returns nothing
            'image' or 'im': outputs just the image
            'affine': outputs just the affine matrix
            'all': outputs all as a dictionary
    '''

    if not os.path.isfile(fim):
        raise ValueError('The input path is incorrect!')

    # > get the image dictionary
    mghd = getmgh(fim, output='all')
    im = mghd['im']

    # > sort out the output
    if fout is None:
        fout = fim.parent / (fim.name.split('.')[0] + '.nii.gz')

    out = fout
    if output == 'image' or output == 'im':
        out = fout, im
    elif output == 'affine':
        out = fout, mghd['affine']
    elif output == 'all':
        out = mghd
        out['fout'] = fout

    array2nii(
        mghd['im'], mghd['affine'], fout,
        trnsp=(mghd['transpose'].index(0), mghd['transpose'].index(1), mghd['transpose'].index(2)),
        flip=mghd['flip'])

    return out


def getmgh(fim, nan_replace=None, output='image'):
    '''
    Get image from `*.mgz` or `*.mgh` file (FreeSurfer).
    Arguments:
        fim: input file name for the MGH/Z image
        output: option for choosing output: 'image', 'affine' matrix or
                'all' for a dictionary with all the info.
    Return:
        'image': outputs just the image
        'affine': outputs just the affine matrix
        'all': outputs all as a dictionary
    '''

    if not os.path.isfile(fim):
        raise ValueError('The input path is incorrect!')

    mgh = nib.freesurfer.load(str(fim))

    if output == 'image' or output == 'all':

        imr = np.asanyarray(mgh.dataobj)
        # replace NaNs if requested
        if isinstance(nan_replace, numbers.Number):
            imr[np.isnan(imr)] = nan_replace

        imr = np.squeeze(imr)
        dimno = imr.ndim

        # > get orientations from the affine
        ornt = nib.io_orientation(mgh.affine)
        trnsp = tuple(np.flip(np.argsort(ornt[:, 0])))
        flip = tuple(np.int8(ornt[:, 1]))

        # > flip y-axis and z-axis and then transpose
        if dimno == 4:   # dynamic
            imr = np.transpose(imr[::-flip[0], ::-flip[1], ::-flip[2], :], (3,) + trnsp)
        elif dimno == 3: # static
            imr = np.transpose(imr[::-flip[0], ::-flip[1], ::-flip[2]], trnsp)

        # # > voxel size
        # voxsize = mgh.header.get('pixdim')[1:mgh.header.get('dim')[0] + 1]
        # # > rearrange voxel size according to the orientation
        # voxsize = voxsize[np.array(trnsp)]

    if output == 'all':
        out = {
            'im': imr, 'affine': mgh.affine, 'fim': fim, 'dtype': mgh.get_data_dtype(),
            'shape': imr.shape, 'hdr': mgh.header, 'transpose': trnsp, 'flip': flip}
    elif output == 'image':
        out = imr
    elif output == 'affine':
        out = mgh.affine
    else:
        raise NameError("Unrecognised output request!")

    return out


def getnii_descr(fim):
    '''Extracts the custom description header field to dictionary'''
    nim = nib.load(fim)
    hdr = nim.header
    rcnlst = hdr['descrip'].item().split(';')
    rcndic = {}

    if rcnlst[0] == '':
        return rcndic

    for ci in range(len(rcnlst)):
        tmp = rcnlst[ci].split('=')
        rcndic[tmp[0]] = tmp[1]
    return rcndic


def orientnii(imfile, Cnt=None):
    '''Get the orientation from NIfTI sform.  Not fully functional yet.'''

    # > check if the dictionary of constant is given
    if Cnt is None:
        Cnt = {}

    strorient = ['L-R', 'S-I', 'A-P']
    niiorient = []
    niixyz = np.zeros(3, dtype=np.int8)
    if os.path.isfile(imfile):
        nim = nib.load(imfile)
        A = nim.get_sform()
        for i in range(3):
            niixyz[i] = np.argmax(abs(A[i, :-1]))
            niiorient.append(strorient[niixyz[i]])
        log.info('NIfTI orientation:\n{}'.format(niiorient))

    return niiorient


def pick_t1w(mri):
    '''Pick the MR T1w from the dictionary for MR->PET registration.'''
    if not isinstance(mri, dict):
        raise IOError('incorrect input given for the T1w image')

    # check if NIfTI file is given
    if 'T1N4' in mri and os.path.isfile(mri['T1N4']):
        return mri['T1N4']
    # or another bias corrected
    elif 'T1bc' in mri and os.path.isfile(mri['T1bc']):
        return mri['T1bc']
    elif 'T1nii' in mri and os.path.isfile(mri['T1nii']):
        return mri['T1nii']
    elif 'T1DCM' in mri and os.path.exists(mri['MRT1W']):
        return dcm2nii(mri['T1nii'], 'converted')
    raise IOError('could not find a T1w image!')


# ======================================================================
def dcminfo(dcmvar, Cnt=None, output='class', t1_name='mprage'):
    """
    Get DICOM info from file/header.
    Args:
      dcmvar: DICOM header as a file/string/dictionary
      Cnt(dict): constants used in advanced reconstruction or analysis
      output(str): 'class' outputs all with classification;
        'basic' outputs scanner ID and series/protocol string;
        'detail' gives most relevant tags in dictionary
      t1_name(str): helps identify T1w MR image present in series or file names
    """
    if Cnt is None:
        Cnt = {}

    if isinstance(dcmvar, str):
        log.debug('provided DICOM file: {}'.format(dcmvar))
        dhdr = dcm.dcmread(dcmvar)
    elif isinstance(dcmvar, dict):
        dhdr = dcmvar
    elif isinstance(dcmvar, dcm.dataset.FileDataset):
        dhdr = dcmvar
    elif isinstance(dcmvar, (pathlib.Path, pathlib.PurePath)):
        dhdr = dcm.dcmread(dcmvar)

    dtype = dhdr[0x08, 0x08].value
    log.debug('   Image Type: {}'.format(dtype))

    # > output dictionary
    outdct = {}

    # ------------------------------------------
    # > scanner ID
    scanner_vendor = 'unknown'
    if [0x008, 0x070] in dhdr:
        scanner_vendor = dhdr[0x008, 0x070].value
        outdct['scnr_vndr'] = scanner_vendor

    scanner_model = 'unknown'
    if [0x008, 0x1090] in dhdr:
        scanner_model = dhdr[0x008, 0x1090].value
        outdct['scnr_mdl'] = scanner_model

    scanner_id = 'other'
    if any(s in scanner_model
           for s in ['mMR', 'Biograph']) and 'siemens' in scanner_vendor.lower():
        scanner_id = 'mmr'
    elif 'signa' in scanner_model.lower() and 'ge' in scanner_vendor.lower():
        scanner_id = 'signa'

    outdct['scnr_id'] = scanner_id
    # ------------------------------------------

    # ------------------------------------------
    # > date/time
    study_time = None
    if [0x008, 0x030] in dhdr and [0x008, 0x020] in dhdr:
        val = dhdr[0x008, 0x020].value + dhdr[0x008, 0x030].value
        if '.' in val:
            study_time = datetime.datetime.strptime(val, '%Y%m%d%H%M%S.%f')
        else:
            study_time = datetime.datetime.strptime(val, '%Y%m%d%H%M%S')

        outdct['st_time'] = study_time

    series_time = None
    if [0x008, 0x031] in dhdr and [0x008, 0x021] in dhdr:
        val = dhdr[0x008, 0x021].value + dhdr[0x008, 0x031].value
        if '.' in val:
            series_time = datetime.datetime.strptime(val, '%Y%m%d%H%M%S.%f')
        else:
            series_time = datetime.datetime.strptime(val, '%Y%m%d%H%M%S')

        outdct['sr_time'] = series_time

    acq_time = None
    if [0x008, 0x032] in dhdr and [0x008, 0x022] in dhdr:
        val = dhdr[0x008, 0x022].value + dhdr[0x008, 0x032].value
        if '.' in val:
            acq_time = datetime.datetime.strptime(val, '%Y%m%d%H%M%S.%f')
        else:
            acq_time = datetime.datetime.strptime(val, '%Y%m%d%H%M%S')

        outdct['aq_time'] = acq_time

    # > frame duration time
    if [0x018, 0x1242] in dhdr:
        val = int(dhdr[0x018, 0x1242].value)
        outdct['frm_dur'] = datetime.timedelta(milliseconds=val)

    # ------------------------------------------

    # > institution
    inst = ''
    if [0x008, 0x080] in dhdr:
        inst = dhdr[0x008, 0x080].value
        outdct['inst_name'] = inst

    prtcl = ''
    if [0x18, 0x1030] in dhdr:
        prtcl = dhdr[0x18, 0x1030].value
        outdct['pr_name'] = prtcl

    srs = ''
    if [0x08, 0x103e] in dhdr:
        srs = dhdr[0x08, 0x103e].value
        outdct['sr_name'] = srs

    # > units
    unt = None
    if [0x054, 0x1001] in dhdr:
        unt = dhdr[0x054, 0x1001].value
        outdct['units'] = unt

    # > sequence name
    if [0x018, 0x024] in dhdr:
        outdct['sq_name'] = dhdr[0x018, 0x024].value

    # ------- PATIENT -------
    # > patient size
    if [0x010, 0x1020] in dhdr:
        outdct['pat_size'] = dhdr[0x010, 0x1020].value

    # > patient sex
    if [0x010, 0x040] in dhdr:
        outdct['pat_sex'] = dhdr[0x010, 0x040].value

    # > patient DoB
    if [0x010, 0x030] in dhdr:
        outdct['pat_dob'] = dhdr[0x010, 0x030].value

    # > patient weight
    if [0x010, 0x1030] in dhdr:
        outdct['pat_weight'] = dhdr[0x010, 0x1030].value

    # > patient age
    if [0x010, 0x1010] in dhdr:
        outdct['pat_age'] = dhdr[0x010, 0x1010].value
    # -----------------------

    # +++++++++++++++++++++++++++++++++++++++++++++
    if output == 'basic':
        out = [prtcl, srs, scanner_id]
        return out
    # +++++++++++++++++++++++++++++++++++++++++++++

    # =================================================================
    # PET
    # -----------------------------------------------------------------
    if [0x054, 0x1000] in dhdr or [0x054, 0x000] in dhdr:
        outdct['PET'] = {}

    # > PET parameters
    srs_type = None
    if [0x054, 0x1000] in dhdr:
        srs_type = dhdr[0x054, 0x1000].value[0]
        outdct['PET']['sr_type'] = srs_type

    recon = None
    if [0x054, 0x1103] in dhdr:
        recon = dhdr[0x054, 0x1103].value
        outdct['PET']['recon'] = recon

    decay_corr = None
    if [0x054, 0x1102] in dhdr:
        decay_corr = dhdr[0x054, 0x1102].value
        outdct['PET']['decay_crr'] = decay_corr

    # > decay factor
    dcf = None
    if [0x054, 0x1321] in dhdr:
        dcf = float(dhdr[0x054, 0x1321].value)
        outdct['PET']['decay_fcr'] = dcf

    atten = None
    if [0x054, 0x1101] in dhdr:
        atten = dhdr[0x054, 0x1101].value
        outdct['PET']['ac_mth'] = atten

    scat = None
    if [0x054, 0x1105] in dhdr:
        scat = dhdr[0x054, 0x1105].value
        outdct['PET']['sc_mth'] = scat

    # > scatter factor
    scf = None
    if [0x054, 0x1323] in dhdr:
        scf = float(dhdr[0x054, 0x1323].value)
        outdct['PET']['sc_fcr'] = scf

    # > randoms correction method
    rand = None
    if [0x054, 0x1100] in dhdr:
        rand = dhdr[0x054, 0x1100].value
        outdct['PET']['rc_mth'] = rand

    # > dose calibration factor
    dscf = None
    if [0x054, 0x1322] in dhdr:
        dscf = float(dhdr[0x054, 0x1322].value)
        outdct['PET']['dose_calib_fcr'] = dscf

    # > dead time factor
    dt = None
    if [0x054, 0x1324] in dhdr:
        dt = float(dhdr[0x054, 0x1324].value)
        outdct['PET']['dt_fcr'] = dt

    # -----------------------------------------------------------------
    # RADIO TRACER
    tracer = None
    tdose = None
    hlife = None
    pfract = None
    ttime0 = None
    ttime1 = None

    if [0x054, 0x016] in dhdr:
        # > all tracer info
        tinf = dhdr[0x054, 0x016][0]

        if [0x018, 0x031] in tinf:
            tracer = tinf[0x018, 0x031].value
            outdct['PET']['tracer'] = tracer

        if [0x018, 0x1074] in tinf:
            tdose = float(tinf[0x018, 0x1074].value)
            outdct['PET']['dose_tot'] = tdose

        if [0x018, 0x1075] in tinf:
            hlife = float(tinf[0x018, 0x1075].value)
            outdct['PET']['hlife'] = hlife

        if [0x018, 0x1076] in tinf:
            pfract = float(tinf[0x018, 0x1076].value)
            outdct['PET']['positron_frc'] = pfract

        if [0x018, 0x1078] in tinf:
            val = tinf[0x018, 0x1078].value
            if '.' in val:
                ttime0 = datetime.datetime.strptime(val, '%Y%m%d%H%M%S.%f')
            else:
                ttime0 = datetime.datetime.strptime(val, '%Y%m%d%H%M%S')

            outdct['PET']['radio_start_time'] = ttime0

        if [0x018, 0x1079] in tinf:
            val = tinf[0x018, 0x1079].value
            if '.' in val:
                ttime1 = datetime.datetime.strptime(val, '%Y%m%d%H%M%S.%f')
            else:
                ttime1 = datetime.datetime.strptime(val, '%Y%m%d%H%M%S')

            outdct['PET']['radio_stop_time'] = ttime1
    # -----------------------------------------------------------------

    # ---------------------------------------------
    # > MR parameters (echo time, etc)
    TR = None
    TE = None
    if [0x18, 0x80] in dhdr:
        TR = float(dhdr[0x18, 0x80].value)
        log.debug('   TR: {}'.format(TR))
    if [0x18, 0x81] in dhdr:
        TE = float(dhdr[0x18, 0x81].value)
        log.debug('   TE: {}'.format(TE))

    validTs = TR is not None and TE is not None

    if validTs:
        outdct['MR'] = {'TR': TR, 'TE': TE}
        """
        mrdct = {
            'series': srs, 'protocol': prtcl, 'units': unt, 'study_time': study_time, 'inst': inst,
            'series_time': series_time, 'acq_time': acq_time, 'scanner_id': scanner_id, 'TR': TR,
            'TE': TE}
        """
        mrdct = outdct
    # ---------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++
    if output == 'detail':
        return outdct
    # +++++++++++++++++++++++++++++++++++++++++++++

    isPET = (tracer is not None) and (srs_type in ['STATIC', 'DYNAMIC', 'WHOLE BODY', 'GATED'])

    # > CSA type (mMR)
    csatype = ''
    if [0x29, 0x1108] in dhdr:
        csatype = dhdr[0x29, 0x1108].value
        log.debug('   CSA Data Type: {}'.format(csatype))

    # > DICOM comment or on MR parameters
    cmmnt = ''
    if [0x20, 0x4000] in dhdr:
        cmmnt = dhdr[0x0020, 0x4000].value
        log.debug('   Comments: {}'.format(cmmnt))

    # > check for RAW data
    if any('PET_NORM' in s
           for s in dtype) or cmmnt == 'PET Normalization data' or csatype == 'MRPETNORM':
        out = ['raw', 'norm', scanner_id]

    elif any('PET_LISTMODE' in s
             for s in dtype) or cmmnt == 'Listmode' or csatype == 'MRPETLM_LARGE':
        out = ['raw', 'list', scanner_id]

    elif any('PET_EM_SINO' in s for s in dtype) or cmmnt == 'Sinogram' or csatype == 'MRPETSINO':
        out = ['raw', 'sinogram', scanner_id]

    # > physio data
    elif 'PET_PHYSIO' in dtype or 'physio' in cmmnt.lower():
        out = ['raw', 'physio', scanner_id]

    elif any('MRPET_UMAP3D' in s for s in dtype) or cmmnt == 'MR based umap':
        out = ['mr', 'mumap', 'ute', 'mr', scanner_id]

    elif isPET:
        """
        petdct = {
            'series': srs, 'protocol': prtcl, 'study_time': study_time, 'series_time': series_time,
            'inst': inst, 'acq_time': acq_time, 'scanner_id': scanner_id, 'type': srs_type,
            'units': unt, 'recon': recon, 'decay_corr': decay_corr, 'dcf': dcf,
            'attenuation': atten, 'scatter': scat, 'scf': scf, 'randoms': rand, 'dose_calib': dscf,
            'dead_time': dt, 'tracer': tracer, 'total_dose': tdose, 'half_life': hlife,
            'positron_fract': pfract, 'radio_start_time': ttime0, 'radio_stop_time': ttime1}
        """
        petdct = outdct

        out = ['pet', tracer.lower(), srs_type.lower(), scanner_id, petdct]

    # > a less stringent way of looking for the T1w sequence
    # > than the one below
    elif validTs and (t1_name in prtcl.lower() or t1_name in srs.lower()):
        out = ['mr', 't1', 'mprage', scanner_id, mrdct]

    elif validTs and TR > 400 and TR < 2500 and TE < 20:
        if t1_name in prtcl.lower() or t1_name in srs.lower():
            out = ['mr', 't1', 'mprage', scanner_id, mrdct]
        else:
            out = ['mr', 't1', scanner_id]

    elif validTs and TR > 2500 and TE > 50:
        out = ['mr', 't2', scanner_id, mrdct]

    # > UTE's two sequences:
    # > UTE2
    elif validTs and TR < 50 and TE < 20 and TE > 1:
        out = ['mr', 'ute', 'ute2', scanner_id, mrdct]

    # > UTE1
    elif validTs and TR < 50 and TE < 0.1 and TR > 0 and TE > 0:
        out = ['mr', 'ute', 'ute1', scanner_id, mrdct]

    else:
        out = ['unknown', str(cmmnt.lower())]

    return out


# ======================================================================


def dcmsort(folder, copy_series=False, Cnt=None, outpath=None, grouping='t+d'):
    '''
        sort out the DICOM files in the folder according to the recorded series.
        options:
        - copy_series: if True it makes folders for each series and copies DICOM
                       files into each corresponding folder.
        - Cnt: dictionary of constants (currently not used)
        - outpath: the output path where the series folders are created
        - grouping: defines how series are recognised, i.e., either by time plus
                    series description ('t+d') or by the description only ('d'),
                    or by acquisition and series times plus series description
                    ('a+t+d'), or using series instance unique id ('d+suid').
    '''

    # > insure that `folder` is Path object
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError('Incorrect input folder!')

    # > check if the dictionary of constant is given
    if Cnt is None:
        Cnt = {}

    srs = {}

    for f in folder.iterdir():

        # ---------------------------------
        if not f.is_file(): continue
        try:
            dhdr = dcm.dcmread(f)
        except Exception:
            continue
        # ---------------------------------

        # --------------------------------
        # > image size
        imsz = np.zeros(2, dtype=np.int64)
        if [0x028, 0x010] in dhdr:
            imsz[0] = dhdr[0x028, 0x010].value
        if [0x028, 0x011] in dhdr:
            imsz[1] = dhdr[0x028, 0x011].value

        # > voxel size
        vxsz = np.zeros(3, dtype=np.float64)
        if [0x028, 0x030] in dhdr and [0x018, 0x050] in dhdr:
            pxsz = np.array([float(e) for e in dhdr[0x028, 0x030].value])
            vxsz[:2] = pxsz
            vxsz[2] = float(dhdr[0x018, 0x050].value)

        # > orientation
        ornt = np.zeros(6, dtype=np.float64)
        if [0x020, 0x037] in dhdr:
            ornt = np.array([float(e) for e in dhdr[0x20, 0x37].value])

        # > series description, time and study time
        srs_dcrp = ''
        if [0x0008, 0x103e] in dhdr:
            srs_dcrp = dhdr[0x0008, 0x103e].value

        srs_uid = ''
        if [0x020, 0x00e] in dhdr:
            srs_uid = dhdr[0x020, 0x00e].value

        prtcl = ''
        # > protocol
        if [0x018, 0x1030] in dhdr:
            prtcl = dhdr[0x018, 0x1030].value

        # > series, study and acquisition times
        srs_time = dhdr[0x0008, 0x0031].value[:6]
        std_time = dhdr[0x0008, 0x0030].value[:6]
        acq_time = ''
        if [0x0008, 0x0032] in dhdr:
            acq_time = dhdr[0x0008, 0x0032].value[:6]

        # > study date
        std_date = dhdr[0x008, 0x020].value

        frm_dur = None
        # > frame duration time (for PET)
        if [0x018, 0x1242] in dhdr:
            val = np.round(int(dhdr[0x018, 0x1242].value) / 1e3, decimals=0)
            frm_dur = datetime.timedelta(seconds=val)

        # > time of tracer administration (start)
        tinjct = None
        # > PET tracer if present
        trcr = None
        if [0x054, 0x016] in dhdr:
            trinf = dhdr[0x054, 0x016][0]
            if [0x018, 0x1078] in trinf:
                val = trinf[0x018, 0x1078].value
                if '.' in val:
                    tinjct = datetime.datetime.strptime(val, '%Y%m%d%H%M%S.%f')
                else:
                    tinjct = datetime.datetime.strptime(val, '%Y%m%d%H%M%S')

            if [0x018, 0x031] in trinf:
                trcr = trinf[0x018, 0x031].value
        # --------------------------------

        log.debug(
            dedent('''\
            --------------------------------------
            DICOM series desciption: {}
            DICOM series time: {}
            DICOM study  time: {}
            --------------------------------------''').format(srs_dcrp, srs_time, std_time))

        # ---------
        # series for any category (can be multiple scans within the same category)
        recognised_series = False
        srs_k = list(srs.keys())
        for s in srs_k:

            if grouping == 't+d':
                if (np.array_equal(srs[s]['imorient'], ornt)
                        and np.array_equal(srs[s]['imsize'], imsz)
                        and np.array_equal(srs[s]['voxsize'], vxsz)
                        and srs[s]['tseries'] == srs_time and srs[s]['series'] == srs_dcrp):
                    recognised_series = True
                    break

            elif grouping == 'a+t+d':
                if (np.array_equal(srs[s]['imorient'], ornt)
                        and np.array_equal(srs[s]['imsize'], imsz)
                        and np.array_equal(srs[s]['voxsize'], vxsz)
                        and srs[s]['tseries'] == srs_time and srs[s]['tacq'] == acq_time
                        and srs[s]['series'] == srs_dcrp):
                    recognised_series = True
                    break

            elif grouping == 'd':
                if (np.array_equal(srs[s]['imorient'], ornt)
                        and np.array_equal(srs[s]['imsize'], imsz)
                        and np.array_equal(srs[s]['voxsize'], vxsz)
                        and srs[s]['series'] == srs_dcrp):
                    recognised_series = True
                    break

            elif grouping == 'd+suid':
                if (np.array_equal(srs[s]['imorient'], ornt)
                        and np.array_equal(srs[s]['imsize'], imsz)
                        and np.array_equal(srs[s]['voxsize'], vxsz)
                        and srs[s]['series'] == srs_dcrp and srs[s]['series_uid'] == srs_uid):
                    recognised_series = True
                    break

            else:
                raise ValueError('Unrecognised grouping option')

        # >  if series was not found, create one
        if not recognised_series:
            if grouping == 't+d':
                s = srs_time + '_' + srs_dcrp
            elif grouping == 'a+t+d':
                s = acq_time + '_' + srs_time + '_' + srs_dcrp
            elif grouping == 'd':
                s = srs_dcrp
            elif grouping == 'd+suid':
                s = srs_dcrp + '_' + srs_uid[-10:]
            else:
                raise ValueError('Unrecognised grouping option')

            srs[s] = {}
            srs[s]['imorient'] = ornt
            srs[s]['imsize'] = imsz
            srs[s]['voxsize'] = vxsz
            srs[s]['tacq'] = acq_time
            srs[s]['tseries'] = srs_time
            srs[s]['tstudy'] = std_time
            srs[s]['dstudy'] = std_date
            srs[s]['series'] = srs_dcrp
            srs[s]['series_uid'] = srs_uid
            srs[s]['protocol'] = prtcl

            if tinjct is not None:
                srs[s]['radio_start_time'] = tinjct
            if frm_dur is not None:
                srs[s]['frm_dur'] = frm_dur
            if trcr is not None:
                srs[s]['tracer'] = trcr

        # append the file name
        srs[s].setdefault('files', [])

        if copy_series:
            out = folder
            if outpath is not None:
                try:
                    create_dir(outpath)
                except Exception as e:
                    log.warning(
                        f"could not create specified output folder, using input folder.\n\n{e}")
                else:
                    out = Path(outpath)

            srsdir = out / rem_chars(s)
            create_dir(srsdir)
            shutil.copy(f, srsdir)
            srs[s]['files'].append(srsdir / f.name)
        else:
            srs[s]['files'].append(f)

    return srs


# ======================================================================


def list_dcm_datain(datain):
    '''List all DICOM file paths in the datain dictionary of input data.'''

    if not isinstance(datain, dict):
        raise ValueError('The input is not a dictionary!')

    dcmlst = []
    # list of mu-map DICOM files
    if 'mumapDCM' in datain:
        dcmump = os.listdir(datain['mumapDCM'])
        # accept only *.dcm extensions
        dcmump = [os.path.join(datain['mumapDCM'], d) for d in dcmump if d.endswith(dcmext)]
        dcmlst += dcmump

    if 'T1DCM' in datain:
        dcmt1 = os.listdir(datain['T1DCM'])
        # accept only *.dcm extensions
        dcmt1 = [os.path.join(datain['T1DCM'], d) for d in dcmt1 if d.endswith(dcmext)]
        dcmlst += dcmt1

    if 'T2DCM' in datain:
        dcmt2 = os.listdir(datain['T2DCM'])
        # accept only *.dcm extensions
        dcmt2 = [os.path.join(datain['T2DCM'], d) for d in dcmt2 if d.endswith(dcmext)]
        dcmlst += dcmt2

    if 'UTE1' in datain:
        dcmute1 = os.listdir(datain['UTE1'])
        # accept only *.dcm extensions
        dcmute1 = [os.path.join(datain['UTE1'], d) for d in dcmute1 if d.endswith(dcmext)]
        dcmlst += dcmute1

    if 'UTE2' in datain:
        dcmute2 = os.listdir(datain['UTE2'])
        # accept only *.dcm extensions
        dcmute2 = [os.path.join(datain['UTE2'], d) for d in dcmute2 if d.endswith(dcmext)]
        dcmlst += dcmute2

    # list-mode data dcm
    if 'lm_dcm' in datain:
        dcmlst += [datain['lm_dcm']]

    if 'lm_ima' in datain:
        dcmlst += [datain['lm_ima']]

    # norm
    if 'nrm_dcm' in datain:
        dcmlst += [datain['nrm_dcm']]

    if 'nrm_ima' in datain:
        dcmlst += [datain['nrm_ima']]

    return dcmlst


def dcmanonym(dcmpth, displayonly=False, patient='anonymised', physician='anonymised',
              dob='19800101', Cnt=None):
    '''
    Anonymise DICOM file(s)
    Arguments:
        > dcmpth:   it can be passed as a single DICOM file, or
                    a folder containing DICOM files, or a list of DICOM file paths.
        > patient:  the name of the patient.
        > physician:the name of the referring physician.
        > dob:      patient's date of birth.
        > Cnt:      dictionary of constants (containing logging variable)
    '''
    # > check if the dictionary of constant is given
    if Cnt is None:
        Cnt = {}

    # > check if a single DICOM file
    if isinstance(dcmpth, str) and os.path.isfile(dcmpth):
        dcmlst = [dcmpth]
        log.debug('recognised the input argument as a single DICOM file.')

    # > check if a folder containing DICOM files
    elif isinstance(dcmpth, str) and os.path.isdir(dcmpth):
        dircontent = os.listdir(dcmpth)
        # > create a list of DICOM files inside the folder
        dcmlst = [
            os.path.join(dcmpth, d) for d in dircontent
            if os.path.isfile(os.path.join(dcmpth, d)) and d.endswith(dcmext)]
        log.debug('recognised the input argument as the folder containing DICOM files.')

    # > check if a folder containing DICOM files
    elif isinstance(dcmpth, list):
        if not all(os.path.isfile(d) and d.endswith(dcmext) for d in dcmpth):
            raise IOError('Not all files in the list are DICOM files.')
        dcmlst = dcmpth
        log.debug('recognised the input argument as the list of DICOM file paths.')

    # > check if dictionary of data input <datain>
    elif isinstance(dcmpth, dict) and 'corepath' in dcmpth:
        dcmlst = list_dcm_datain(dcmpth)
        log.debug('recognised the input argument as the dictionary of scanner data.')

    else:
        raise IOError('Unrecognised input!')

    for dcmf in dcmlst:
        # > read the file
        dhdr = dcm.dcmread(dcmf)

        # > get the basic info about the DICOM file
        dcmtype = dcminfo(dhdr)
        log.debug(
            dedent('''\
            --------------------------------------------------
            DICOM file is for: {}
            --------------------------------------------------''').format(dcmtype))

        # > anonymise mMR data.
        if 'mmr' in dcmtype:

            if [0x029, 0x1120] in dhdr and dhdr[0x029, 0x1120].name == '[CSA Series Header Info]':
                csafield = dhdr[0x029, 0x1120]
                csa = csafield.value
            elif [0x029, 0x1020] in dhdr and dhdr[0x029,
                                                  0x1020].name == '[CSA Series Header Info]':
                csafield = dhdr[0x029, 0x1020]
                csa = csafield.value
            else:
                csa = ''

            # string length considered for replacement
            strlen = 200

            idx = [m.start() for m in re.finditer(r'([Pp]atients{0,1}[Nn]ame)', csa)]
            if idx:
                log.debug(
                    'DICOM> found sensitive information deep in the headers: {}'.format(dcmtype))

            # > run the anonymisation
            iupdate = 0

            for i in idx:
                ci = i - iupdate

                if displayonly:
                    log.debug(
                        dedent('''\
                        DICOM> sensitive info:
                             {}''').format(csa[ci:ci + strlen]))
                    continue

                rplcmnt = re.sub(r'(\{\s*\"{1,2}\W*\w+\W*\w+\W*\"{1,2}\s*\})',
                                 '{ ""' + patient + '"" }', csa[ci:ci + strlen])
                # > update string
                csa = csa[:ci] + rplcmnt + csa[ci + strlen:]
                log.debug('DICOM> removed sensitive information.')
                # > correct for the number of removed letters
                iupdate = strlen - len(rplcmnt)

            # > update DICOM
            if not displayonly and csa != '':
                csafield.value = csa

        # > Patient's name
        if [0x010, 0x010] in dhdr:
            if displayonly:
                log.debug(
                    dedent('''\
                    DICOM> sensitive info: {}
                         > {}''').format(dhdr[0x010, 0x010].name, dhdr[0x010, 0x010].value))
            else:
                dhdr[0x010, 0x010].value = patient
                log.debug('DICOM> anonymised patients name')

        # > date of birth
        if [0x010, 0x030] in dhdr:
            if displayonly:
                log.debug(
                    dedent('''\
                DICOM> sensitive info: {}
                     > {}''').format(dhdr[0x010, 0x030].name, dhdr[0x010, 0x030].value))
            else:
                dhdr[0x010, 0x030].value = dob
                log.debug('   > anonymised date of birth')

        # > physician's name
        if [0x008, 0x090] in dhdr:
            if displayonly:
                log.debug(
                    dedent('''\
                DICOM> sensitive info: {}
                     > {}''').format(dhdr[0x008, 0x090].name, dhdr[0x008, 0x090].value))
            else:
                dhdr[0x008, 0x090].value = physician
                log.debug('   > anonymised physician name')

        dhdr.save_as(dcmf)


def dcm2nii(
    dcmpth,
    fimout='',
    fprefix='converted-from-DICOM_',
    fcomment='',
    outpath='',
    timestamp=True,
    executable=None,
    force=False,
):
    """
    Convert DICOM folder `dcmpth` to NIfTI using `dcm2niix` third-party software.
    Args:
      dcmpth: directory containing DICOM files
    """
    # skip conversion if the output already exists and not force is selected
    if os.path.isfile(fimout) and not force:
        return fimout

    if not executable:
        executable = getattr(rs, 'DCM2NIIX', None)
        if not executable:
            import dcm2niix
            executable = dcm2niix.bin
    if not os.path.isfile(executable):
        raise IOError(f"executable not found:{executable}")

    if not os.path.isdir(dcmpth):
        raise IOError("the provided `dcmpth` is not a folder")

    # output path
    if not outpath and fimout and '/' in fimout:
        opth = os.path.dirname(fimout)
        opth = opth or dcmpth
        fimout = os.path.basename(fimout)
    else:
        opth = outpath or dcmpth

    create_dir(opth)

    if not fimout:
        fimout = fprefix
        if timestamp:
            fimout += time_stamp(simple_ascii=True)
    fimout = fimout.split('.nii')[0]

    run([executable, '-f', fimout, '-o', opth, dcmpth])
    fniiout = list(pathlib.Path(opth).glob(f"*{fimout}*.nii*"))

    if fniiout:
        return str(fniiout[0])
    raise ValueError("could not find output nii file")


def dcm2im(fpth):
    '''
    Get the DICOM files from 'fpth' into an image with the affine transformation.
    fpth can be a list of DICOM files or a path (string) to the folder with DICOM files.
    '''
    # possible DICOM file extensions
    ext = ('dcm', 'DCM', 'ima', 'IMA')

    # case when given a folder path
    if isinstance(fpth, str) and os.path.isdir(fpth):
        SZ0 = len([d for d in os.listdir(fpth) if d.endswith(ext)])
        # list of DICOM files
        fdcms = os.listdir(fpth)
        fdcms = [os.path.join(fpth, f) for f in fdcms if f.endswith(ext)]

    # case when list of DICOM files is given
    elif isinstance(fpth, list) and os.path.isfile(os.path.join(fpth[0])):
        SZ0 = len(fpth)
        # list of DICOM files
        fdcms = fpth
        fdcms = [f for f in fdcms if f.endswith(ext)]
    else:
        raise NameError('Unrecognised input for DICOM files.')

    if SZ0 < 1:
        log.error('No DICOM images in the specified path.')
        raise IOError('Input DICOM images not recognised')

    # pick single DICOM header
    dhdr = dcm.read_file(fdcms[0])

    # -----------------------------------
    # some info, e.g.: patient position and series UID
    if [0x018, 0x5100] in dhdr:
        ornt = dhdr[0x18, 0x5100].value
    else:
        ornt = 'unkonwn'
    # Series UID
    sruid = dhdr[0x0020, 0x000e].value
    # -----------------------------------

    # -----------------------------------
    # INIT
    # image position
    P = np.zeros((SZ0, 3), dtype=np.float64)
    # image orientation
    Orn = np.zeros((SZ0, 6), dtype=np.float64)
    # xy resolution
    R = np.zeros((SZ0, 2), dtype=np.float64)
    # slice thickness
    S = np.zeros((SZ0, 1), dtype=np.float64)
    # slope and intercept
    SI = np.ones((SZ0, 2), dtype=np.float64)
    SI[:, 1] = 0

    # image data as an list of array for now
    IM = []
    # -----------------------------------

    c = 0
    for d in fdcms:
        dhdr = dcm.read_file(d)
        if [0x20, 0x32] in dhdr and [0x20, 0x37] in dhdr and [0x28, 0x30] in dhdr:
            P[c, :] = np.array([float(f) for f in dhdr[0x20, 0x32].value])
            Orn[c, :] = np.array([float(f) for f in dhdr[0x20, 0x37].value])
            R[c, :] = np.array([float(f) for f in dhdr[0x28, 0x30].value])
            S[c, :] = float(dhdr[0x18, 0x50].value)
        else:
            log.error('could not read all the DICOM tags.')
            return {'im': [], 'affine': [], 'shape': [], 'orient': ornt, 'sruid': sruid}

        if [0x28, 0x1053] in dhdr and [0x28, 0x1052] in dhdr:
            SI[c, 0] = float(dhdr[0x28, 0x1053].value)
            SI[c, 1] = float(dhdr[0x28, 0x1052].value)
        IM.append(dhdr.pixel_array)
        c += 1

    # check if orientation/resolution is the same for all slices
    if np.sum(Orn - Orn[0, :]) > 1e-6:
        log.error('varying orientation for slices')
    else:
        Orn = Orn[0, :]
    if np.sum(R - R[0, :]) > 1e-6:
        log.error('varying resolution for slices')
    else:
        R = R[0, :]

    # Patient Position
    # Rows and Columns
    if [0x28, 0x10] in dhdr and [0x28, 0x11] in dhdr:
        SZ2 = dhdr[0x28, 0x10].value
        SZ1 = dhdr[0x28, 0x11].value
    # image resolution
    SZ_VX2 = R[0]
    SZ_VX1 = R[1]

    # now sort the images along k-dimension
    k = np.argmin(abs(Orn[:3] + Orn[3:]))
    # sorted indeces
    si = np.argsort(P[:, k])
    Pos = np.zeros(P.shape, dtype=np.float64)
    im = np.zeros((SZ0, SZ1, SZ2), dtype=np.float32)

    # check if the detentions are in agreement (the pixel array could be transposed...)
    if IM[0].shape[0] == SZ1:
        for i in range(SZ0):
            im[i, :, :] = IM[si[i]] * SI[si[i], 0] + SI[si[i], 1]
            Pos[i, :] = P[si[i]]
    else:
        for i in range(SZ0):
            im[i, :, :] = IM[si[i]].T * SI[si[i], 0] + SI[si[i], 1]
            Pos[i, :] = P[si[i]]

    # proper slice thickness
    Zz = (P[si[-1], 2] - P[si[0], 2]) / (SZ0-1)
    Zy = (P[si[-1], 1] - P[si[0], 1]) / (SZ0-1)
    Zx = (P[si[-1], 0] - P[si[0], 0]) / (SZ0-1)

    # dictionary for affine and image size for the image
    A = {
        'AFFINE': np.array([[SZ_VX2 * Orn[0], SZ_VX1 * Orn[3], Zx, Pos[0, 0]],
                            [SZ_VX2 * Orn[1], SZ_VX1 * Orn[4], Zy, Pos[0, 1]],
                            [SZ_VX2 * Orn[2], SZ_VX1 * Orn[5], Zz, Pos[0, 2]], [0., 0., 0., 1.]]),
        'SHAPE': (SZ0, SZ1, SZ2)}

    # the returned image is already scaled according to the dcm header
    return {'im': im, 'affine': A['AFFINE'], 'shape': A['SHAPE'], 'orient': ornt, 'sruid': sruid}
