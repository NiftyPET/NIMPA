"""image input/output functionalities."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
#-------------------------------------------------------------------------------


import sys
import os
import nibabel as nib
import pydicom as dcm
import numpy as np
import datetime
import re
import shutil
import glob

from subprocess import call

#> NiftyPET resources
import resources as rs

# possible extentions for DICOM files
dcmext = ('dcm', 'DCM', 'ima', 'IMA')


#---------------------------------------------------------------
def create_dir(pth):
    if not os.path.exists(pth):    
        os.makedirs(pth)

#---------------------------------------------------------------
def time_stamp(simple_ascii=False):
    now    = datetime.datetime.now()
    if simple_ascii:
        nowstr = str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'_'+str(now.hour)+'h'+str(now.minute)
    else:
        nowstr = str(now.year)+'-'+str(now.month)+'-'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)
    return nowstr

#---------------------------------------------------------------
def fwhm2sig (fwhm, voxsize=2.0):
    return (fwhm/voxsize) / (2*(2*np.log(2))**.5)


#================================================================================
def getnii(fim, nan_replace=None, output='image'):
    '''Get PET image from NIfTI file.
    fim: input file name for the nifty image
    nan_replace:    the value to be used for replacing the NaNs in the image.
                    by default no change (None).
    output: option for choosing output: image, affine matrix or a dictionary with all info.
    ----------
    Return:
        'image': outputs just an image (4D or 3D)
        'affine': outputs just the affine matrix
        'all': outputs all as a dictionary
    '''

    import numbers

    nim = nib.load(fim)

    dim = nim.header.get('dim')
    dimno = dim[0]

    if output=='image' or output=='all':
        imr = nim.get_data()
        # replace NaNs if requested
        if isinstance(nan_replace, numbers.Number): imr[np.isnan(imr)]

        imr = np.squeeze(imr)
        if dimno!=imr.ndim and dimno==4:
            dimno = imr.ndim
        
        #> get orientations from the affine
        ornt = nib.orientations.axcodes2ornt(nib.aff2axcodes(nim.affine))
        trnsp = tuple(np.int8(ornt[::-1,0]))
        flip  = tuple(np.int8(ornt[:,1]))

        #> flip y-axis and z-axis and then transpose.  Depends if dynamic (4 dimensions) or static (3 dimensions)
        if dimno==4:
            imr = np.transpose(imr[::-flip[0],::-flip[1],::-flip[2],:], (3,)+trnsp)
        elif  dimno==3:
            imr = np.transpose(imr[::-flip[0],::-flip[1],::-flip[2]], trnsp)
    
    if output=='affine' or output=='all':
        # A = nim.get_sform()
        # if not A[:3,:3].any():
        #     A = nim.get_qform()
        A = nim.affine

    if output=='all':
        out = { 'im':imr,
                'affine':A,
                'fim':fim,
                'dtype':nim.get_data_dtype(),
                'shape':imr.shape,
                'hdr':nim.header,
                'transpose':trnsp,
                'flip':flip}
    elif output=='image':
        out = imr
    elif output=='affine':
        out = A
    else:
        raise NameError('Unrecognised output request!')

    return out

def getnii_descr(fim):
    '''
    Extracts the custom description header field to dictionary
    '''
    nim = nib.load(fim)
    hdr = nim.header
    rcnlst = hdr['descrip'].item().split(';')
    rcndic = {}
    
    if rcnlst[0]=='':
        # print 'w> no description in the NIfTI header'
        return rcndic
    
    for ci in range(len(rcnlst)):
        tmp = rcnlst[ci].split('=')
        rcndic[tmp[0]] = tmp[1]
    return rcndic


def array2nii(im, A, fnii, descrip='', trnsp=(), flip=(), storage_as=[]):
    '''Store the numpy array 'im' to a NIfTI file 'fnii'.
    ----
    Arguments:
        'im':       image to be stored in NIfTI
        'A':        affine transformation
        'fnii':     output NIfTI file name.
        'descrip':  the description given to the file
        'trsnp':    transpose/permute the dimensions.
                    In NIfTI it has to be in this order: [x,y,z,t,...])
        'flip':     flip tupple for flipping the direction of x,y,z axes. 
                    (1: no flip, -1: flip)
        'storage_as': uses the flip and displacement as given by the following
                    NifTI dictionary, obtained using
                    nimpa.getnii(filepath, output='all').
    '''

    if not len(trnsp) in [0,3,4] and not len(flip) in [0,3]:
        raise ValueError('e> number of flip and/or transpose elements is incorrect.')

    #---------------------------------------------------------------------------
    #> TRANSLATIONS and FLIPS
    #> get the same geometry as the input NIfTI file in the form of dictionary,
    #>>as obtained from getnii(..., output='all')

    #> permute the axis order in the image array
    if isinstance(storage_as, dict) and 'transpose' in storage_as \
            and 'flip' in storage_as:

        trnsp = (storage_as['transpose'].index(0),
                 storage_as['transpose'].index(1),
                 storage_as['transpose'].index(2))

        flip = storage_as['flip']
    

    if trnsp==():
        im = im.transpose()
    #> check if the image is 4D (dynamic) and modify as needed
    elif len(trnsp)==3 and im.ndim==4:
        trnsp = tuple([t+1 for t in trnsp] + [0])
        im = im.transpose(trnsp)
    else:
        im = im.transpose(trnsp)
    

    #> perform flip of x,y,z axes after transposition into proper NIfTI order
    if flip!=() and len(flip)==3:
        im = im[::-flip[0], ::-flip[1], ::-flip[2], ...]
    #---------------------------------------------------------------------------

    nii = nib.Nifti1Image(im, A)
    hdr = nii.header
    hdr.set_sform(None, code='scanner')
    hdr['cal_max'] = np.max(im) #np.percentile(im, 90) #
    hdr['cal_min'] = np.min(im)
    hdr['descrip'] = descrip
    nib.save(nii, fnii)



def orientnii(imfile):
    '''Get the orientation from NIfTI sform.  Not fully functional yet.'''
    strorient = ['L-R', 'S-I', 'A-P']
    niiorient = []
    niixyz = np.zeros(3,dtype=np.int8)
    if os.path.isfile(imfile):
        nim = nib.load(imfile)
        pct = nim.get_data()
        A = nim.get_sform()
        for i in range(3):
            niixyz[i] = np.argmax(abs(A[i,:-1]))
            niiorient.append( strorient[ niixyz[i] ] )
        print niiorient

def nii_ugzip(imfile, outpath=''):
    '''Uncompress *.gz file'''
    import gzip
    with gzip.open(imfile, 'rb') as f:
        s = f.read()
    # Now store the uncompressed data
    if outpath=='':
        fout = imfile[:-3]
    else:
        fout = os.path.join(outpath, os.path.basename(imfile)[:-3])
    # store uncompressed file data from 's' variable
    with open(fout, 'wb') as f:
        f.write(s)
    return fout

def nii_gzip(imfile, outpath=''):
    '''Compress *.gz file'''
    import gzip
    with open(imfile, 'rb') as f:
        d = f.read()
    # Now store the compressed data
    if outpath=='':
        fout = imfile+'.gz'
    else:
        fout = os.path.join(outpath, os.path.basename(imfile)+'.gz')
    # store compressed file data from 'd' variable
    with gzip.open(fout, 'wb') as f:
        f.write(d)
    return fout
#===============================================================================


def pick_t1w(mri):
    ''' Pick the MR T1w from the dictionary for MR->PET registration.
    '''

    if isinstance(mri, dict):
        # check if NIfTI file is given
        if 'T1N4' in mri and os.path.isfile(mri['T1N4']):
            ft1w = mri['T1N4']
        # or another bias corrected
        elif 'T1bc' in mri and os.path.isfile(mri['T1bc']):
            ft1w = mri['T1bc']
        elif 'T1nii' in mri and os.path.isfile(mri['T1nii']):
            ft1w = mri['T1nii']
        elif 'T1DCM' in mri and os.path.exists(mri['MRT1W']):
            # create file name for the converted NIfTI image
            fnii = 'converted'
            call( [rs.DCM2NIIX, '-f', fnii, mri['T1nii'] ] )
            ft1nii = glob.glob( os.path.join(mri['T1nii'], '*converted*.nii*') )
            ft1w = ft1nii[0]
        else:
            print 'e> disaster: could not find a T1w image!'
            return None
            
    else:
        ('e> no correct input found for the T1w image')
        return None

    return ft1w


def dcminfo(dcmvar, verbose=True):
    ''' Get basic info about the DICOM file/header.
    '''

    if isinstance(dcmvar, basestring):
        if verbose:
            print 'i> provided DICOM file:', dcmvar
        dhdr = dcm.dcmread(dcmvar)
    elif isinstance(dcmvar, dict):
        dhdr = dcmvar
    elif isinstance(dcmvar, dcm.dataset.FileDataset):
        dhdr = dcmvar

    dtype   = dhdr[0x08, 0x08].value
    if verbose:
        print '   Image Type:', dtype

    #-------------------------------------------
    #> scanner ID
    scanner_vendor = 'unknown'
    if [0x008, 0x070] in dhdr:
        scanner_vendor = dhdr[0x008, 0x070].value

    scanner_model = 'unknown'
    if [0x008, 0x1090] in dhdr:
        scanner_model = dhdr[0x008, 0x1090].value

    scanner_id = 'other'
    if any(s in scanner_model for s in ['mMR', 'Biograph']) and 'siemens' in scanner_vendor.lower():
        scanner_id = 'mmr'
    #-------------------------------------------

    #> CSA type (mMR)
    csatype = ''
    if [0x29, 0x1108] in dhdr:
        csatype = dhdr[0x29, 0x1108].value
        if verbose:
            print '   CSA Data Type:', csatype

    #> DICOM comment or on MR parameters
    cmmnt   = ''
    if [0x20, 0x4000] in dhdr:
        cmmnt = dhdr[0x0020, 0x4000].value
        if verbose:
            print '   Comments:', cmmnt

    #> MR parameters (echo time, etc) 
    TR = 0
    TE = 0
    if [0x18, 0x80] in dhdr:
        TR = float(dhdr[0x18, 0x80].value)
        if verbose: print '   TR:', TR
    if [0x18, 0x81] in dhdr:
        TE = float(dhdr[0x18, 0x81].value)
        if verbose: print '   TE:', TE


    #> check if it is norm file
    if any('PET_NORM' in s for s in dtype) or cmmnt=='PET Normalization data' or csatype=='MRPETNORM':
        out = ['raw', 'norm', scanner_id]

    elif any('PET_LISTMODE' in s for s in dtype) or cmmnt=='Listmode' or csatype=='MRPETLM_LARGE':
        out = ['raw', 'list', scanner_id]

    elif any('MRPET_UMAP3D' in s for s in dtype) or cmmnt=='MR based umap':
        out = ['raw', 'mumap', 'ute', 'mr', scanner_id]

    elif TR>400 and TR<2500 and TE<20:
        out = ['mr', 't1', scanner_id]

    elif TR>2500 and TE>50:
        out = ['mr', 't2', scanner_id]

    #> UTE's two sequences: UTE2
    elif TR<50 and TE<20 and TE>1:
        out = ['mr', 'ute', 'ute2', scanner_id]

    #> UTE1
    elif TR<50 and TE<20 and TE<0.1 and TR>0 and TE>0:
        out = ['mr', 'ute', 'ute1', scanner_id]

    #> physio data
    elif 'PET_PHYSIO' in dtype or 'physio' in cmmnt.lower():
        out = ['raw', 'physio', scanner_id]

    else:
        out = ['unknown', cmmnt]

    return out


def list_dcm_datain(datain):
    ''' List all DICOM file paths in the datain dictionary of input data.
    '''

    if not isinstance(datain, dict):
        raise ValueError('The input is not a dictionary!')

    dcmlst = []
    # list of mu-map DICOM files
    if 'mumapDCM' in datain:
        dcmump = os.listdir(datain['mumapDCM'])
        # accept only *.dcm extensions
        dcmump = [os.path.join(datain['mumapDCM'],d) for d in dcmump if d.endswith(dcmext)]
        dcmlst += dcmump

    if 'T1DCM' in datain:
        dcmt1 = os.listdir(datain['T1DCM'])
        # accept only *.dcm extensions
        dcmt1 = [os.path.join(datain['T1DCM'],d) for d in dcmt1 if d.endswith(dcmext)]
        dcmlst += dcmt1

    if 'T2DCM' in datain:
        dcmt2 = os.listdir(datain['T2DCM'])
        # accept only *.dcm extensions
        dcmt2 = [os.path.join(datain['T2DCM'],d) for d in dcmt2 if d.endswith(dcmext)]
        dcmlst += dcmt2

    if 'UTE1' in datain:
        dcmute1 = os.listdir(datain['UTE1'])
        # accept only *.dcm extensions
        dcmute1 = [os.path.join(datain['UTE1'],d) for d in dcmute1 if d.endswith(dcmext)]
        dcmlst += dcmute1

    if 'UTE2' in datain:
        dcmute2 = os.listdir(datain['UTE2'])
        # accept only *.dcm extensions
        dcmute2 = [os.path.join(datain['UTE2'],d) for d in dcmute2 if d.endswith(dcmext)]
        dcmlst += dcmute2

    #-list-mode data dcm
    if 'lm_dcm' in datain:
        dcmlst += [datain['lm_dcm']]

    if 'lm_ima' in datain:
        dcmlst += [datain['lm_ima']]

    #-norm
    if 'nrm_dcm' in datain:
        dcmlst += [datain['nrm_dcm']]

    if 'nrm_ima' in datain:
        dcmlst += [datain['nrm_ima']]

    return dcmlst



def dcmanonym(
        dcmpth,
        displayonly=False,
        patient='anonymised',
        physician='anonymised',
        dob='19800101',
        verbose=True):

    ''' Anonymise DICOM file(s)
        Arguments:
        > dcmpth:   it can be passed as a single DICOM file, or
                    a folder containing DICOM files, or a list of DICOM file paths.
        > patient:  the name of the patient.
        > physician:the name of the referring physician.
        > dob:      patient's date of birth.
        > verbose:  display processing output.
    '''

    #> check if a single DICOM file
    if isinstance(dcmpth, basestring) and os.path.isfile(dcmpth):
        dcmlst = [dcmpth]
        if verbose:
            print 'i> recognised the input argument as a single DICOM file.'

    #> check if a folder containing DICOM files
    elif isinstance(dcmpth, basestring) and os.path.isdir(dcmpth):
        dircontent = os.listdir(dcmpth)
        #> create a list of DICOM files inside the folder
        dcmlst = [os.path.join(dcmpth,d) for d in dircontent if os.path.isfile(os.path.join(dcmpth,d)) and d.endswith(dcmext)]
        if verbose:
            print 'i> recognised the input argument as the folder containing DICOM files.'

    #> check if a folder containing DICOM files
    elif isinstance(dcmpth, list):
        if not all([os.path.isfile(d) and d.endswith(dcmext) for d in dcmpth]):
            raise IOError('Not all files in the list are DICOM files.')
        dcmlst = dcmpth
        if verbose:
            print 'i> recognised the input argument as the list of DICOM file paths.'

    #> check if dictionary of data input <datain>
    elif isinstance(dcmpth, dict) and 'corepath' in dcmpth:
        dcmlst = list_dcm_datain(dcmpth)
        if verbose:
            print 'i> recognised the input argument as the dictionary of scanner data.'

    else:
        raise IOError('Unrecognised input!')



    for dcmf in dcmlst:

        #> read the file
        dhdr = dcm.dcmread(dcmf)

        #> get the basic info about the DICOM file
        dcmtype = dcminfo(dhdr, verbose=False)
        if verbose:
            print '-------------------------------'
            print 'i> the DICOM file is for:', dcmtype

        #> anonymise mMR data.
        if 'mmr' in dcmtype:

            if [0x029, 0x1120] in dhdr and dhdr[0x029, 0x1120].name=='[CSA Series Header Info]':
                csafield = dhdr[0x029, 0x1120]
                csa = csafield.value
            elif [0x029, 0x1020] in dhdr and dhdr[0x029, 0x1020].name=='[CSA Series Header Info]':
                csafield = dhdr[0x029, 0x1020]
                csa = csafield.value
            else:
                csa = ''

            # string length considered for replacement
            strlen = 200
            
            idx = [m.start() for m in re.finditer(r'([Pp]atients{0,1}[Nn]ame)', csa)]
            if idx and verbose:
                print '   > found sensitive information deep in DICOM headers:', dcmtype


            #> run the anonymisation    
            iupdate = 0

            for i in idx:
                ci = i - iupdate

                if displayonly:
                    print '   > sensitive info:'
                    print '    ', csa[ci:ci+strlen]
                    continue

                rplcmnt = re.sub( r'(\{\s*\"{1,2}\W*\w+\W*\w+\W*\"{1,2}\s*\})',
                        '{ ""' +patient+ '"" }',
                         csa[ci:ci+strlen]
                )
                #> update string
                csa = csa[:ci] + rplcmnt + csa[ci+strlen:]
                print '   > removed sensitive information.'
                #> correct for the number of removed letters
                iupdate = strlen-len(rplcmnt)

            #> update DICOM
            if not displayonly and csa!='':
                csafield.value = csa


        #> Patient's name
        if [0x010,0x010] in dhdr:
            if displayonly:
                print '   > sensitive info:', dhdr[0x010,0x010].name
                print '    ', dhdr[0x010,0x010].value
            else:
                dhdr[0x010,0x010].value = patient
                if verbose: print '   > anonymised patients name'

        #> date of birth
        if [0x010,0x030] in dhdr:
            if displayonly:
                print '   > sensitive info:', dhdr[0x010,0x030].name
                print '    ', dhdr[0x010,0x030].value
            else:
                dhdr[0x010,0x030].value = dob
                if verbose: print '   > anonymised date of birh'

        #> physician's name
        if [0x008, 0x090] in dhdr:
            if displayonly:
                print '   > sensitive info:',  dhdr[0x008,0x090].name
                print '    ', dhdr[0x008,0x090].value
            else:
                dhdr[0x008,0x090].value = physician
                if verbose: print '   > anonymised physician name'

        dhdr.save_as(dcmf)



#================================================================================


def dcmsort(folder, copy_series=False, verbose=False):
    ''' Sort out the DICOM files in the folder according to the recorded series.
    '''

    # list files in the input folder
    files = os.listdir(folder)

    srs = {}
    for f in files:
        if os.path.isfile(os.path.join(folder, f)) and f.endswith(dcmext):
            dhdr = dcm.read_file(os.path.join(folder, f))
            #---------------------------------
            # image size
            imsz = np.zeros(2, dtype=np.int64)
            if [0x028,0x010] in dhdr:
                imsz[0] = dhdr[0x028,0x010].value
            if [0x028,0x011] in dhdr:
                imsz[1] = dhdr[0x028,0x011].value
            # voxel size
            vxsz = np.zeros(3, dtype=np.float64)
            if [0x028,0x030] in dhdr and [0x018,0x050] in dhdr:
                pxsz =  np.array([float(e) for e in dhdr[0x028,0x030].value])
                vxsz[:2] = pxsz
                vxsz[2] = float(dhdr[0x018,0x050].value)
            # orientation
            ornt = np.zeros(6, dtype=np.float64)
            if [0x020,0x037] in dhdr:
                ornt = np.array([float(e) for e in dhdr[0x20,0x37].value])
            # seires description, time and study time
            srs_dcrp = dhdr[0x0008, 0x103e].value
            srs_time = dhdr[0x0008, 0x0031].value[:6]
            std_time = dhdr[0x0008, 0x0030].value[:6]

            if verbose:
                print 'series desciption:', srs_dcrp
                print 'series time:', srs_time
                print 'study  time:', std_time
                print '---------------------------------------------------'

            #----------
            # series for any category (can be multiple scans within the same category)
            recognised_series = False
            srs_k = srs.keys()
            for s in srs_k:
                if  np.array_equal(srs[s]['imorient'],  ornt) and \
                    np.array_equal(srs[s]['imsize'],    imsz) and \
                    np.array_equal(srs[s]['voxsize'],   vxsz) and \
                    srs[s]['tseries'] == srs_time:
                    recognised_series = True
                    break
            # if series was not found, create one
            if not recognised_series:
                s = srs_dcrp + '_' + srs_time
                srs[s] = {}
                srs[s]['imorient']  = ornt
                srs[s]['imsize']    = imsz
                srs[s]['voxsize']   = vxsz
                srs[s]['tseries']   = srs_time
            # append the file name
            if 'files' not in srs[s]: srs[s]['files'] = []
            if copy_series:
                srsdir = os.path.join(folder, s)
                create_dir( srsdir )
                shutil.copy(os.path.join(folder, f), srsdir)
                srs[s]['files'].append( os.path.join(srsdir, f) )
            else:
                srs[s]['files'].append( os.path.join(folder, f) )

    return srs
#-------------------------------------------------------------------------------




#===============================================================================

def niisort(
        fims,
        memlim=True
    ):

    ''' Sort all input NIfTI images and check their shape.
        Output dictionary of image files and their properties.
        Options:
            memlim -- when processing large numbers of frames the memory may
            not be large enough.  memlim causes that the output does not contain
            all the arrays corresponding to the images.
    '''
    # number of NIfTI images in folder
    Nim = 0
    # sorting list (if frame number is present in the form '_frm<dd>', where d is a digit)
    sortlist = []

    for f in fims:
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            Nim += 1
            _match = re.search('(?<=_frm)\d*', f)
            if _match:
                frm = int(_match.group(0))
                freelists = [frm not in l for l in sortlist]
                listidx = [i for i,f in enumerate(freelists) if f]
                if listidx:
                    sortlist[listidx[0]].append(frm)
                else:
                    sortlist.append([frm])
            else:
                sortlist.append([None])

    if len(sortlist)>1:
        # if more than one dynamic set is given, the dynamic mode is cancelled.
        dyn_flg = False
        sortlist = range(Nim)
    elif len(sortlist)==1:
        dyn_flg = True
        sortlist = sortlist[0]
    else:
        raise ValueError('e> niisort input error.')
        
    
    # number of frames (can be larger than the # images)
    Nfrm = max(sortlist)+1
    # sort the list according to the frame numbers
    _fims = ['Blank']*Nfrm
    # list of NIfTI image shapes and data types used
    shape = []
    dtype = []
    _nii = []
    for i in range(Nim):
        if dyn_flg:
            _fims[sortlist[i]] = fims[i]
            _nii = nib.load(fims[i])
            dtype.append(_nii.get_data_dtype()) 
            shape.append(_nii.shape)
        else:
            _fims[i] = fims[i]
            _nii = nib.load(fims[i])
            dtype.append(_nii.get_data_dtype()) 
            shape.append(_nii.shape)

    # check if all images are of the same shape and data type
    if _nii and not shape.count(_nii.shape)==len(shape):
        raise ValueError('Input images are of different shapes.')
    if _nii and not dtype.count(_nii.get_data_dtype())==len(dtype):
        raise TypeError('Input images are of different data types.')
    # image shape must be 3D
    if _nii and not len(_nii.shape)==3:
        raise ValueError('Input image(s) must be 3D.')

    out = {'shape':_nii.shape[::-1],
            'files':_fims, 
            'sortlist':sortlist,
            'dtype':_nii.get_data_dtype(), 
            'N':Nim}

    if memlim and Nfrm>50:
        imdic = getnii(_fims[0], output='all')
        affine = imdic['affine']
    else:
        # get the images into an array
        _imin = np.zeros((Nfrm,)+_nii.shape[::-1], dtype=_nii.get_data_dtype())
        for i in range(Nfrm):
            if i in sortlist:
                imdic = getnii(_fims[i], output='all')
                _imin[i,:,:,:] = imdic['im']
                affine = imdic['affine']
        out['im'] = _imin[:Nfrm,:,:,:]

    out['affine'] = affine

    return out


#================================================================================
def dcm2nii(
        dcmpth,
        fimout = '',
        fprefix = 'converted-from-DICOM_',
        fcomment = '',
        outpath = '',
        timestamp = True,
        executable = '',
        force = False,
    ):
    ''' Convert DICOM files in folder (indicated by <dcmpth>) using DCM2NIIX
        third-party software.
    '''

    # skip conversion if the output already exists and not force is selected
    if os.path.isfile(fimout) and not force:
        return fimout

    if executable=='':
        try:
            import resources
            executable = resources.DCM2NIIX
        except:
            raise NameError('e> could not import resources \
                    or find variable DCM2NIIX in resources.py')
    elif not os.path.isfile(executable):
        raise IOError('e> the executable is incorrect!')

    if not os.path.isdir(dcmpth):
        raise IOError('e> the provided DICOM path is not a folder!')

    #> output path
    if outpath=='' and fimout!='' and '/' in fimout:
        opth = os.path.dirname(fimout)
        if opth=='':
            opth = dcmpth
        fimout = os.path.basename(fimout)

    elif outpath=='':
        opth = dcmpth

    else:
        opth = outpath

    create_dir(opth)

    if fimout=='':
        fimout = fprefix
        if timestamp:
            fimout += time_stamp(simple_ascii=True)

    fimout = fimout.split('.nii')[0]


    # convert the DICOM mu-map images to nii
    call([executable, '-f', fimout, '-o', opth, dcmpth])

    fniiout = glob.glob( os.path.join(opth, '*'+fimout+'*.nii*') )

    if fniiout:
        return fniiout[0]
    else:
        raise ValueError('e> could not get the output file!')




#================================================================================
def dcm2im(fpth):
    ''' Get the DICOM files from 'fpth' into an image with the affine transformation.
        fpth can be a list of DICOM files or a path (string) to the folder with DICOM files.
    '''

    # possible DICOM file extensions
    ext = ('dcm', 'DCM', 'ima', 'IMA') 
    
    # case when given a folder path
    if isinstance(fpth, basestring) and os.path.isdir(fpth):
        SZ0 = len([d for d in os.listdir(fpth) if d.endswith(ext)])
        # list of DICOM files
        fdcms = os.listdir(fpth)
        fdcms = [os.path.join(fpth,f) for f in fdcms if f.endswith(ext)]
    
    # case when list of DICOM files is given
    elif isinstance(fpth, list) and os.path.isfile(os.path.join(fpth[0])):
        SZ0 = len(fpth)
        # list of DICOM files
        fdcms = fpth
        fdcms = [f for f in fdcms if f.endswith(ext)]
    else:
        raise NameError('Unrecognised input for DICOM files.')

    if SZ0<1:
        print 'e> no DICOM images in the specified path.'
        raise IOError('Input DICOM images not recognised')

    # pick single DICOM header
    dhdr = dcm.read_file(fdcms[0])

    #------------------------------------
    # some info, e.g.: patient position and series UID
    if [0x018, 0x5100] in dhdr:
        ornt = dhdr[0x18,0x5100].value
    else:
        ornt = 'unkonwn'
    # Series UID
    sruid = dhdr[0x0020, 0x000e].value
    #------------------------------------

    #------------------------------------
    # INIT
    # image position 
    P = np.zeros((SZ0,3), dtype=np.float64)
    #image orientation
    Orn = np.zeros((SZ0,6), dtype=np.float64)
    #xy resolution
    R = np.zeros((SZ0,2), dtype=np.float64)
    #slice thickness
    S = np.zeros((SZ0,1), dtype=np.float64)
    #slope and intercept
    SI = np.ones((SZ0,2), dtype=np.float64)
    SI[:,1] = 0

    #image data as an list of array for now
    IM = []
    #------------------------------------

    c = 0
    for d in fdcms:
        dhdr = dcm.read_file(d)
        if [0x20,0x32] in dhdr and [0x20,0x37] in dhdr and [0x28,0x30] in dhdr:
            P[c,:] = np.array([float(f) for f in dhdr[0x20,0x32].value])
            Orn[c,:] = np.array([float(f) for f in dhdr[0x20,0x37].value])
            R[c,:] = np.array([float(f) for f in dhdr[0x28,0x30].value])
            S[c,:] = float(dhdr[0x18,0x50].value)
        else:
            print 'e> could not read all the DICOM tags.'
            return {'im':[], 'affine':[], 'shape':[], 'orient':ornt, 'sruid':sruid}
            
        if [0x28,0x1053] in dhdr and [0x28,0x1052] in dhdr:
            SI[c,0] = float(dhdr[0x28,0x1053].value)
            SI[c,1] = float(dhdr[0x28,0x1052].value)
        IM.append(dhdr.pixel_array)
        c += 1


    #check if orientation/resolution is the same for all slices
    if np.sum(Orn-Orn[0,:]) > 1e-6:
        print 'e> varying orientation for slices'
    else:
        Orn = Orn[0,:]
    if np.sum(R-R[0,:]) > 1e-6:
        print 'e> varying resolution for slices'
    else:
        R = R[0,:]

    # Patient Position
    #patpos = dhdr[0x18,0x5100].value
    # Rows and Columns
    if [0x28,0x10] in dhdr and [0x28,0x11] in dhdr:
        SZ2 = dhdr[0x28,0x10].value
        SZ1 = dhdr[0x28,0x11].value
    # image resolution
    SZ_VX2 = R[0]
    SZ_VX1 = R[1]

    #now sort the images along k-dimension
    k = np.argmin(abs(Orn[:3]+Orn[3:]))
    #sorted indeces
    si = np.argsort(P[:,k])
    Pos = np.zeros(P.shape, dtype=np.float64)
    im = np.zeros((SZ0, SZ1, SZ2 ), dtype=np.float32)

    #check if the detentions are in agreement (the pixel array could be transposed...)
    if IM[0].shape[0]==SZ1:
        for i in range(SZ0):
            im[i,:,:] = IM[si[i]]*SI[si[i],0] + SI[si[i],1]
            Pos[i,:] = P[si[i]]
    else:
        for i in range(SZ0):
            im[i,:,:] = IM[si[i]].T * SI[si[i],0] + SI[si[i],1]
            Pos[i,:] = P[si[i]]

    # proper slice thickness
    Zz = (P[si[-1],2] - P[si[0],2])/(SZ0-1)
    Zy = (P[si[-1],1] - P[si[0],1])/(SZ0-1)
    Zx = (P[si[-1],0] - P[si[0],0])/(SZ0-1)
    

    # dictionary for affine and image size for the image
    A = {
        'AFFINE':np.array([[SZ_VX2*Orn[0], SZ_VX1*Orn[3], Zx, Pos[0,0]],
                           [SZ_VX2*Orn[1], SZ_VX1*Orn[4], Zy, Pos[0,1]],
                           [SZ_VX2*Orn[2], SZ_VX1*Orn[5], Zz, Pos[0,2]],
                           [0., 0., 0., 1.]]),
        'SHAPE':(SZ0, SZ1, SZ2)
    }

    #the returned image is already scaled according to the dcm header
    return {'im':im, 'affine':A['AFFINE'], 'shape':A['SHAPE'], 'orient':ornt, 'sruid':sruid}