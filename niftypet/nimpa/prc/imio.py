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
    if output=='image' or output=='all':
        imr = nim.get_data()
        # replace NaNs if requested
        if isinstance(nan_replace, numbers.Number): imr[np.isnan(imr)]=nan_replace
        # Flip y-axis and z-axis and then transpose.  Depends if dynamic (4 dimensions) or static (3 dimensions)
        if len(nim.shape)==4:
            imr  = np.transpose(imr[:,::-1,::-1,:], (3, 2, 1, 0))
        elif len(nim.shape)==3:
            imr  = np.transpose(imr[:,::-1,::-1], (2, 1, 0))
    if output=='affine' or output=='all':
        A = nim.get_sform()
        if not A[:3,:3].any():
            A = nim.get_qform()

    if output=='all':
        out = {'im':imr, 'affine':A, 'dtype':nim.get_data_dtype(), 'shape':imr.shape, 'hdr':nim.header}
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

def array2nii(im, A, fnii, descrip=''):
    '''Store the numpy array 'im' to a NIfTI file 'fnii'.
    ----
    Arguments:
        'im': image to be stored in NIfTI
        'A': affine transformation
        'fnii': NIfTI file name.
        'descrip': the description given to the file
    '''

    if im.ndim==3:
        im = np.transpose(im, (2, 1, 0))
    elif im.ndim==4:
        im = np.transpose(im, (3, 2, 1, 0))
    else:
        raise StandardError('unrecognised image dimensions')

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

def nii_ugzip(imfile):
    '''Uncompress *.nii.gz file'''
    import gzip
    with gzip.open(imfile, 'rb') as f:
        s = f.read()
    # Now store the uncompressed data
    fout = imfile[:-3]
    # store uncompressed file data from 's' variable
    with open(fout, 'wb') as f:
        f.write(s)
    return fout

def nii_gzip(imfile):
    '''Compress *.nii.gz file'''
    import gzip
    with open(imfile, 'rb') as f:
        d = f.read()
    # Now store the uncompressed data
    fout = imfile+'.gz'
    # store compressed file data from 'd' variable
    with gzip.open(fout, 'wb') as f:
        f.write(d)
    return fout
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
    #------------------------------------------------------




#================================================================================

def niisort(fims):
    ''' Sort all input NIfTI images and check their shape.
        Output dictionary of image files and their properties.
    '''
    # number of NIfTI images in folder
    Nim = 0
    # sorting list (if frame number is present in the form '_frm-dd', where d is a digit)
    sortlist = []

    # non dynamic frames input, assuming False (dynamic input)
    ndf_flg = False

    for f in fims:
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            Nim += 1
            _match = re.search('(?<=_frm-)\d*', f)
            if _match:
                sortlist.append(int(_match.group(0)))
            else:
                sortlist.append(None)
    notfrm = [e==None for e in sortlist]
    if any(notfrm):
        print 'w> only some images may be dynamic frames.'
    if all(notfrm):
        print 'w> none image is a dynamic frame.'
        sortlist = range(Nim)
        ndf_flg = True
    # number of frames (can be larger than the # images)
    Nfrm = max(sortlist)+1
    # sort the list according to the frame numbers
    _fims = ['Blank']*Nfrm
    # list of NIfTI image shapes and data types used
    shape = []
    dtype = []
    _nii = []
    for i in range(Nim):
        if not notfrm[i]:
            _fims[sortlist[i]] = fims[i]
            _nii = nib.load(fims[i])
            dtype.append(_nii.get_data_dtype()) 
            shape.append(_nii.shape)
        elif ndf_flg:
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

    # get the images into an array
    _imin = np.zeros((Nfrm,)+_nii.shape[::-1], dtype=_nii.get_data_dtype())
    for i in range(Nfrm):
        if i in sortlist:
            imdic = getnii(_fims[i], output='all')
            _imin[i,:,:,:] = imdic['im']
            affine = imdic['affine']

    # return a dictionary
    return {'shape':_nii.shape[::-1],
            'files':_fims, 
            'sortlist':sortlist,
            #'im':_imin[:Nim-sum(notfrm),:,:,:],
            'im':_imin[:Nfrm,:,:,:],
            'affine':affine,
            'dtype':_nii.get_data_dtype(), 
            'N':Nim}


#================================================================================

def dcm2im(fpth):
    ''' Get the DICOM files from 'fpth' into an image with the affine transformation.
        fpth can be a list of DICOM files or a path (string) to the folder with DICOM files.
    '''

    # possible DICOM file extensions
    ext = ('dcm', 'DCM', 'ima', 'IMA') 
    # case when given a folder path
    if isinstance(fpth, basestring) and os.path.isdir(fpth):
        SZ0 = len([d for d in os.listdir(fpth) if d.endswith(".dcm")])
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