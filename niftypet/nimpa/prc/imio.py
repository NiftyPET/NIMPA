"""image input/output functionalities."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
#-------------------------------------------------------------------------------


import sys
import os
import nibabel as nib
import numpy as np
import datetime
import re

#---------------------------------------------------------------
def create_dir(pth):
    if not os.path.exists(pth):    
        os.makedirs(pth)

#---------------------------------------------------------------
def time_stamp():
    now    = datetime.datetime.now()
    nowstr = str(now.year)+'-'+str(now.month)+'-'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)
    return nowstr

#---------------------------------------------------------------
def fwhm2sig (fwhm, voxsize=2.0):
    return (fwhm/voxsize) / (2*(2*np.log(2))**.5)


#================================================================================
def getnii(fim, output='image'):
    '''Get PET image from NIfTI file.
    ----------
    Return:
        'image': outputs just an image (4D or 3D)
        'affine': outputs just the affine matrix
        'all': outputs all as a dictionary
    '''
    nim = nib.load(fim)
    if output=='image' or output=='all':
        imr = nim.get_data()
        imr[np.isnan(imr)]=0
        # Flip y-axis and z-axis and then transpose.  Depends if dynamic (4 dimensions) or static (3 dimensions)
        if len(nim.shape)==4:
            imr  = np.transpose(imr[:,::-1,::-1,:], (3, 2, 1, 0))
        elif len(nim.shape)==3:
            imr  = np.transpose(imr[:,::-1,::-1], (2, 1, 0))
    if output=='affine' or output=='all':
        A = nim.get_sform()

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



def niisort(fims):
    ''' Sort all input NIfTI images and check their shape.
        Output dictionary of image files and their properties.
    '''
    # number of NIfTI images in folder
    Nim = 0
    # sorting list (if frame number is present in the form '_frm-dd', where d is a digit)
    sortlist = []

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
        print 'w> only some images are dynamic frames.'
    if all(notfrm):
        print 'w> none image is a dynamic frame.'
        sortlist = range(Nim)
    # number of frames (can be larger than the # images)
    Nfrm = max(sortlist)+1
    # sort the list according to the frame numbers
    _fims = ['Blank']*Nfrm
    # list of NIfTI image shapes and data types used
    shape = []
    dtype = []
    for i in range(Nim):
        if not notfrm[i]:
            _fims[sortlist[i]] = fims[i]
            _nii = nib.load(fims[i])
            dtype.append(_nii.get_data_dtype()) 
            shape.append(_nii.shape)

    # check if all images are of the same shape and data type
    if not shape.count(_nii.shape)==len(shape):
        raise ValueError('Input images are of different shapes.')
    if not dtype.count(_nii.get_data_dtype())==len(dtype):
        raise TypeError('Input images are of different data types.')
    # image shape must be 3D
    if not len(_nii.shape)==3:
        raise ValueError('Input image(s) must be 3D.')

    # get the images into an array
    _imin = np.zeros((Nfrm,)+_nii.shape[::-1], dtype=_nii.get_data_dtype())
    for i in range(Nfrm):
        if i in sortlist:
            imdic = getnii(_fims[i], 'all')
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