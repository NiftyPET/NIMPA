""" NIMPA: functions for neuro image processing and analysis.
	Includes functions relating to image registration/segmentation.
    
"""
__author__    = "Pawel Markiewicz"
__copyright__ = "Copyright 2018"
#-------------------------------------------------------------------------------

import numpy as np
import sys, os
import imio


def dice_coeff(im1, im2, val=1):
    ''' Calculate Dice score for parcellation images <im1> and <im2> and ROI value <val>.
        Input images can be given as:
            1. paths to NIfTI image files or as
            2. Numpy arrays.
        The ROI value can be given as:
            1. a single integer representing one ROI out of many in the parcellation
               images (<im1> and <im2>) or as
            2. a list of integers to form a composite ROI used for the association test.
        Outputs a float number representing the Dice score.
    '''

    if isinstance(im1, basestring) and isinstance(im2, basestring) \
    and os.path.isfile(im1) and os.path.basename(im1).endswith(('nii', 'nii.gz')) \
    and os.path.isfile(im2) and os.path.basename(im2).endswith(('nii', 'nii.gz')):
        imn1 = imio.getnii(im1, output='image')
        imn2 = imio.getnii(im2, output='image')
    elif isinstance(im1, (np.ndarray, np.generic)) and isinstance(im1, (np.ndarray, np.generic)):
        imn1 = im1
        imn2 = im2
    else:
        raise TypeError('Unrecognised or Mismatched Images.')

    # a single value corresponding to one ROI
    if isinstance(val, (int, long)):
        imv1 = (imn1 == val)
        imv2 = (imn2 == val)
    # multiple values in list corresponding to a composite ROI
    elif isinstance(val, list) and all([isinstance(v, (int, long)) for v in val]):
        imv1 = (imn1==val[0])
        imv2 = (imn2==val[0])
        for v in val[1:]:
            # boolean addition to form a composite ROI
            imv1 += (imn1==v)
            imv2 += (imn2==v)
    else:
        raise TypeError('ROI Values have to be integer (single or in a list).')
        

    if imv1.shape != imv2.shape:
        raise ValueError('Shape Mismatch: Input images must have the same shape.')

    #-compute Dice coefficient
    intrsctn = np.logical_and(imv1, imv2)

    return 2. * intrsctn.sum() / (imv1.sum() + imv2.sum())




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dice_coeff_multiclass(im1, im2, roi2ind):
    ''' Calculate Dice score for parcellation images <im1> and <im2> and ROI value <val>.
        Input images can be given as:
            1. paths to NIfTI image files or as
            2. Numpy arrays.
        The ROI value must be given as a dictionary of lists of indexes for each ROI
        Outputs a float number representing the Dice score.
    '''

    if isinstance(im1, basestring) and isinstance(im2, basestring) \
    and os.path.isfile(im1) and os.path.basename(im1).endswith(('nii', 'nii.gz')) \
    and os.path.isfile(im2) and os.path.basename(im2).endswith(('nii', 'nii.gz')):
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
    for k in roi2ind.keys():

    	# multiple values in list corresponding to a composite ROI
        imv1 = (imn1==roi2ind[k][0])
        imv2 = (imn2==roi2ind[k][0])
        for v in roi2ind[k][1:]:
            # boolean addition to form a composite ROI
            imv1 += (imn1==v)
            imv2 += (imn2==v)

	    #-compute Dice coefficient
    	intrsctn = np.logical_and(imv1, imv2)
    	out[k] = 2. * intrsctn.sum() / (imv1.sum() + imv2.sum())

    return out