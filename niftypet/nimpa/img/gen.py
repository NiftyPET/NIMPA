""" NIMPA: functions for neuro image processing and analysis
    Generates images.
"""
__author__    = "Pawel Markiewicz"
__copyright__ = "Copyright 2019"
#-------------------------------------------------------------------------------

import sys
import os

import numpy as np
import math
import scipy.ndimage as ndi




#=================================================
def create_disk(shape_in, r=1, a=0, b=0, gen_scale=1, threshold=None):

    if len(shape_in)==2:
        shape = (1,)+shape_in
    if len(shape_in)==3:
        shape = shape_in
    
    imsk = np.zeros((gen_scale*shape[1], gen_scale*shape[2]), dtype=np.float32)
    for t in np.arange(0, math.pi, math.pi/(gen_scale*400)):
        x = gen_scale*r*np.cos(t) + gen_scale*a
        y = gen_scale*r*np.sin(t) + gen_scale*b
        
        for ys in np.arange(-y+2*b*gen_scale, y, 0.5):
            v = 0.5*gen_scale*shape[1] - np.ceil(ys)
            u = 0.5*gen_scale*shape[2] + np.floor(x)
            imsk[v.astype(np.int16), u.astype(np.int16)] = 1.

    if gen_scale>1:
        imsk = ndi.interpolation.zoom(imsk,gen_scale**-1,order=1)

    if threshold:
        imsk = imsk>threshold
    
    if len(shape_in)==3:
        msk = np.repeat(imsk.reshape((1, shape[1], shape[1])), shape[0], axis=0)
    elif len(shape_in)==2:
        msk = imsk
    return msk
#=================================================



#=================================================
def profile_points(im, p0, p1, steps=100):

    p = np.array([p1[0]-p0[0], p1[1]-p0[1]])
    nrm = np.sum(p**2)**.5
    p = p/nrm

    tt = np.linspace(0, nrm, steps)
    profile = np.zeros(len(tt), dtype=im.dtype)

    c = 0
    for t in tt:
        u = t*p[0] + p0[0]
        v = t*p[1] + p0[1]

        profile[c] = im[int(round(v)), int(round(u))]
        c+=1

    return profile
#=================================================