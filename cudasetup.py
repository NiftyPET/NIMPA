#!/usr/bin/env python
"""Tools for CUDA compilation and set-up for Python 3."""
from distutils.sysconfig import get_python_inc
import logging
import os
from pkg_resources import resource_filename
import platform
import re
import shutil
import subprocess
import sys
from textwrap import dedent

import numpy as np
__author__      = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__   = "Copyright 2020"
log = logging.getLogger('nimpa.cudasetup')

prefix = sys.prefix
pyhdr = get_python_inc()  # Python header paths
nphdr = np.get_include()  # numpy header path
mincc = 35  # minimum required CUDA compute capability


def path_niftypet_local():
    '''Get the path to the local (home) folder for NiftyPET resources.'''
    # if using conda put the resources in the folder with the environment name
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env = os.environ['CONDA_DEFAULT_ENV']
        log.info('install> conda environment found: {}'.format(env))
    else:
        env = ''
    # create the path for the resources files according to the OS platform
    if platform.system() in ('Linux', 'Darwin'):
        path_resources = os.path.expanduser('~')
    elif platform.system() == 'Windows' :
        path_resources = os.getenv('LOCALAPPDATA')
    else:
        raise ValueError('Unknown operating system: {}'.format(platform.system()))
    path_resources = os.path.join(path_resources, '.niftypet', env)

    return path_resources


def find_cuda():
    '''Locate the CUDA environment on the system.'''
    # search the PATH for NVCC
    for fldr in os.environ['PATH'].split(os.pathsep):
        cuda_path = os.path.join(fldr, 'nvcc')
        if os.path.exists(cuda_path):
            cuda_path = os.path.dirname(os.path.dirname(cuda_path))
            break
        else:
            cuda_path = None

    if cuda_path is None:
        log.warning('nvcc compiler could not be found from the PATH!')
        return

    # serach for the CUDA library path
    lcuda_path = os.path.join(cuda_path, 'lib64')
    if 'LD_LIBRARY_PATH' in os.environ:
        if lcuda_path in os.environ['LD_LIBRARY_PATH'].split(os.pathsep):
            log.info('found CUDA lib64 in LD_LIBRARY_PATH: {}'.format(lcuda_path))
    elif os.path.isdir(lcuda_path):
        log.info('found CUDA lib64 in: {}'.format(lcuda_path))
    else:
        log.warning('folder for CUDA library (64-bit) could not be found!')

    return cuda_path, lcuda_path


def dev_setup():
    '''figure out what GPU devices are available and choose the supported ones.'''
    # check first if NiftyPET was already installed and use the choice of GPU
    path_resources = path_niftypet_local()
    # if so, import the resources and get the constants
    if os.path.isfile(os.path.join(path_resources,'resources.py')):
        sys.path.append(path_resources)
        try:
            import resources
        except ImportError as ie:
            log.error(dedent('''\
                --------------------------------------------------------------------------
                NiftyPET resources file <resources.py> could not be imported.
                It should be in ~/.niftypet/resources.py (Linux) or
                in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
                but likely it does not exists.
                --------------------------------------------------------------------------'''))
    else:
        log.error('resources file not found/installed.')
        return None

    # get all constants and check if device is already chosen
    Cnt = resources.get_setup()
    if 'CCARCH' in Cnt and 'DEVID' in Cnt:
        log.info('using this CUDA architecture(s): {}'.format(Cnt['CCARCH']))
        return Cnt['CCARCH']

    from miutil import cuinfo
    if 'DEVID' in Cnt:
        ccstr = cuinfo.get_nvcc_flags(int(Cnt['DEVID']))
        devid = Cnt['DEVID']
    else:
        devid = cuinfo.get_device_count() - 1
        ccstr = ';'.join(set(map(cuinfo.get_nvcc_flags, range(devid + 1))))

    # passing this setting to resources.py
    fpth = os.path.join(path_resources,'resources.py') #resource_filename(__name__, 'resources/resources.py')
    with open(fpth, 'r') as f:
        rsrc = f.read()
    # get the region of keeping in synch with Python
    i0 = rsrc.find('### start GPU properties ###')
    i1 = rsrc.find('### end GPU properties ###')
    # list of constants which will be kept in sych from Python
    cnt_dict = {'DEV_ID': str(devid), 'CC_ARCH': repr(ccstr)}
    # update the resource.py file
    with  open(fpth, 'w') as f:
        f.write(rsrc[:i0])
        f.write('### start GPU properties ###\n')
        for k, v in cnt_dict.items():
            f.write(k + ' = ' + v + '\n')
        f.write(rsrc[i1:])

    return ccstr


def resources_setup(gpu=True):
    '''
    This function checks CUDA devices, selects some and installs resources.py
    '''
    log.info('installing file <resources.py> into home directory if it does not exist.')
    path_current = os.path.dirname( os.path.realpath(__file__) )
    # path to the install version of resources.py.
    path_install = os.path.join(path_current, 'resources')
    # get the path to the local resources.py (on Linux machines it is in ~/.niftypet)
    path_resources = path_niftypet_local()
    log.info('current path: {}'.format(path_current))

    # flag for the resources file if already installed (initially assumed not)
    flg_resources = False
    # does the local folder for niftypet exists? if not create one.
    if not os.path.exists(path_resources):
        os.makedirs(path_resources)
    # is resources.py in the folder?
    if not os.path.isfile(os.path.join(path_resources,'resources.py')):
        if os.path.isfile(os.path.join(path_install,'resources.py')):
            shutil.copyfile( os.path.join(path_install,'resources.py'), os.path.join(path_resources,'resources.py') )
        else:
            raise IOError('could not find <resources.py')
    else:
        log.info('<resources.py> should be already in the local NiftyPET folder:\n  {}'.format(path_resources))
        # set the flag that the resources file is already there
        flg_resources = True
        sys.path.append(path_resources)
        try:
            import resources
        except ImportError as ie:
            log.error(dedent('''\
                --------------------------------------------------------------------------
                NiftyPET resources file <resources.py> could not be imported.
                It should be in ~/.niftypet/resources.py (Linux) or
                in //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
                but likely it does not exists.
                --------------------------------------------------------------------------'''))

    # find available GPU devices, select one or more and output the compilation flags
    # return gpuarch for cmake compilation
    return dev_setup() if gpu else ''
