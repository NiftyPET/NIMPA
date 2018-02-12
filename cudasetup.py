#!/usr/bin/env python
"""ccompile.py: tools for CUDA compilation and set-up for Python."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
# ---------------------------------------------------------------------------------

from distutils.sysconfig import get_python_inc
from pkg_resources import resource_filename
import re
import os
from os.path import join
import sys
import subprocess
import numpy as np
import platform
import shutil

# get Python prefix
prefix = sys.prefix

# get Python header paths:
pyhdr = get_python_inc()

# get numpy header path:
nphdr = np.get_include()

# minimum required CUDA compute capability 
mincc = 35

# ---------------------------------------------------------------------------------
def path_niftypet_local():
    '''Get the path to the local (home) folder for NiftyPET resources.'''
    # if using conda put the resources in the folder with the environment name
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env = os.environ['CONDA_DEFAULT_ENV']
        print 'i> conda environment found:', env
    else:
        env = ''
    # create the path for the resources files according to the OS platform
    if platform.system() == 'Linux' :
        path_resources = os.path.join( os.path.join(os.path.expanduser('~'),   '.niftypet'), env )
    elif platform.system() == 'Windows' :
        path_resources = os.path.join( os.path.join(os.getenv('LOCALAPPDATA'), '.niftypet'), env )
    else:
        print 'e> only Linux and Windows operating systems are supported!'
        return None

    return path_resources

# ---------------------------------------------------------------------------------
def find_cuda():
    '''Locate the CUDA environment on the system.'''
    # search the PATH for NVCC
    for fldr in os.environ['PATH'].split(os.pathsep):
        cuda_path = join(fldr, 'nvcc')
        if os.path.exists(cuda_path):
            cuda_path = os.path.dirname(os.path.dirname(cuda_path))
            break
        cuda_path = None
    
    if cuda_path is None:
        print 'w> nvcc compiler could not be found from the PATH!'
        return None

    # serach for the CUDA library path
    lcuda_path = os.path.join(cuda_path, 'lib64')
    if 'LD_LIBRARY_PATH' in os.environ.keys():
        if lcuda_path in os.environ['LD_LIBRARY_PATH'].split(os.pathsep):
            print 'i> found CUDA lib64 in LD_LIBRARY_PATH:   ', lcuda_path
    elif os.path.isdir(lcuda_path):
        print 'i> found CUDA lib64 in :   ', lcuda_path
    else:
        print 'w> folder for CUDA library (64-bit) could not be found!'


    return cuda_path, lcuda_path
# ---------------------------------------------------------------------------------

# =================================================================================
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
            print '----------------------------'
            print 'e> Import Error: NiftyPET''s resources file <resources.py> could not be imported.  It should be in ''~/.niftypet/resources.py'' but likely it does not exists.'
            print '----------------------------'
    else:
        print 'e> resources file not found/installed.'
        return None

    # get all constants and check if device is already chosen
    Cnt = resources.get_setup()
    if 'CCARCH' in Cnt and 'DEVID' in Cnt:
        print 'i> using this CUDA architecture(s):', Cnt['CCARCH']
        return Cnt['CCARCH']

    # get the current locations
    path_current = os.path.dirname( os.path.realpath(__file__) )
    path_resins = os.path.join(path_current, 'resources')
    path_dinf = os.path.join(path_current, 'niftypet')
    path_dinf = os.path.join(path_dinf, 'nimpa')
    path_dinf = os.path.join(path_dinf, 'dinf')
    # temporary installation location for identifying the CUDA devices
    path_tmp_dinf = os.path.join(path_resins,'dinf')
    # if the folder 'path_tmp_dinf' exists, delete it
    if os.path.isdir(path_tmp_dinf):
        shutil.rmtree(path_tmp_dinf)
    # copy the device_info module to the resources folder within the installation package
    shutil.copytree( path_dinf, path_tmp_dinf)
    # create a build using cmake
    if platform.system()=='Windows':
        path_tmp_build = os.path.join(path_tmp_dinf, 'build')
    elif platform.system()=='Linux':
        path_tmp_build = os.path.join(path_tmp_dinf, 'build')
        
    os.makedirs(path_tmp_build)
    os.chdir(path_tmp_build)
    if platform.system()=='Windows':
        subprocess.call(
            ['cmake', '../', '-DPYTHON_INCLUDE_DIRS='+pyhdr,
             '-DPYTHON_PREFIX_PATH='+prefix, '-G', Cnt['MSVC_VRSN']]
        )
        subprocess.call(['cmake', '--build', './', '--config', 'Release'])
        path_tmp_build = os.path.join(path_tmp_build, 'Release')
    elif platform.system()=='Linux':
        subprocess.call(
            ['cmake', '../', '-DPYTHON_INCLUDE_DIRS='+pyhdr,
             '-DPYTHON_PREFIX_PATH='+prefix]
        )
        subprocess.call(['cmake', '--build', './'])
    else:
        print 'e> only Linux and Windows operating systems are supported!'
        return None
    
    # imoprt the new module for device properties
    sys.path.insert(0, path_tmp_build)
    import dinf
    # get the list of installed CUDA devices
    Ldev = dinf.dev_info(0)
    # extract the compute capability as a single number 
    cclist = [int(str(e[2])+str(e[3])) for e in Ldev]
    # get the list of supported CUDA devices (with minimum compute capability)
    spprtd = [str(cc) for cc in cclist if cc>=mincc]
    # best for the default CUDA device
    i = [int(s) for s in spprtd]
    devid = i.index(max(i))
    #-----------------------------------------------------------------------------------
    # form return list of compute capability numbers for which the software will be compiled
    ccstr = ''
    for cc in spprtd:
        ccstr += '-gencode=arch=compute_'+cc+',code=compute_'+cc+';'
    #-----------------------------------------------------------------------------------

    # remove the temporary path
    sys.path.remove(path_tmp_build)
    # delete the build once the info about the GPUs has been obtained
    os.chdir(path_current)
    shutil.rmtree(path_tmp_dinf, ignore_errors=True)

    # passing this setting to resources.py
    fpth = os.path.join(path_resources,'resources.py') #resource_filename(__name__, 'resources/resources.py')
    f = open(fpth, 'r')
    rsrc = f.read()
    f.close()
    # get the region of keeping in synch with Python
    i0 = rsrc.find('### start GPU properties ###')
    i1 = rsrc.find('### end GPU properties ###')
    # list of constants which will be kept in sych from Python
    cnt_list = ['DEV_ID', 'CC_ARCH']
    val_list = [str(devid), '\''+ccstr+'\'']
    # update the resource.py file
    strNew = '### start GPU properties ###\n'
    for i in range(len(cnt_list)):
        strNew += cnt_list[i]+' = '+val_list[i] + '\n'
    rsrcNew = rsrc[:i0] + strNew + rsrc[i1:] 
    f = open(fpth, 'w')
    f.write(rsrcNew)
    f.close()

    return ccstr

#=================================================================================================
def resources_setup():
    '''
    This function checks CUDA devices, selects some and installs resources.py
    '''
    print 'i> installing file <resources.py> into home directory if it does not exist.'
    path_current = os.path.dirname( os.path.realpath(__file__) )
    # path to the install version of resources.py.
    path_install = os.path.join(path_current, 'resources')
    # get the path to the local resources.py (on Linux machines it is in ~/.niftypet)
    path_resources = path_niftypet_local()
    print path_current

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
            print 'e> could not fine file <resources.py> to be installed!'
            raise IOError('could not find <resources.py')
    else:
        print 'i> <resources.py> should be already in the local NiftyPET folder.', path_resources
        # set the flag that the resources file is already there
        flg_resources = True
        sys.path.append(path_resources)
        try:
            import resources
        except ImportError as ie:
            print '----------------------------'
            print 'e> Import Error: NiftyPET''s resources file <resources.py> could not be imported.  It should be in ''~/.niftypet/resources.py'' but likely it does not exists.'
            print '----------------------------'

    # find available GPU devices, select one or more and output the compilation flags
    gpuarch = dev_setup()

    # return gpuarch for cmake compilation
    return gpuarch
#=================================================================================================
