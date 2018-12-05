#!/usr/bin/env python
"""Compile CUDA source code and setup Python package 'nimpa' for package 'niftypet'."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"
# ---------------------------------------------------------------------------------

from setuptools import setup, find_packages

import os
import sys
import platform
from subprocess import call, Popen, PIPE

import cudasetup as cs
import install_tools as tls

# check if git and cmake are installed
chk = tls.check_depends()
if not chk['cmake'] or not chk['git']:
    print '-----------------'
    print 'e> [cmake] and/or [git] are not installed but are required.'
    print '-----------------'
    sys.exit()

if not 'Windows' in platform.system() and not 'Linux' in platform.system():
    print 'e> the operating system is not supported:', platform.system()
    raise SystemError('Unknown Sysytem.')


if chk['cuda']:
    #----------------------------------------------------
    # select the supported GPU device and install resources.py
    print ' '
    print '---------------------------------------------'
    print 'i> setting up CUDA ...'
    gpuarch = cs.resources_setup()
    #----------------------------------------------------
else:
    gpuarch = ''



#===============================================================
# First install third party apps for NiftyPET tools
print ' '
print '---------------------------------------------'
print 'i> setting up NiftyPET tools ...'

#get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
if os.path.isfile(os.path.join(path_resources,'resources.py')):
    sys.path.append(path_resources)
    try:
        import resources
    except ImportError as ie:
        print '---------------------------------------------------------------------------------'
        print 'e> Import Error: NiftyPET''s resources file <resources.py> could not be imported.'
        print '---------------------------------------------------------------------------------'
        raise SystemError('Missing resources file')
    # get the current setup, if any
    Cnt = resources.get_setup()
    # check the installation of tools
    chck_tls = tls.check_version(Cnt, chcklst=['RESPATH','REGPATH','DCM2NIIX'])
    
    #-------------------------------------------
    # NiftyPET tools:
    #-------------------------------------------
    # DCM2NIIX
    if not chck_tls['DCM2NIIX']:
        # reply = tls.query_yesno('q> the latest compatible version of dcm2niix seems to be missing.\n   Do you want to install it?')
        # if reply:
        #-------- Install dmc2niix -------------
        print '---------------------------------------------'
        print 'i> installing dcm2niix:'
        Cnt = tls.install_tool('dcm2niix', Cnt)
    #---------------------------------------
    
    #-------------------------------------------
    # NiftyReg
    if not chck_tls['REGPATH'] or not chck_tls['RESPATH']:
        # reply = tls.query_yesno('q> the latest compatible version of NiftyReg seems to be missing.\n   Do you want to install it?')
        # if reply:
        #-------- Install NiftyReg -------------
        print '---------------------------------------------'
        print 'i> installing NiftyReg:'
        Cnt = tls.install_tool('niftyreg', Cnt)
    #-------------------------------------------
    
else:
    raise SystemError('Missing file: resources.py')

print '---------------------------------------'
print 'i> installation of NiftyPET-tools done.'
print '---------------------------------------'
#===============================================================

if chk['cuda']:
    #===============================================================
    print '---------------------------------'
    print 'i> CUDA compilation for NIMPA ...'
    print '---------------------------------'

    path_current = os.path.dirname( os.path.realpath(__file__) )
    path_build = os.path.join(path_current, 'build')
    if not os.path.isdir(path_build): os.makedirs(path_build)
    os.chdir(path_build)

    # cmake installation commands
    cmd = []
    cmd.append([
        'cmake',
        os.path.join('..','niftypet'),
        '-DPYTHON_INCLUDE_DIRS='+cs.pyhdr,
        '-DPYTHON_PREFIX_PATH='+cs.prefix,
        '-DCUDA_NVCC_FLAGS='+gpuarch
    ])
    cmd.append(['cmake', '--build', './'])

    if platform.system()=='Windows':
        cmd[0] += ['-G', Cnt['MSVC_VRSN']]
        cmd[1] += ['--config', 'Release']

    # error string for later reporting
    errstr = []
    # the log files the cmake results are written
    cmakelog = ['py_cmake_config.log', 'py_cmake_build.log'] 
    # run commands with logging
    for ci in range(len(cmd)):
        with open(cmakelog[ci], 'w') as f:
            p = Popen(cmd[ci], stdout=PIPE, stderr=PIPE)
            for c in iter(lambda: p.stdout.read(1), ''):
                sys.stdout.write(c)
                f.write(c)
        # get the pipes outputs
        stdout, stderr = p.communicate()
        ei = stderr.find('error')
        if ei>=0:
            errstr.append(stderr[ei:ei+40]+'...')
        else:
            errstr.append('_')

        if stderr:
            print 'c>-------- reports -----------'
            print stderr+'c>------------ end ---------------'

        print ' '
        print stdout


    print ' '
    print '--- error report ---'
    for ci in range(len(cmd)):
        if errstr[ci] != '_':
            print 'e> found error(s) in ', ' '.join(cmd[ci]), '>>', errstr[ci]
            print ' '
    print '--- end ---'

    # come back from build folder
    os.chdir(path_current)
    #===============================================================



#===============================================================
# PYTHON SETUP
#===============================================================

print 'i> found those packages:'
print find_packages(exclude=['docs'])

with open('README.rst') as file:
    long_description = file.read()

#---- for setup logging -----
stdout = sys.stdout
stderr = sys.stderr
log_file = open('setup_nimpa.log', 'w')
sys.stdout = log_file
sys.stderr = log_file
#----------------------------

if platform.system() == 'Linux' :
    fex = '*.so'
elif platform.system() == 'Windows' : 
    fex = '*.pyd'
#----------------------------
setup(
    name='nimpa',
    license = 'Apache 2.0',
    version='1.1.1',
    description='CUDA-accelerated Python utilities for high-throughput PET/MR image processing and analysis.',
    long_description=long_description,
    author='Pawel J. Markiewicz',
    author_email='p.markiewicz@ucl.ac.uk',
    url='https://github.com/pjmark/NIMPA',
    keywords='PET MR processing analysis',
    install_requires=['nibabel'],
    packages=find_packages(exclude=['docs']),
    package_data={
        'niftypet': ['auxdata/*'],
        'niftypet.nimpa.dinf': [fex],
        'niftypet.nimpa.prc' : [fex],
    },
    zip_safe=False,
    # namespace_packages=['niftypet'],
    # classifiers=[
    #     'Development Status :: 5 - Production/Stable',
    #     'Intended Audience :: Science/Research',
    #     'Intended Audience :: Healthcare Industry'
    #     'Programming Language :: Python :: 2.7',
    #     'License :: OSI Approved :: Apache Software License',
    #     'Operating System :: POSIX :: Linux',
    #     'Programming Language :: C',
    #     'Topic :: Scientific/Engineering :: Medical Science Apps.'
    # ],
)
#===============================================================
