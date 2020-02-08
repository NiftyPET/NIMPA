#!/usr/bin/env python
"""
Compile CUDA source code and setup Python 3 package 'nimpa'
for namespace 'niftypet'.
"""
import logging
import os
import platform
from setuptools import setup, find_packages
from subprocess import run, PIPE
import sys

import cudasetup as cs
import install_tools as tls
__author__      = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__   = "Copyright 2020"
__licence__ = __license__ = "Apache 2.0"

logging.basicConfig(level=logging.INFO)
logroot = logging.getLogger('nimpa')
hand = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s:%(funcName)s\n> %(message)s')
hand.setFormatter(formatter)
logroot.addHandler(hand)
log = logging.getLogger('nipet.setup')

if not platform.system() in ['Windows', 'Darwin', 'Linux']:
    log.error('''\
        \rthe operating system is not supported: {}
        \ronly Linux, Windows and macOS are supported.
        '''.format(platform.system()))
    raise SystemError('unknown operating system (OS).')

# check if git, CMake and CUDA are installed
chk = tls.check_depends()

if not chk['git']:
    log.error('''\
        \r--------------------------------------------------------------
        \rGit is not installed but is required for tools installation.
        \r--------------------------------------------------------------
        ''')
    raise SystemError('Git is missing.')

#> check if CUDA and CMake are available to compile C/CUDA C code
if chk['cuda'] and chk['cmake']:
    #----------------------------------------------------
    # select the supported GPU device and install resources.py
    log.info('''
        \r--------------------------------------------------------------
        \rSetting up CUDA ...
        \r--------------------------------------------------------------
        ''')
    gpuarch = cs.resources_setup()
    #----------------------------------------------------
else:
    gpuarch = cs.resources_setup(gpu=False)


#===============================================================
# First install third party apps for NiftyPET tools
log.info('''
    \r--------------------------------------------------------------
    \rSetting up NiftyPET tools ...
    \r--------------------------------------------------------------
    ''')

#get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
if os.path.isfile(os.path.join(path_resources,'resources.py')):
    sys.path.append(path_resources)
    try:
        import resources
    except ImportError as ie:
        log.info('''
            \r------------------------------------------------------------------
            \rNiftyPET resources file <resources.py> could not be imported.
            \rIt should be in ~/.niftypet/resources.py (Linux) or
            \rin //Users//USERNAME//AppData//Local//niftypet//resources.py (Windows)
            \rbut likely it does not exists.
            \r------------------------------------------------------------------
            ''')
        raise SystemError('Missing resource file')
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
        log.info('''
            \r--------------------------------------------------------------
            \rInstalling dcm2niix ...
            \r--------------------------------------------------------------
            ''')
        Cnt = tls.install_tool('dcm2niix', Cnt)
    #---------------------------------------

    #-------------------------------------------
    # NiftyReg
    if not chck_tls['REGPATH'] or not chck_tls['RESPATH']:

        if gpuarch=='':
            try:
                reply = tls.query_yesno('q> the latest compatible version of NiftyReg seems to be missing.\n   Do you want to install it?')
            except:
                reply = True
        else:
            reply = True

        if reply:
            log.info('''
                \r--------------------------------------------------------------
                \rInstalling NiftyReg ...
                \r--------------------------------------------------------------
                ''')
            Cnt = tls.install_tool('niftyreg', Cnt)
    #-------------------------------------------

else:
    raise SystemError('Missing file: resources.py')

log.info('''
    \r--------------------------------------------------------------
    \rInstallation of NiftyPET-tools is done.
    \r--------------------------------------------------------------
    ''')
#===============================================================


#===============================================================
#>CUDA installation
if chk['cuda'] and gpuarch!='':

    log.info('''
        \r--------------------------------------------------------------
        \rCUDA compilation for NIMPA ...
        \r--------------------------------------------------------------
        ''')

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

        p = run(cmd[ci], stdout=PIPE, stderr=PIPE)

        stdout = p.stdout.decode('utf-8')
        stderr = p.stderr.decode('utf-8')

        with open(cmakelog[ci], 'w') as f:
            f.write(stdout)



        ei = stderr.find('error')
        if ei>=0:
            errstr.append(stderr[ei:ei+60]+'...')
        else:
            errstr.append('_')

        if p.stderr:
            log.warning('''\n
            \r---------- process warnings/errors ------------\n
            \r{}
            \r--------------------- end ---------------------
            '''.format(stderr))

        log.info('''\n
        \r---------- compilation output ------------\n
        \r{}
        \r------------------- end ------------------
        '''.format(stdout))


    log.info('\n------------- error report -------------')
    for ci in range(len(cmd)):
        if errstr[ci] != '_':
            log.error(' found error(s) in ' + ' '.join(cmd[ci]) + ' >> ' + errstr[ci])
    log.info('------------------ end -----------------\n')

    # come back from build folder
    os.chdir(path_current)
#===============================================================



#===============================================================
# PYTHON SETUP
#===============================================================
log.info('''found those packages:\n{}'''.format(find_packages(exclude=['docs'])))

freadme = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.rst')
log.info('''\
    \rUsing this README file:
    {}
    '''.format(freadme))

with open(freadme) as file:
    long_description = file.read()

#---- for setup logging -----
stdout = sys.stdout
stderr = sys.stderr
log_file = open('setup_nimpa.log', 'w')
sys.stdout = log_file
sys.stderr = log_file
#----------------------------

if platform.system() in ['Linux', 'Darwin'] :
    fex = '*.so'
elif platform.system() == 'Windows' :
    fex = '*.pyd'
#----------------------------
setup(
    name='nimpa',
    license=__licence__,
    version='2.0.0',
    description='CUDA-accelerated Python utilities for high-throughput PET/MR image processing and analysis.',
    long_description=long_description,
    author=__author__[0],
    author_email='p.markiewicz@ucl.ac.uk',
    url='https://github.com/NiftyPET/NIMPA',
    keywords='PET MR processing analysis',
    install_requires=[
        'nibabel>=2.2.1,<=3.0.1',
        'numpy>=1.14',
        'pydicom>=1.0.2,<=1.3.1',
        'scipy',
        #'SimpleITK>=1.2.0',
        ],
    python_requires='>=3.4',
    packages=find_packages(exclude=['docs']),
    package_data={
        'niftypet': ['auxdata/*'],
        'niftypet.nimpa.dinf': [fex],
        'niftypet.nimpa.prc' : [fex],
    },
    zip_safe=False,
    # namespace_packages=['niftypet'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

    ],
)
#===============================================================
