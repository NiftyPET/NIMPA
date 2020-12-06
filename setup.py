#!/usr/bin/env python3
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
from textwrap import dedent

from niftypet.ninst import cudasetup as cs
from niftypet.ninst import install_tools as tls
__author__ = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__ = "Copyright 2020"
__licence__ = __license__ = "Apache 2.0"

logging.basicConfig(level=logging.INFO)
logroot = logging.getLogger('nimpa')
logroot.addHandler(tls.LogHandler())
log = logging.getLogger('nimpa.setup')

tls.check_platform()
ext = tls.check_depends()  # external dependencies

if not ext["git"]:
    log.error(
        dedent(
            """
            --------------------------------------------------------------
            Git is not installed but is required for tools installation.
            --------------------------------------------------------------"""
        )
    )
    raise SystemError("Git is missing.")

# install resources.py
if ext["cuda"] and ext["cmake"]:
    # select the supported GPU device and
    gpuarch = cs.resources_setup()
else:
    gpuarch = cs.resources_setup(gpu=False)


# First install third party apps for NiftyPET tools
log.info(
    dedent(
        """
        --------------------------------------------------------------
        Setting up NiftyPET tools ...
        --------------------------------------------------------------"""
    )
)

# get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
resources = cs.get_resources()
# get the current setup, if any
Cnt = resources.get_setup()
# check the installation of tools
chck_tls = tls.check_version(Cnt, chcklst=["RESPATH", "REGPATH", "DCM2NIIX"])

# -------------------------------------------
# NiftyPET tools:
# -------------------------------------------
# DCM2NIIX
if not chck_tls["DCM2NIIX"]:
    # reply = tls.query_yesno('q> the latest compatible version of dcm2niix seems to be missing.\n   Do you want to install it?')
    # if reply:
    log.info(
        dedent("""
        --------------------------------------------------------------
        Installing dcm2niix ...
        --------------------------------------------------------------""")
    )
    Cnt = tls.install_tool("dcm2niix", Cnt)
# ---------------------------------------

# -------------------------------------------
# NiftyReg
if not chck_tls["REGPATH"] or not chck_tls["RESPATH"]:
    if gpuarch == "":
        try:
            reply = tls.query_yesno(
                "q> the latest compatible version of NiftyReg seems to be missing.\n   Do you want to install it?"
            )
        except:
            reply = True
    else:
        reply = True

    if reply:
        log.info(dedent(
            """
            --------------------------------------------------------------
            Installing NiftyReg ...
            --------------------------------------------------------------"""
        ))
        Cnt = tls.install_tool("niftyreg", Cnt)
# -------------------------------------------

log.info(
    dedent(
        """
        --------------------------------------------------------------
        Installation of NiftyPET-tools is done.
        --------------------------------------------------------------"""
    )
)


# CUDA installation
if ext["cuda"] and gpuarch != "":
    log.info(
        dedent(
            """
            --------------------------------------------------------------
            CUDA compilation for NIMPA ...
            --------------------------------------------------------------"""
        )
    )

    path_current = os.path.dirname(os.path.realpath(__file__))
    path_build = os.path.join(path_current, "build")
    if not os.path.isdir(path_build):
        os.makedirs(path_build)
    os.chdir(path_build)

    # cmake installation commands
    cmds = [
        [
            "cmake",
            os.path.join("..", "niftypet"),
            "-DPYTHON_INCLUDE_DIRS=" + cs.pyhdr,
            "-DPYTHON_PREFIX_PATH=" + cs.prefix,
            "-DCUDA_NVCC_FLAGS=" + gpuarch,
        ],
        ["cmake", "--build", "./"]
    ]

    if platform.system() == "Windows":
        cmds[0] += ["-G", Cnt["MSVC_VRSN"]]
        cmds[1] += ["--config", "Release"]

    # error string for later reporting
    errs = []
    # the log files the cmake results are written
    cmakelogs = ["py_cmake_config.log", "py_cmake_build.log"]
    # run commands with logging
    for cmd, cmakelog in zip(cmds, cmakelogs):
        p = run(cmd, stdout=PIPE, stderr=PIPE)
        stdout = p.stdout.decode("utf-8")
        stderr = p.stderr.decode("utf-8")

        with open(cmakelog, "w") as fd:
            fd.write(stdout)

        ei = stderr.find("error")
        if ei >= 0:
            errs.append(stderr[ei : ei + 60] + "...")
        else:
            errs.append("_")

        if p.stderr:
            log.warning(
                dedent(
                    """
                    ---------- process warnings/errors ------------
                    {}
                    --------------------- end ---------------------"""
                ).format(
                    stderr
                )
            )

        log.info(
            dedent(
                """
                ---------- compilation output ------------
                {}
                ------------------- end ------------------"""
            ).format(stdout)
        )

    log.info("\n------------- error report -------------")
    for cmd, err in zip(cmds, errs):
        if err != "_":
            log.error(" found error(s) in %s >> %s", " ".join(cmd), err)
    log.info("------------------ end -----------------")

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
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs']),
    package_data={
        'niftypet': ['auxdata/*'],
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
