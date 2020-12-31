#!/usr/bin/env python3
"""
Compile CUDA source code and setup Python 3 package 'nimpa'
for namespace 'niftypet'.
"""
import logging
import os
import platform
import sys
from textwrap import dedent

from setuptools import find_packages, setup

from niftypet.ninst import cudasetup as cs
from niftypet.ninst import dinf
from niftypet.ninst import install_tools as tls

__author__ = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__ = "Copyright 2020"
__licence__ = __license__ = "Apache 2.0"

logging.basicConfig(level=logging.INFO, format=tls.LOG_FORMAT)
log = logging.getLogger("nimpa.setup")

tls.check_platform()
ext = tls.check_depends()  # external dependencies

if not ext["git"]:
    raise SystemError(
        dedent(
            """\
            --------------------------------------------------------------
            Git is not installed but is required for tools installation.
            --------------------------------------------------------------"""
        )
    )

cs.resources_setup(gpu=False)  # install resources.py
try:
    gpuarch = cs.dev_setup()  # update resources.py with a supported GPU device
except Exception as exc:
    log.error("could not set up CUDA:\n%s", exc)
    gpuarch = None

# First install third party apps for NiftyPET tools
log.info(
    dedent(
        """\
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
        dedent(
            """\
        --------------------------------------------------------------
        Installing dcm2niix ...
        --------------------------------------------------------------"""
        )
    )
    Cnt = tls.install_tool("dcm2niix", Cnt)
# ---------------------------------------

# -------------------------------------------
# NiftyReg
if not chck_tls["REGPATH"] or not chck_tls["RESPATH"]:
    reply = True
    if not gpuarch:
        try:
            reply = tls.query_yesno(
                "q> the latest compatible version of NiftyReg seems to be missing.\n"
                "   Do you want to install it?"
            )
        except:
            pass

    if reply:
        log.info(
            dedent(
                """\
                --------------------------------------------------------------
                Installing NiftyReg ...
                --------------------------------------------------------------"""
            )
        )
        Cnt = tls.install_tool("niftyreg", Cnt)
log.info(
    dedent(
        """\
        --------------------------------------------------------------
        Installation of NiftyPET-tools is done.
        --------------------------------------------------------------"""
    )
)

cmake_args = [f"-DPython3_ROOT_DIR={sys.prefix}"]
try:
    nvcc_arches = {"{2:d}{3:d}".format(*i) for i in dinf.gpuinfo()}
except Exception as exc:
    log.warning("could not detect CUDA architectures:\n%s", exc)
else:
    cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + " ".join(sorted(nvcc_arches)))
log.info("cmake_args:%s", cmake_args)
setup_kwargs = {
    "version": "2.0.0",
    "packages": find_packages(exclude=["tests"]),
    "package_data": {"niftypet": ["nimpa/auxdata/*"]},
}

if nvcc_arches:
    from skbuild import setup as sksetup
    sksetup(
        cmake_source_dir="niftypet",
        cmake_languages=("C", "CXX", "CUDA"),
        cmake_minimum_required_version="3.18",
        cmake_args=cmake_args,
        **setup_kwargs
    )
else:
    setup(**setup_kwargs)
