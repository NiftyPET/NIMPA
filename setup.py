#!/usr/bin/env python3
"""
Compile CUDA source code and setup Python 3 package 'nimpa'
for namespace 'niftypet'.
"""
import logging
import os
import platform
from setuptools import setup, find_packages
import sys
from textwrap import dedent

from niftypet.ninst import cudasetup as cs
from niftypet.ninst import install_tools as tls

__author__ = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
__copyright__ = "Copyright 2020"
__licence__ = __license__ = "Apache 2.0"

logging.basicConfig(level=logging.INFO, format=tls.LOG_FORMAT)
log = logging.getLogger("nimpa.setup")

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
gpuarch = cs.resources_setup(gpu=ext["cmake"])

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
        dedent(
            """
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
    if gpuarch == "":
        try:
            reply = tls.query_yesno(
                "q> the latest compatible version of NiftyReg seems to be missing.\n   Do you want to install it?"
            )
        except:
            pass

    if reply:
        log.info(
            dedent(
                """
            --------------------------------------------------------------
            Installing NiftyReg ...
            --------------------------------------------------------------"""
            )
        )
        Cnt = tls.install_tool("niftyreg", Cnt)
log.info(
    dedent(
        """
        --------------------------------------------------------------
        Installation of NiftyPET-tools is done.
        --------------------------------------------------------------"""
    )
)

# ===============================================================
# CUDA BUILD
# ===============================================================
if gpuarch != "":
    path_current = os.path.dirname(os.path.realpath(__file__))
    path_build = os.path.join(path_current, "build")
    path_source = os.path.join(path_current, "niftypet")
    cs.cmake_cuda(
        path_source,
        path_build,
        gpuarch,
        logfile_prefix="nimpa_",
        msvc_version=Cnt["MSVC_VRSN"],
    )

# ===============================================================
# PYTHON SETUP
# ===============================================================
log.info("""found those packages:\n{}""".format(find_packages(exclude=["docs"])))

# ---- for setup logging -----
stdout = sys.stdout
stderr = sys.stderr
log_file = open("setup_nimpa.log", "w")
sys.stdout = log_file
sys.stderr = log_file
# ----------------------------

if platform.system() in ["Linux", "Darwin"]:
    fex = "*.so"
elif platform.system() == "Windows":
    fex = "*.pyd"
# ----------------------------
setup(
    version="2.0.0",
    package_data={
        "niftypet": ["auxdata/*"],
        "niftypet.nimpa.prc": [fex],
    },
)
