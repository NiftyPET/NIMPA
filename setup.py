#!/usr/bin/env python3
"""
Compile CUDA source code and setup Python 3 package 'nimpa'
for namespace 'niftypet'.
"""
import logging
import re
import sys
from pathlib import Path
from textwrap import dedent

from setuptools import find_packages, setup
from setuptools_scm import get_version

from niftypet.ninst import cudasetup as cs
from niftypet.ninst import dinf
from niftypet.ninst import install_tools as tls

__version__ = get_version(root=".", relative_to=__file__)

logging.basicConfig(level=logging.INFO, format=tls.LOG_FORMAT)
log = logging.getLogger("nimpa.setup")

tls.check_platform()
ext = tls.check_depends() # external dependencies

if not ext["git"]:
    raise SystemError(
        dedent("""\
            --------------------------------------------------------------
            Git is not installed but is required for tools installation.
            --------------------------------------------------------------"""))

cs.resources_setup(gpu=False) # install resources.py
try:
    gpuarch = cs.dev_setup()  # update resources.py with a supported GPU device
except Exception as exc:
    log.error("could not set up CUDA:\n%s", exc)
    gpuarch = None

# First install third party apps for NiftyPET tools
log.info(
    dedent("""\
        --------------------------------------------------------------
        Setting up NiftyPET tools ...
        --------------------------------------------------------------"""))
# get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
resources = cs.get_resources()
# get the current setup, if any
Cnt = resources.get_setup()
# check the installation of tools
chck_tls = tls.check_version(Cnt, chcklst=["RESPATH", "REGPATH"])

# -------------------------------------------
# NiftyPET tools:
# -------------------------------------------
if "sdist" not in sys.argv or any(i in sys.argv for i in ["build", "bdist", "wheel"]):
    # NiftyReg
    if not chck_tls["REGPATH"] or not chck_tls["RESPATH"]:
        reply = True
        if not gpuarch:
            try:
                reply = tls.query_yesno(
                    "q> the latest compatible version of NiftyReg seems to be missing.\n"
                    "   Do you want to install it?")
            except BaseException:
                pass

        if reply:
            log.info(
                dedent("""\
                    --------------------------------------------------------------
                    Installing NiftyReg ...
                    --------------------------------------------------------------"""))
            Cnt = tls.install_tool("niftyreg", Cnt)
    log.info(
        dedent("""\
            --------------------------------------------------------------
            Installation of NiftyPET-tools is done.
            --------------------------------------------------------------"""))
else:
    log.info(
        dedent("""\
            --------------------------------------------------------------
            Skipping installation of NiftyPET-tools.
            --------------------------------------------------------------"""))

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
setup_kwargs = {
    "use_scm_version": True, "packages": find_packages(exclude=["tests"]),
    "package_data": {"niftypet": ["nimpa/auxdata/*"]}}
cmake_args = [
    f"-DNIMPA_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}",
    f"-DNIMPA_KERNEL_RADIUS={getattr(resources, 'RSZ_PSF_KRNL', 8)}"]

try:
    from skbuild import setup as sksetup
    nvcc_arches = {"{2:d}{3:d}".format(*i) for i in dinf.gpuinfo() if i[2:4] >= (3, 5)}
    if nvcc_arches:
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(sorted(nvcc_arches)))
except Exception as exc:
    log.warning("Import or CUDA device detection error:\n%s", exc)
    setup(**setup_kwargs)
else:
    for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
        i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
    sksetup(cmake_source_dir="niftypet", cmake_languages=("C", "CXX", "CUDA"),
            cmake_minimum_required_version="3.18", cmake_args=cmake_args, **setup_kwargs)
