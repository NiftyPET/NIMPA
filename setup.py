#!/usr/bin/env python3
"""
Compile CUDA source code and setup Python 3 package 'nimpa'
for namespace 'niftypet'.
"""
import logging
import re
import sys
from pathlib import Path

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

cs.resources_setup(gpu=False) # install resources.py
try:
    cs.dev_setup()            # update resources.py with a supported GPU device
except Exception as exc:
    log.error("could not set up CUDA:\n%s", exc)

# get the local path to NiftyPET resources.py
path_resources = cs.path_niftypet_local()
# if exists, import the resources and get the constants
resources = cs.get_resources()
# get the current setup, if any
Cnt = resources.get_setup()

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
setup_kwargs = {
    "use_scm_version": True, "packages": find_packages(exclude=["tests"]),
    "package_data": {"niftypet": ["nimpa/auxdata/*"]}, "install_requires": [
        'dipy>=1.3.0', 'miutil[nii]>=0.10.0', 'nibabel>=2.4.0', 'ninst>=0.12.0', 'numpy>=1.14',
        'pydicom>=1.0.2', 'scipy', 'setuptools', 'spm12']}
# 'SimpleITK>=1.2.0'
cmake_args = [
    f"-DNIMPA_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}",
    f"-DNIMPA_KERNEL_RADIUS={getattr(resources, 'RSZ_PSF_KRNL', 8)}"]

try:
    import cuvec as cu
    from skbuild import setup as sksetup
    assert cu.include_path.is_dir()
    nvcc_arches = {"{2:d}{3:d}".format(*i) for i in dinf.gpuinfo() if i[2:4] >= (3, 5)}
    if nvcc_arches:
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(sorted(nvcc_arches)))
except Exception as exc:
    log.warning("Import or CUDA device detection error:\n%s", exc)
    setup(**setup_kwargs)
else:
    setup_kwargs['install_requires'].extend(["cuvec>=2.3.1", "numcu"])
    for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
        i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
    sksetup(cmake_source_dir="niftypet", cmake_languages=("C", "CXX", "CUDA"),
            cmake_minimum_required_version="3.18", cmake_args=cmake_args, **setup_kwargs)
