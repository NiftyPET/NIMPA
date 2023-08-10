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
# some setup kwargs cannot be in `pyproject.toml` since
# `install_requires` is dynamically set depending on CUDA GPU detection
setup_kwargs = {
    "use_scm_version": True, "packages": find_packages(exclude=["tests"]), "package_data": {
        "niftypet": [
            "nimpa/auxdata/*", "nimpa/acr_design/core_mumap/*", "nimpa/acr_design/core_nac/*",
            "nimpa/acr_design/rods/*", "nimpa/acr_design/sampling/*"]}, "install_requires": [
                'dcm2niix', 'dipy>=1.3.0', 'imageio', 'miutil[nii]>=0.10.0', 'nibabel>=2.4.0',
                'ninst>=0.12.0', 'numpy>=1.14', 'pydicom>=1.0.2', 'scipy', 'setuptools', 'spm12',
                'SimpleITK']}
# 'SimpleITK>=1.2.0'
cmake_args = [
    f"-DNIMPA_BUILD_VERSION={build_ver}",
    f"-DNIMPA_KERNEL_RADIUS={getattr(resources, 'RSZ_PSF_KRNL', 8)}"]

try:
    import cuvec as cu
    from miutil import cuinfo
    from skbuild import setup as sksetup
    assert cu.include_path.is_dir()
    try:
        nvcc_arch_raw = map(cuinfo.compute_capability, range(cuinfo.num_devices()))
        nvcc_arches = {"%d%d" % i for i in nvcc_arch_raw if i >= (3, 5)}
        if nvcc_arches:
            cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(sorted(nvcc_arches)))
    except Exception as exc:
        if "sdist" not in sys.argv or any(i in sys.argv for i in ["build", "bdist", "wheel"]):
            log.warning("CUDA device detection error:\n%s", exc)
            log.warning("Compiling for all architectures")
except Exception as exc:
    log.warning("Import or CUDA device detection error:\n%s", exc)
    setup(**setup_kwargs)
else:
    setup_kwargs['install_requires'].extend(["cuvec>=2.3.1", "numcu"])
    for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
        i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
    sksetup(cmake_source_dir="niftypet", cmake_languages=("C", "CXX", "CUDA"),
            cmake_minimum_required_version="3.18", cmake_args=cmake_args, **setup_kwargs)
