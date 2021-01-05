=======================================================
NIMPA: Neuro and NiftyPET Image Processing and Analysis
=======================================================

|Docs| |PyPI-Status| |PyPI-Downloads|

NIMPA is a stand-alone Python sub-package of NiftyPET_, dedicated to high-throughput processing and analysis of brain images, particularly those, which are acquired using positron emission tomography (PET) and magnetic resonance (MR).  Although, it is an essential part of the NiftyPET_ package for seamless PET image reconstruction, NIMPA is equally well suited for independent image processing, including image trimming, upsampling and partial volume correction (PVC).

.. _NiftyPET: https://github.com/NiftyPET/NiftyPET

Trimming is performed in order to reduce the unused image voxels in brain imaging, when using whole body PET scanners, for which only some part of the field of view (FOV) is used.

The upsampling is needed for more accurate extraction (sampling) of PET data using regions of interest (ROI), obtained using parcellation of the corresponding T1w MR image, usually of higher image resolution.

PVC is needed to correct for the spill-in and spill-out of PET signal from defined ROIs (specific for any given application).

In order to facilitate these operations, NIMPA relies on third-party software for image conversion from DICOM to NIfTI (dcm2niix) and image registration (NiftyReg).  The additional software is installed automatically to a user specified location.

**Documentation with installation manual and tutorials**: https://niftypet.readthedocs.io/

Quick Install
~~~~~~~~~~~~~

Note that installation prompts for setting the path to ``NiftyPET_tools``.
This can be avoided by setting the environment variables ``PATHTOOLS``.

.. code:: sh

    # optional (Linux syntax) to avoid prompts
    export PATHTOOLS=$HOME/NiftyPET_tools
    # cross-platform install
    conda install -c conda-forge python=3 \
      ipykernel numpy scipy scikit-image matplotlib ipywidgets
    pip install --verbose "git+https://github.com/NiftyPET/NIMPA@dev2#egg=nimpa"

External CMake Projects
~~~~~~~~~~~~~~~~~~~~~~~

The raw C/CUDA libraries may be included in external projects using ``cmake``.
Simply build the project and use ``find_package(NiftyPETnimpa)``.

.. code:: sh

    # print installation directory (after `pip install nimpa`)...
    python -c "from niftypet.nimpa import cmake_prefix; print(cmake_prefix)"

    # ... or build & install directly with cmake
    mkdir build && cd build
    cmake ../niftypet && cmake --build . && cmake --install . --prefix /my/install/dir

At this point any external project may include NIMPA as follows
(Once setting ``-DCMAKE_PREFIX_DIR=<installation prefix from above>``):

.. code:: cmake

    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
    project(myproj)
    find_package(NiftyPETnimpa COMPONENTS improc REQUIRED)
    add_executable(myexe ...)
    target_link_libraries(myexe PRIVATE NiftyPET::improc)

Licence
~~~~~~~

|Licence|

- Author: `Pawel J. Markiewicz <https://github.com/pjmark>`__ @ University College London
- `Contributors <https://github.com/NiftyPET/NIMPA/graphs/contributors>`__:

  - `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ King's College London

Copyright 2018-20

.. |Docs| image:: https://readthedocs.org/projects/niftypet/badge/?version=latest
   :target: https://niftypet.readthedocs.io/en/latest/?badge=latest
.. |Licence| image:: https://img.shields.io/pypi/l/nimpa.svg?label=licence
   :target: https://github.com/NiftyPET/NIMPA/blob/master/LICENCE
.. |PyPI-Downloads| image:: https://img.shields.io/pypi/dm/nimpa.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/nimpa
.. |PyPI-Status| image:: https://img.shields.io/pypi/v/nimpa.svg?label=latest
   :target: https://pypi.org/project/nimpa
