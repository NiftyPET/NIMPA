=======================================================
NIMPA: Neuro and NiftyPET Image Processing and Analysis
=======================================================

|Docs| |Version| |Downloads| |Py-Versions| |DOI| |Licence| |Tests|

NIMPA is a stand-alone Python sub-package of NiftyPET_, dedicated to high-throughput processing and analysis of brain images, particularly those, which are acquired using positron emission tomography (PET) and magnetic resonance (MR).  Although, it is an essential part of the NiftyPET_ package for seamless PET image reconstruction, NIMPA is equally well suited for independent image processing, including image trimming, upsampling and partial volume correction (PVC).

.. _NiftyPET: https://github.com/NiftyPET/NiftyPET

Trimming is performed in order to reduce the unused image voxels in brain imaging, when using whole body PET scanners, for which only some part of the field of view (FOV) is used.

The upsampling is needed for more accurate extraction (sampling) of PET data using regions of interest (ROI), obtained using parcellation of the corresponding T1w MR image, usually of higher image resolution.

PVC is needed to correct for the spill-in and spill-out of PET signal from defined ROIs (specific for any given application).

**Documentation with installation manual and tutorials**: https://niftypet.readthedocs.io/

Quick Install
~~~~~~~~~~~~~

Note that it's recommended (but not required) to use `conda`.

.. code:: sh

    # cross-platform install
    conda install -c conda-forge python=3 \
      ipykernel numpy scipy scikit-image matplotlib ipywidgets dcm2niix
    pip install "nimpa>=2"

For optional `dcm2niix <https://github.com/rordenlab/dcm2niix>`_ (image conversion from DICOM to NIfTI) and/or `niftyreg <https://github.com/KCL-BMEIS/niftyreg>`_ (image registration) support, simply install them separately (``pip install dcm2niix niftyreg``).

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

|Licence| |DOI|

Copyright 2018-21

- `Pawel J. Markiewicz <https://github.com/pjmark>`__ @ University College London
- `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ University College London/King's College London
- `Contributors <https://github.com/NiftyPET/NIMPA/graphs/contributors>`__

.. |Docs| image:: https://readthedocs.org/projects/niftypet/badge/?version=latest
   :target: https://niftypet.readthedocs.io/en/latest/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4417633.svg
   :target: https://doi.org/10.5281/zenodo.4417633
.. |Licence| image:: https://img.shields.io/pypi/l/nimpa.svg?label=licence
   :target: https://github.com/NiftyPET/NIMPA/blob/master/LICENCE
.. |Tests| image:: https://img.shields.io/github/workflow/status/NiftyPET/NIMPA/Test?logo=GitHub
   :target: https://github.com/NiftyPET/NIMPA/actions
.. |Downloads| image:: https://img.shields.io/pypi/dm/nimpa.svg?logo=pypi&logoColor=white&label=PyPI%20downloads
   :target: https://pypi.org/project/nimpa
.. |Version| image:: https://img.shields.io/pypi/v/nimpa.svg?logo=python&logoColor=white
   :target: https://github.com/NiftyPET/NIMPA/releases
.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/nimpa.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/nimpa
