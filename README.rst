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
    conda create -n niftypet -c conda-forge python=2.7 \
      ipykernel matplotlib numpy scikit-image ipywidgets
    git clone https://github.com/NiftyPET/NIMPA.git nimpa
    conda activate niftypet
    pip install --no-binary :all: --verbose -e ./nimpa

Licence
~~~~~~~

|Licence|

- Author: `Pawel J. Markiewicz <https://github.com/pjmark>`__ @ University College London
- `Contributors <https://github.com/NiftyPET/NIMPA/graphs/contributors>`__:

  - `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ King's College London

Copyright 2018-19

.. |Docs| image:: https://readthedocs.org/projects/niftypet/badge/?version=latest
   :target: https://niftypet.readthedocs.io/en/latest/?badge=latest
.. |Licence| image:: https://img.shields.io/pypi/l/nimpa.svg?label=licence
   :target: https://github.com/NiftyPET/NIMPA/blob/master/LICENCE
.. |PyPI-Downloads| image:: https://img.shields.io/pypi/dm/nimpa.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/nimpa
.. |PyPI-Status| image:: https://img.shields.io/pypi/v/nimpa.svg?label=latest
   :target: https://pypi.org/project/nimpa
