========================================================
NIMPA: Neuro and NiftyPET Image Processing and Analysis
========================================================

NIMPA is a stand-alone and independent package dedicated to high-throughput processing and analysis of brain images, particularly those, which are acquired using positron emission tomography (PET) and magnetic resonance (MR).  Although, it is an essential part of the *NiftyPET* package for seamless PET image reconstruction, NIMPA is equally well suited for independent image processing, including image trimming, upsampling and partial volume correction (PVC).

Trimming is performed in order to reduce the unused image voxels in brain imaging, when using whole body PET scanners, for which only some part of the field of view (FOV) is used.

The upsampling is needed for more accurate extraction (sampling) of PET data using regions of interest (ROI), obtained using parcellation of the corresponding T1w MR image, usually of higher image resolution.

PVC is needed to correct for the spill-in and spill-out of PET signal from defined ROIs (specific for any given application).

In order to facilitate these operations, NIMPA relies on third-party software for image conversion from DICOM to NIfTI (dcm2niix) and image registration (NiftyReg).  The additional software is installed automatically to a user specified location.


Dependencies
------------

NIMPA relies on GPU computing using NVidia's CUDA platform.  The CUDA routines are wrapped in Python C extensions.  The provided software has to be compiled from source (done automatically) for any given Linux flavour (Linux is preferred over Windows) using Cmake.

The following software has to be installed prior to NIMPA installation:

* CUDA (currently the latest is 9.1): https://developer.nvidia.com/cuda-downloads

* Cmake (version 3.xx): https://cmake.org/download/

* Python with the recommended Anaconda distribution: https://www.anaconda.com/download


Installation
------------

To install NIMPA from source for any given CUDA version and operating system (Linux is preferred), simply type:

.. code-block:: bash

  pip install --no-binary :all: --verbose nimpa


Usage
-----

.. code-block:: python

  from niftypet import nimpa



Author: Pawel J. Markiewicz

Copyright 2018