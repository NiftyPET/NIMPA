========================================================
NIMPA: NeuroImage Processing and Analysis Python package
========================================================

Author: Pawel Markiewicz
Copyright 2018

NIMPA is an stand-alone and independent package dedicated to the high-throughput processing and analysis of brain images, particularly those which are acquired using positron emission tomography (PET).  It is part of NiftyPET and it is essential for seamless PET image reconstruction using NiftyPET.

Currently the package offers trimming and upsampling of brain images.  Trimming is performed in order to reduce the wasted image voxel in brain imaging using whole body PET scanners, where only some part of the field of view (FOV) is used.

The upsampling is needed for more accurate extraction (sampling) of PET data using regions of interest (ROI) obtained based on parcellation of the corresponding T1w MR image.

In order to facilitate these operations, this package relies on third-party software for image conversion from DICOM to NIfTI (dcm2niix) and image registration (NiftyReg).  The additional software is installed automatically to a user specified location.


Dependencies
------------

This package relies on GPU computing using NVidia's CUDA platform.  The CUDA routines are wrapped in Python C extensions.  The provided software has to be compiled from source (done automatically) for any given Linux flavour (Linux is preferred over Windows) using Cmake.

The following software has to be installed prior to NIMPA installation:

* CUDA (Currently the latest is 9.1): https://developer.nvidia.com/cuda-downloads

* Cmake (version 3.xx): https://cmake.org/download/

* Python, recommended is Anaconda: https://www.anaconda.com/download


Installation
------------

To install the package from source for the given CUDA version and operating system (OS), simply type:

.. code-block:: bash

	pip install nimpa