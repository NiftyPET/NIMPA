/*----------------------------------------------------------------------
CUDA C extension for Python
Provides fast image-based operations on the GPUs

author: Pawel Markiewicz
Copyrights: 2019
----------------------------------------------------------------------*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include "conv.h"
#include "cuhelpers.h"
#include "nlm.h"
#include "pycuvec.cuh"
#include "rsmpl.h"
#include <Python.h>
#include <stdlib.h>

//=== START PYTHON INIT ===

//--- Available functions
static PyObject *img_resample(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *img_convolve(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *img_nlm(PyObject *self, PyObject *args, PyObject *kwargs);
//---

//> Module Method Table
static PyMethodDef improc_methods[] = {
    {"resample", (PyCFunction)img_resample, METH_VARARGS | METH_KEYWORDS,
     "Does rigid body transformation with very fine sampling."},
    {"convolve", (PyCFunction)img_convolve, METH_VARARGS | METH_KEYWORDS,
     "Fast 3D image convolution with separable kernel."},
    {"nlm", (PyCFunction)img_nlm, METH_VARARGS | METH_KEYWORDS,
     "3D Non-local means (NLM) guided filtering."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef improc_module = {
    PyModuleDef_HEAD_INIT,
    "improc", //> name of module
    //> module documentation, may be NULL
    "This module provides GPU routines for image processing (convolving & resampling).",
    -1, //> the module keeps state in global variables.
    improc_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_improc(void) {

  Py_Initialize();

  return PyModule_Create(&improc_module);
}

static PyObject *img_resample(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *dst = NULL; // output image
  PyCuVec<float> *src = NULL; // original image (to be transformed)
  PyCuVec<float> *A = NULL;   // transformation matrix
  PyObject *o_Cim;            // Dictionary of image constants
  bool MEMSET = true;         // whether to zero `dst` first
  bool SYNC = true;           // whether to ensure deviceToHost copy on return
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"src", "A", "Cnt", "output", "memset", "sync", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|Obb", (char **)kwds, (PyObject **)&src,
                                   (PyObject **)&A, &o_Cim, (PyObject **)&dst, &MEMSET, &SYNC,
                                   &LOG))
    return NULL;
  if (!A || !src) return NULL;

  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF((PyObject *)dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src);
    if (!dst) return NULL;
    MEMSET = false;
  }

  // Structure for constants
  Cimg Cim;
  PyObject *pd_vxsox = PyDict_GetItemString(o_Cim, "VXSOx");
  Cim.VXSOx = (float)PyFloat_AsDouble(pd_vxsox);
  PyObject *pd_vxsoy = PyDict_GetItemString(o_Cim, "VXSOy");
  Cim.VXSOy = (float)PyFloat_AsDouble(pd_vxsoy);
  PyObject *pd_vxsoz = PyDict_GetItemString(o_Cim, "VXSOz");
  Cim.VXSOz = (float)PyFloat_AsDouble(pd_vxsoz);
  PyObject *pd_vxnox = PyDict_GetItemString(o_Cim, "VXNOx");
  Cim.VXNOx = (short)PyLong_AsLong(pd_vxnox);
  PyObject *pd_vxnoy = PyDict_GetItemString(o_Cim, "VXNOy");
  Cim.VXNOy = (short)PyLong_AsLong(pd_vxnoy);
  PyObject *pd_vxnoz = PyDict_GetItemString(o_Cim, "VXNOz");
  Cim.VXNOz = (short)PyLong_AsLong(pd_vxnoz);
  PyObject *pd_offox = PyDict_GetItemString(o_Cim, "OFFOx");
  Cim.OFFOx = (float)PyFloat_AsDouble(pd_offox);
  PyObject *pd_offoy = PyDict_GetItemString(o_Cim, "OFFOy");
  Cim.OFFOy = (float)PyFloat_AsDouble(pd_offoy);
  PyObject *pd_offoz = PyDict_GetItemString(o_Cim, "OFFOz");
  Cim.OFFOz = (float)PyFloat_AsDouble(pd_offoz);

  PyObject *pd_vxsrx = PyDict_GetItemString(o_Cim, "VXSRx");
  Cim.VXSRx = (float)PyFloat_AsDouble(pd_vxsrx);
  PyObject *pd_vxsry = PyDict_GetItemString(o_Cim, "VXSRy");
  Cim.VXSRy = (float)PyFloat_AsDouble(pd_vxsry);
  PyObject *pd_vxsrz = PyDict_GetItemString(o_Cim, "VXSRz");
  Cim.VXSRz = (float)PyFloat_AsDouble(pd_vxsrz);
  PyObject *pd_vxnrx = PyDict_GetItemString(o_Cim, "VXNRx");
  Cim.VXNRx = (short)PyLong_AsLong(pd_vxnrx);
  PyObject *pd_vxnry = PyDict_GetItemString(o_Cim, "VXNRy");
  Cim.VXNRy = (short)PyLong_AsLong(pd_vxnry);
  PyObject *pd_vxnrz = PyDict_GetItemString(o_Cim, "VXNRz");
  Cim.VXNRz = (short)PyLong_AsLong(pd_vxnrz);
  PyObject *pd_offrx = PyDict_GetItemString(o_Cim, "OFFRx");
  Cim.OFFRx = (float)PyFloat_AsDouble(pd_offrx);
  PyObject *pd_offry = PyDict_GetItemString(o_Cim, "OFFRy");
  Cim.OFFRy = (float)PyFloat_AsDouble(pd_offry);
  PyObject *pd_offrz = PyDict_GetItemString(o_Cim, "OFFRz");
  Cim.OFFRz = (float)PyFloat_AsDouble(pd_offrz);

  // for (int i=0; i<12; i++) fprintf(stderr, "A[%d] = %f\n",i,A[i] );

  //=================================================================
  rsmpl(dst->vec.data(), src->vec.data(), A->vec.data(), Cim, MEMSET, SYNC);
  //=================================================================

  fprintf(stderr, "i> new image (x,y,z) = (%d,%d,%d)\n   voxel size: (%6.4f, %6.4f, %6.4f)\n",
          Cim.VXNRx, Cim.VXNRy, Cim.VXNRz, Cim.VXSRx, Cim.VXSRy, Cim.VXSRz);

  // //--- form output tuples
  // PyObject *tuple_out = PyTuple_New(2);
  // PyTuple_SetItem(tuple_out, 0, Py_BuildValue("i", 23));
  // PyTuple_SetItem(tuple_out, 1, PyArray_Return(p_imr));
  // //---

  return (PyObject *)dst;
}

static PyObject *img_convolve(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *src = NULL; // input image
  PyCuVec<float> *knl = NULL; // kernel matrix, for x, y, and z dimensions
  PyCuVec<float> *dst = NULL; // output image
  int DEVID = 0;
  bool MEMSET = true; // whether to zero `dst` first
  bool SYNC = true;   // whether to ensure deviceToHost copy on return
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"img", "knl", "output", "dev_id", "memset", "sync", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Oibbi", (char **)kwds, (PyObject **)&src,
                                   (PyObject **)&knl, (PyObject **)&dst, &DEVID, &MEMSET, &SYNC,
                                   &LOG))
    return NULL;
  if (!src || !knl) return NULL;

  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF((PyObject *)dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src);
    if (!dst) return NULL;
    MEMSET = false;
  }

  if (dst->shape.size() != 3 || src->shape.size() != 3) {
    PyErr_SetString(PyExc_IndexError, "input & output volumes must have ndim == 3");
    return NULL;
  }

  int Nvk = src->shape[0];
  int Nvj = src->shape[1];
  int Nvi = src->shape[2];
  if (LOG <= LOGDEBUG) fprintf(stderr, "d> input image size x,y,z=%d,%d,%d\n", Nvk, Nvj, Nvi);

  int Nkr = (int)knl->shape[1];
  if (LOG <= LOGDEBUG) fprintf(stderr, "d> kernel size [voxels]: %d\n", Nkr);
  if (Nkr != KERNEL_LENGTH || knl->shape.size() != 2 || knl->shape[0] != 3) {
    PyErr_SetString(PyExc_IndexError, "wrong kernel size");
    return NULL;
  }

  if (LOG <= LOGDEBUG) fprintf(stderr, "d> using device: %d\n", DEVID);
  if (!HANDLE_PyErr(cudaSetDevice(DEVID))) return NULL;

  //=================================================================
  setConvolutionKernel(knl->vec.data(), false);
  if (!HANDLE_PyErr(cudaGetLastError())) return NULL;

  d_conv(dst->vec.data(), src->vec.data(), Nvk, Nvj, Nvi, MEMSET, SYNC, LOG);
  if (!HANDLE_PyErr(cudaGetLastError())) return NULL;
  //=================================================================

  return (PyObject *)dst;
}

static PyObject *img_nlm(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *src = NULL; // input image
  PyCuVec<float> *ref = NULL; // guidance image
  PyCuVec<float> *dst = NULL; // output image
  float sigma = 1;
  int half_width = 4;
  int DEVID = 0;
  bool SYNC = true; // whether to ensure deviceToHost copy on return
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"img",    "ref",  "output", "sigma", "half_width",
                               "dev_id", "sync", "log",    NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Ofiibi", (char **)kwds, (PyObject **)&src,
                                   (PyObject **)&ref, (PyObject **)&dst, &sigma, &half_width,
                                   &DEVID, &SYNC, &LOG))
    return NULL;
  if (!src || !ref || sigma < 0 || half_width < 0) return NULL;

  if (LOG <= LOGDEBUG) fprintf(stderr, "d> using device: %d\n", DEVID);
  if (!HANDLE_PyErr(cudaSetDevice(DEVID))) return NULL;

  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF((PyObject *)dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src);
    if (!dst) return NULL;
  }

  if (dst->shape.size() != 3 || src->shape.size() != 3 || ref->shape.size() != 3) {
    PyErr_SetString(PyExc_IndexError, "input & output volumes must have ndim == 3");
    return NULL;
  }

  int Z = src->shape[0];
  int Y = src->shape[1];
  int X = src->shape[2];
  if (LOG <= LOGDEBUG) fprintf(stderr, "d> input image size z,y,x=%d,%d,%d\n", Z, Y, X);

  d_nlm3d(dst->vec.data(), src->vec.data(), ref->vec.data(), sigma, Z, Y, X, half_width, SYNC);
  if (!HANDLE_PyErr(cudaGetLastError())) return NULL;

  return (PyObject *)dst;
}
