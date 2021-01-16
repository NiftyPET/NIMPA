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
#include "pycuvec.cuh"
#include "rsmpl.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

//=== START PYTHON INIT ===

//--- Available functions
static PyObject *img_resample(PyObject *self, PyObject *args);
static PyObject *img_convolve(PyObject *self, PyObject *args, PyObject *kwargs);
//---

//> Module Method Table
static PyMethodDef improc_methods[] = {
    {"resample", img_resample, METH_VARARGS,
     "Does rigid body transformation with very fine sampling."},
    {"convolve", (PyCFunction)img_convolve, METH_VARARGS | METH_KEYWORDS,
     "Fast 3D image convolution with separable kernel."},
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

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&improc_module);
}

//======================================================================================
// P R O C E S I N G   I M A G E   D A T A
//--------------------------------------------------------------------------------------

static PyObject *img_resample(PyObject *self, PyObject *args) {
  // transformation matrix
  PyObject *o_A;
  // Structure for constants
  Cimg Cim;
  // Dictionary of image constants
  PyObject *o_Cim;
  // original image (to be transformed)
  PyObject *o_imo;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOO", &o_imo, &o_A, &o_Cim)) return NULL;

  // the dictionary of constants
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

  PyArrayObject *p_A = NULL;
  p_A = (PyArrayObject *)PyArray_FROM_OTF(o_A, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_imo = NULL;
  p_imo = (PyArrayObject *)PyArray_FROM_OTF(o_imo, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  /* If that didn't work, throw an exception. */
  if (p_A == NULL || p_imo == NULL) {
    Py_XDECREF(p_A);
    Py_XDECREF(p_imo);
    return NULL;
  }

  float *A = (float *)PyArray_DATA(p_A);
  float *imo = (float *)PyArray_DATA(p_imo);

  // for (int i=0; i<12; i++) printf("A[%d] = %f\n",i,A[i] );

  //=================================================================
  float *imr = rsmpl(imo, A, Cim);
  //=================================================================

  printf("i> new image (x,y,z) = (%d,%d,%d)\n   voxel size: (%6.4f, %6.4f, %6.4f)\n", Cim.VXNRx,
         Cim.VXNRy, Cim.VXNRz, Cim.VXSRx, Cim.VXSRy, Cim.VXSRz);

  npy_intp dims[3];
  dims[2] = Cim.VXNRx;
  dims[1] = Cim.VXNRy;
  dims[0] = Cim.VXNRz;
  PyArrayObject *p_imr = (PyArrayObject *)PyArray_SimpleNewFromData(3, dims, NPY_FLOAT32, imr);

  // //--- form output tuples
  // PyObject *tuple_out = PyTuple_New(2);
  // PyTuple_SetItem(tuple_out, 0, Py_BuildValue("i", 23));
  // PyTuple_SetItem(tuple_out, 1, PyArray_Return(p_imr));
  // //---

  // Clean up:
  Py_DECREF(p_A);
  Py_DECREF(p_imo);

  return PyArray_Return(p_imr);
}

//======================================================================================
// I M A G E    C O N V O L U T I O N
//--------------------------------------------------------------------------------------

static PyObject *img_convolve(PyObject *self, PyObject *args, PyObject *kwargs) {
  int DEVID = 0;
  bool MEMSET = true; // whether to zero `dst` first
  bool SYNC = true;   // whether to ensure deviceToHost copy on return
  int LOG = LOGDEBUG;
  PyObject *o_krnl; // kernel matrix, for x, y, and z dimensions
  PyObject *o_imi;  // input image
  PyObject *o_imo;  // output image

  // Parse the input tuple
  static const char *kwds[] = {"dst", "src", "knl", "dev_id", "memset", "sync", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ibbi", (char **)kwds, &o_imo, &o_imi,
                                   &o_krnl, &DEVID, &MEMSET, &SYNC, &LOG))
    return NULL;
  if (!o_imo || !o_imi || !o_krnl) return NULL;

  PyCuVec<float> *p_imo = (PyCuVec<float> *)o_imo;
  PyCuVec<float> *p_imi = (PyCuVec<float> *)o_imi;
  PyCuVec<float> *p_krnl = (PyCuVec<float> *)o_krnl;

  if (p_imo->shape.size() != 3 || p_imi->shape.size() != 3) {
    PyErr_SetString(PyExc_IndexError, "input & output volumes must have ndim == 3");
    return NULL;
  };

  float *imo = p_imo->vec.data();
  float *imi = p_imi->vec.data();
  float *krnl = p_krnl->vec.data();

  int Nvk = p_imi->shape[0];
  int Nvj = p_imi->shape[1];
  int Nvi = p_imi->shape[2];
  if (LOG <= LOGDEBUG) printf("d> input image size x,y,z=%d,%d,%d\n", Nvk, Nvj, Nvi);

  int Nkr = (int)p_krnl->shape[1];
  if (LOG <= LOGDEBUG) printf("d> kernel size [voxels]: %d\n", Nkr);
  if (Nkr != KERNEL_LENGTH || p_krnl->shape[0] != 3) {
    PyErr_SetString(PyExc_IndexError, "wrong kernel size");
    return NULL;
  }

  if (LOG <= LOGDEBUG) printf("d> using device: %d\n", DEVID);
  if (!HANDLE_PyErr(cudaSetDevice(DEVID))) return NULL;

  //=================================================================
  setConvolutionKernel(krnl, false);
  if (!HANDLE_PyErr(cudaGetLastError())) return NULL;

  gpu_cnv(imo, imi, Nvk, Nvj, Nvi, MEMSET, SYNC);
  if (!HANDLE_PyErr(cudaGetLastError())) return NULL;
  //=================================================================

  Py_INCREF(Py_None);
  return Py_None;
}
