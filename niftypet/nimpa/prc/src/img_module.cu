#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "rsmpl.h"
#include "conv.h"


//=== PYTHON STUFF ===

//--- Docstrings
static char module_docstring[] =
"This module provides GPU routines for (mostly PET) image processing.";
static char rsmpl_docstring[] =
"Does rigid body transformation with very fine sampling.";
static char conv_docstring[] =
"Fast 3D image convolution with separable kernel.";
//---

//--- Available functions
static PyObject *img_resample(PyObject *self, PyObject *args);
static PyObject *img_convolve(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
	{ "resample", img_resample,   METH_VARARGS, rsmpl_docstring },
	{ "convolve", img_convolve,   METH_VARARGS, conv_docstring },
	{ NULL, NULL, 0, NULL }
};
//---

//--- Initialize the module
PyMODINIT_FUNC initimproc(void)  //it HAS to be init______ and then the name of the shared lib.
{
	PyObject *m = Py_InitModule3("improc", module_methods, module_docstring);
	if (m == NULL)
		return;

	/* Load NumPy functionality. */
	import_array();
}
//=======================


//======================================================================================
// P R O C E S I N G   I M A G E   D A T A
//--------------------------------------------------------------------------------------

static PyObject *img_resample(PyObject *self, PyObject *args)
{
	// transformation matrix
	PyObject * o_A;
	// Structure for constants
	Cimg Cim;
	//Dictionary of image constants
	PyObject * o_Cim;
	//original image (to be transformed)
	PyObject * o_imo;

	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOO", &o_imo, &o_A, &o_Cim))
		return NULL;

	//the dictionary of constants
	PyObject* pd_vxsox = PyDict_GetItemString(o_Cim, "VXSOx");
	Cim.VXSOx = (float)PyFloat_AsDouble(pd_vxsox);
	PyObject* pd_vxsoy = PyDict_GetItemString(o_Cim, "VXSOy");
	Cim.VXSOy = (float)PyFloat_AsDouble(pd_vxsoy);
	PyObject* pd_vxsoz = PyDict_GetItemString(o_Cim, "VXSOz");
	Cim.VXSOz = (float)PyFloat_AsDouble(pd_vxsoz);
	PyObject* pd_vxnox = PyDict_GetItemString(o_Cim, "VXNOx");
	Cim.VXNOx = (short)PyInt_AS_LONG(pd_vxnox);
	PyObject* pd_vxnoy = PyDict_GetItemString(o_Cim, "VXNOy");
	Cim.VXNOy = (short)PyInt_AS_LONG(pd_vxnoy);
	PyObject* pd_vxnoz = PyDict_GetItemString(o_Cim, "VXNOz");
	Cim.VXNOz = (short)PyInt_AS_LONG(pd_vxnoz);
	PyObject* pd_offox = PyDict_GetItemString(o_Cim, "OFFOx");
	Cim.OFFOx = (float)PyFloat_AsDouble(pd_offox);
	PyObject* pd_offoy = PyDict_GetItemString(o_Cim, "OFFOy");
	Cim.OFFOy = (float)PyFloat_AsDouble(pd_offoy);
	PyObject* pd_offoz = PyDict_GetItemString(o_Cim, "OFFOz");
	Cim.OFFOz = (float)PyFloat_AsDouble(pd_offoz);

	PyObject* pd_vxsrx = PyDict_GetItemString(o_Cim, "VXSRx");
	Cim.VXSRx = (float)PyFloat_AsDouble(pd_vxsrx);
	PyObject* pd_vxsry = PyDict_GetItemString(o_Cim, "VXSRy");
	Cim.VXSRy = (float)PyFloat_AsDouble(pd_vxsry);
	PyObject* pd_vxsrz = PyDict_GetItemString(o_Cim, "VXSRz");
	Cim.VXSRz = (float)PyFloat_AsDouble(pd_vxsrz);
	PyObject* pd_vxnrx = PyDict_GetItemString(o_Cim, "VXNRx");
	Cim.VXNRx = (short)PyInt_AS_LONG(pd_vxnrx);
	PyObject* pd_vxnry = PyDict_GetItemString(o_Cim, "VXNRy");
	Cim.VXNRy = (short)PyInt_AS_LONG(pd_vxnry);
	PyObject* pd_vxnrz = PyDict_GetItemString(o_Cim, "VXNRz");
	Cim.VXNRz = (short)PyInt_AS_LONG(pd_vxnrz);
	PyObject* pd_offrx = PyDict_GetItemString(o_Cim, "OFFRx");
	Cim.OFFRx = (float)PyFloat_AsDouble(pd_offrx);
	PyObject* pd_offry = PyDict_GetItemString(o_Cim, "OFFRy");
	Cim.OFFRy = (float)PyFloat_AsDouble(pd_offry);
	PyObject* pd_offrz = PyDict_GetItemString(o_Cim, "OFFRz");
	Cim.OFFRz = (float)PyFloat_AsDouble(pd_offrz);

	PyObject *p_A = PyArray_FROM_OTF(o_A, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_imo = PyArray_FROM_OTF(o_imo, NPY_FLOAT32, NPY_IN_ARRAY);


	/* If that didn't work, throw an exception. */
	if (p_A == NULL || p_imo == NULL) {
		Py_XDECREF(p_A);
		Py_XDECREF(p_imo);
		return NULL;
	}

	float *A = (float*)PyArray_DATA(p_A);
	float *imo = (float*)PyArray_DATA(p_imo);

	// for (int i=0; i<12; i++) printf("A[%d] = %f\n",i,A[i] );

	//=================================================================
	float *imr = rsmpl(imo, A, Cim);
	//=================================================================

	printf("i> new image (x,y,z) = (%d,%d,%d)\n   voxel size: (%6.4f, %6.4f, %6.4f)\n", Cim.VXNRx, Cim.VXNRy, Cim.VXNRz, Cim.VXSRx, Cim.VXSRy, Cim.VXSRz);

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

	//Clean up:
	Py_DECREF(p_A);
	Py_DECREF(p_imo);

	return PyArray_Return(p_imr); //tuple_out;
}


//======================================================================================
// I M A G E    C O N V O L U T I O N
//--------------------------------------------------------------------------------------

static PyObject *img_convolve(PyObject *self, PyObject *args)
{

	//Structure of constants
	Cnst Cnt;

	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	// kernel matrix, for x, y, and z dimensions
	PyObject * o_krnl;
	//input image
	PyObject * o_imi;
	//output image
	PyObject * o_imo;

	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOO", &o_imo, &o_imi, &o_krnl, &o_mmrcnst))
		return NULL;

	PyObject *p_imo = PyArray_FROM_OTF(o_imo, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_imi = PyArray_FROM_OTF(o_imi, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_krnl = PyArray_FROM_OTF(o_krnl, NPY_FLOAT32, NPY_IN_ARRAY);

	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);


	/* If that didn't work, throw an exception. */
	if (p_imo == NULL || p_imi == NULL || p_krnl == NULL) {
		Py_XDECREF(p_imo);
		Py_XDECREF(p_imi);
		Py_XDECREF(p_krnl);
		return NULL;
	}

	float *imo = (float*)PyArray_DATA(p_imo);
	float *imi = (float*)PyArray_DATA(p_imi);
	float *krnl = (float*)PyArray_DATA(p_krnl);

	int Nvk = (int)PyArray_DIM(p_imi, 0);
	int Nvj = (int)PyArray_DIM(p_imi, 1);
	int Nvi = (int)PyArray_DIM(p_imi, 2);
	if (Cnt.VERBOSE == 1) printf("ic> input image size x,y,z=%d,%d,%d\n", Nvk, Nvj, Nvi);

	int Nkr = (int)PyArray_DIM(p_krnl, 1);
	if (Cnt.VERBOSE == 1) printf("ic> kernel size [voxels]: %d\n", Nkr);
	// for (int i=0; i<KERNEL_LENGTH; i++) printf("k[%d]=%f\n", i, krnl[i]);

	if (Nkr != KERNEL_LENGTH) {
		printf("ic> wrong kernel size.\n");
		return Py_None;
	}

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//=================================================================
	setConvolutionKernel(krnl);
	gpu_cnv(imo, imi, Nvk, Nvj, Nvi, Cnt);
	//=================================================================



	Py_DECREF(p_imo);
	Py_DECREF(p_imi);
	Py_DECREF(p_krnl);

	Py_INCREF(Py_None);
	return Py_None;
}





