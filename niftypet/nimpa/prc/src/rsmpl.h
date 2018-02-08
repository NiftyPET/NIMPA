#include <stdio.h>

// get the CUDA handle error function for here:
#include "conv.h"

#ifndef RSMPL_H
#define RSMPL_H

//fine subsampling divisor
#define VDIV 10

// #define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
// void HandleError( cudaError_t err, const char *file, int line );

struct Cimg {
	float VXSOx;
	float VXSOy;
	float VXSOz;
	short VXNOx;
	short VXNOy;
	short VXNOz;
	float OFFOx;
	float OFFOy;
	float OFFOz;

	float VXSRx;
	float VXSRy;
	float VXSRz;
	short VXNRx;
	short VXNRy;
	short VXNRz;
	float OFFRx;
	float OFFRy;
	float OFFRz;
};



float *rsmpl(float *imo,
	float *A,
	Cimg Cim);

#endif