#include "rsmpl.h"

// void HandleError( cudaError_t err, const char *file, int line ){
//     if (err != cudaSuccess) {
//         printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
//         exit( EXIT_FAILURE );
//     }
// }
//..................................................................................................................................


__constant__ float cA[12];

__global__
void d_rsmpl(float *imr,
	const float *imo,
	Cimg Cim) {

	// extern __shared__ float s[];

	int ib = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
	//int it = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;


	float x = (blockIdx.x*Cim.VXSOx + Cim.OFFOx) + Cim.VXSOx / VDIV*(0.5 + threadIdx.x);
	float y = -(blockIdx.y*Cim.VXSOy + Cim.OFFOy) - Cim.VXSOy / VDIV*(0.5 + threadIdx.y);
	float z = blockIdx.z*Cim.VXSOz + Cim.OFFOz + Cim.VXSOz / VDIV*(0.5 + threadIdx.z);

	float xp = cA[0] * x + cA[1] * y + cA[2] * z + cA[3];
	float yp = cA[4] * x + cA[5] * y + cA[6] * z + cA[7];
	float zp = cA[8] * x + cA[9] * y + cA[10] * z + cA[11];

	short u = roundf(-Cim.OFFRx / Cim.VXSRx) + floorf((xp) / Cim.VXSRx);
	short v = roundf(-Cim.OFFRy / Cim.VXSRy) - ceilf((yp) / Cim.VXSRy);
	short w = roundf(-Cim.OFFRz / Cim.VXSRz) + floorf((zp) / Cim.VXSRz);

	if ((u<Cim.VXNRx) && (v<Cim.VXNRy) && (w<Cim.VXNRz) && (u >= 0) && (v >= 0) && (w >= 0))
		atomicAdd(imr + u + v*Cim.VXNRx + w*Cim.VXNRx*Cim.VXNRy, imo[ib] / (VDIV*VDIV*VDIV));

}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


float *rsmpl(float *imo,
	float *A,
	Cimg Cim)

{

	float *d_imr;
	HANDLE_ERROR(cudaMalloc(&d_imr, Cim.VXNRx*Cim.VXNRy*Cim.VXNRz * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_imr, 0, Cim.VXNRx*Cim.VXNRy*Cim.VXNRz * sizeof(float)));

	float *d_imo;
	HANDLE_ERROR(cudaMalloc(&d_imo, Cim.VXNOx*Cim.VXNOy*Cim.VXNOz * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_imo, imo, Cim.VXNOx*Cim.VXNOy*Cim.VXNOz * sizeof(float), cudaMemcpyHostToDevice));


	cudaMemcpyToSymbol(cA, A, 12 * sizeof(float));
	// double * d_A;
	// HANDLE_ERROR( cudaMalloc(&d_A, 12*sizeof(double)) );
	// HANDLE_ERROR( cudaMemcpy(d_A, A, 12*sizeof(double), cudaMemcpyHostToDevice) );


	//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
	printf("i> calculating transformation with %d samples per voxel...", VDIV);
	dim3 grid(Cim.VXNOx, Cim.VXNOy, Cim.VXNOz);
	dim3 block(VDIV, VDIV, VDIV);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	d_rsmpl << <grid, block >> >(d_imr, d_imo, Cim);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) { printf("CUDA kernel for image resampling: error: %s\n", cudaGetErrorString(error)); exit(-1); }
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("DONE in %fs.\n\n", 0.001*elapsedTime);
	//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


	//allocate memory for the resampled image for the return
	float *imr = (float*)malloc(Cim.VXNRx*Cim.VXNRy*Cim.VXNRz * sizeof(float));
	//copy the image from GPU to CPU
	HANDLE_ERROR(cudaMemcpy(imr, d_imr, Cim.VXNRx*Cim.VXNRy*Cim.VXNRz * sizeof(float), cudaMemcpyDeviceToHost));


	return imr;
}


