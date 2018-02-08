#include <assert.h>
#include <stdio.h>
#include "conv.h"


void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}


////////////////////////////////////
// Convolution kernel array
////////////////////////////////////
__constant__ float c_Kernel[3 * KERNEL_LENGTH];
void setConvolutionKernel(float *hKrnl) {
	//hKrnl: separable three kernels for x, y and z
	cudaMemcpyToSymbol(c_Kernel, hKrnl, 3 * KERNEL_LENGTH * sizeof(float));
}


/////////////////////////////////////
// Row convolution filter
/////////////////////////////////////
__global__ void cnv_rows(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int pitch
)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Load main data
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
#pragma unroll

	for (int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo
#pragma unroll

	for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Compute and store results
	__syncthreads();
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		float sum = 0;

#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
		}

		d_Dst[i * ROWS_BLOCKDIM_X] = sum;
	}
}

//////////////////////////////////////
// Column convolution filter
//////////////////////////////////////

__global__ void cnv_columns(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int pitch,
	int offKrnl //kernel offset for asymmetric kernels x, y, z (still the same dims though)
)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Main data
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
	}

	//Upper halo
#pragma unroll

	for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Lower halo
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Compute and store results
	__syncthreads();
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		float sum = 0;
#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[offKrnl + KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
		}

		d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
	}
}


//-----------------------------------------------------------------------------------------------
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void gpu_cnv(float *imgout, float *imgint, int Nvk, int Nvj, int Nvi, Cnst Cnt) {

	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
	assert(Nvk % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	assert(Nvj % ROWS_BLOCKDIM_Y == 0);

	assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
	assert(Nvk % COLUMNS_BLOCKDIM_X == 0);
	assert(Nvj % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	assert(Nvi % COLUMNS_BLOCKDIM_X == 0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float *d_imgout;
	HANDLE_ERROR(cudaMalloc((void **)&d_imgout, Nvk*Nvj*Nvi * sizeof(float)));
	cudaMemset(d_imgout, 0, Nvk*Nvj*Nvi * sizeof(float));

	float *d_imgint;
	HANDLE_ERROR(cudaMalloc((void **)&d_imgint, Nvk*Nvj*Nvi * sizeof(float)));
	cudaMemcpy(d_imgint, imgint, Nvk*Nvj*Nvi * sizeof(float), cudaMemcpyHostToDevice);

	//temporary image for intermediate results
	float *d_buff;
	HANDLE_ERROR(cudaMalloc((void **)&d_buff, Nvk*Nvj*Nvi * sizeof(float)));



	// perform smoothing
	for (int k = 0; k<Nvk; k++) {

		//------ ROWS -------
		int Bx = Nvi / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
		int By = Nvj / ROWS_BLOCKDIM_Y;
		int Tx = ROWS_BLOCKDIM_X;
		int Ty = ROWS_BLOCKDIM_Y;
		dim3 blocks(Bx, By);
		dim3 threads(Tx, Ty);
		cnv_rows <<<blocks, threads >>>(d_imgout + k*Nvi*Nvj, d_imgint + k*Nvi*Nvj,
			Nvi, Nvj, Nvi);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) { printf("CUDA kernel ROWS error: %s\n", cudaGetErrorString(error)); exit(-1); }

		//----- COLUMNS ----
		dim3 blocks2(Nvi / COLUMNS_BLOCKDIM_X, Nvj / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
		dim3 threads2(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
		cnv_columns << <blocks2, threads2 >> >(d_buff + k*Nvi*Nvj, d_imgout + k*Nvi*Nvj,
			Nvi, Nvj, Nvi, KERNEL_LENGTH);
		error = cudaGetLastError();
		if (error != cudaSuccess) { printf("CUDA kernel COLUMNS error: %s\n", cudaGetErrorString(error)); exit(-1); }

	}

	//----- THIRD DIM ----
	for (int j = 0; j<Nvj; j++) {
		dim3 blocks3(Nvi / COLUMNS_BLOCKDIM_X, Nvk / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
		dim3 threads3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
		cnv_columns <<<blocks3, threads3 >>>(d_imgout + j*Nvi, d_buff + j*Nvi,
			Nvi, Nvk, Nvi*Nvj, 2 * KERNEL_LENGTH);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) { printf("CUDA kernel THIRD DIM error: %s\n", cudaGetErrorString(error)); exit(-1); }
	}

	HANDLE_ERROR(cudaMemcpy(imgout, d_imgout, Nvi*Nvj*Nvk * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_buff);
	cudaFree(d_imgint);
	cudaFree(d_imgout);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1) printf("ic> elapsed time of convolution: %f\n", 0.001*elapsedTime);


}
