#include "conv.h"
#include "cuhelpers.h"
#include <cassert>
#include <cstdio>

/** separable convolution */
/// Convolution kernel array
__constant__ float c_Kernel[3 * KERNEL_LENGTH];
/// krnl: separable three kernels for x, y and z
void setConvolutionKernel(float *krnl, bool handle_errors) {
  cudaMemcpyToSymbol(c_Kernel, krnl, 3 * KERNEL_LENGTH * sizeof(float));
  if (handle_errors) HANDLE_ERROR(cudaGetLastError());
}
/// sigma: Gaussian sigma
void setKernelGaussian(float sigma, bool handle_errors) {
  float knlRM[KERNEL_LENGTH * 3];
  const double tmpE = -1.0 / (2 * sigma * sigma);
  for (int i = 0; i < KERNEL_LENGTH; ++i)
    knlRM[i] = (float)exp(tmpE * pow(NIMPA_KERNEL_RADIUS - i, 2));
  // normalise
  double knlSum = 0;
  for (size_t i = 0; i < KERNEL_LENGTH; ++i) knlSum += knlRM[i];
  for (size_t i = 0; i < KERNEL_LENGTH; ++i) {
    knlRM[i] /= knlSum;
    // also fill in other dimensions
    knlRM[i + KERNEL_LENGTH] = knlRM[i];
    knlRM[i + KERNEL_LENGTH * 2] = knlRM[i];
  }
  setConvolutionKernel(knlRM, handle_errors);
}

/// Row convolution filter
__global__ void cnv_rows(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch) {
  __shared__ float s_Data[ROWS_BLOCKDIM_Y]
                         [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

  // Offset to the left halo edge
  const int baseX =
      (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
  }

// Load left halo
#pragma unroll
  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

// Load right halo
#pragma unroll
  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
       i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

  // Compute and store results
  __syncthreads();

#pragma unroll
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum = 0;
#pragma unroll
    for (int j = -NIMPA_KERNEL_RADIUS; j <= NIMPA_KERNEL_RADIUS; j++) {
      sum += c_Kernel[NIMPA_KERNEL_RADIUS - j] *
             s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
    }
    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

/// Column convolution filter
__global__ void cnv_columns(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch,
                            int offKrnl // kernel offset for asymmetric kernels
                                        // x, y, z (still the same dims though)
) {
  __shared__ float
      s_Data[COLUMNS_BLOCKDIM_X]
            [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
  const int baseY =
      (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
  }

// Upper halo
#pragma unroll
  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
  }

// Lower halo
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
  }

  // Compute and store results
  __syncthreads();

#pragma unroll
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    float sum = 0;
#pragma unroll
    for (int j = -NIMPA_KERNEL_RADIUS; j <= NIMPA_KERNEL_RADIUS; j++) {
      sum += c_Kernel[offKrnl + NIMPA_KERNEL_RADIUS - j] *
             s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
    }
    d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}

/// main convolution function
void d_conv(float *dst, float *src, int Nvk, int Nvj, int Nvi, bool _memset, bool _sync) {
  assert(dst != src);
  assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= NIMPA_KERNEL_RADIUS);
  assert(Nvk % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
  assert(Nvj % ROWS_BLOCKDIM_Y == 0);

  assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= NIMPA_KERNEL_RADIUS);
  assert(Nvk % COLUMNS_BLOCKDIM_X == 0);
  assert(Nvj % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

  assert(Nvi % COLUMNS_BLOCKDIM_X == 0);

  if (_memset) memset(dst, 0, Nvk * Nvj * Nvi * sizeof(float));

  // temporary image for intermediate results
  float *d_buff;
  HANDLE_ERROR(cudaMalloc((void **)&d_buff, Nvk * Nvj * Nvi * sizeof(float)));
  // HANDLE_ERROR(cudaMemset(d_buff, 0, Nvk * Nvj * Nvi * sizeof(float)));

  // perform smoothing
  for (int k = 0; k < Nvk; k++) {
    //------ ROWS -------
    dim3 blocks(Nvi / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), Nvj / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    cnv_rows<<<blocks, threads>>>(dst + k * Nvi * Nvj, src + k * Nvi * Nvj, Nvi, Nvj, Nvi);
    HANDLE_ERROR(cudaGetLastError());

    //----- COLUMNS ----
    dim3 blocks2(Nvi / COLUMNS_BLOCKDIM_X, Nvj / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads2(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    cnv_columns<<<blocks2, threads2>>>(d_buff + k * Nvi * Nvj, dst + k * Nvi * Nvj, Nvi, Nvj, Nvi,
                                       KERNEL_LENGTH);
    HANDLE_ERROR(cudaGetLastError());
  }

  //----- THIRD DIM ----
  for (int j = 0; j < Nvj; j++) {
    dim3 blocks3(Nvi / COLUMNS_BLOCKDIM_X, Nvk / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    cnv_columns<<<blocks3, threads3>>>(dst + j * Nvi, d_buff + j * Nvi, Nvi, Nvk, Nvi * Nvj,
                                       2 * KERNEL_LENGTH);
    HANDLE_ERROR(cudaGetLastError());
  }

  HANDLE_ERROR(cudaFree(d_buff));
  if (_sync) HANDLE_ERROR(cudaDeviceSynchronize()); // unified memcpy device2host
}
