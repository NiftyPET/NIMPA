#include "conv.h"
#include "cuhelpers.h"
#include <cassert>
#include <cstdio>

/**
 * padding: uses a `__global__ for` loop performing contiguous copies.
 * NB: the following are all slower:
 * - cudaMemcpy
 * - cudaMemcpyAsync
 * - __global__ memcpy
 * - __global__ (one thread per element, i.e. more threads)
 */
/// x, y, z: how many slices to add
__global__ void pad(float *dst, float *src, const int y, const int x, const int Z, const int Y,
                    const int X) {
  int k = threadIdx.x + blockDim.x * blockIdx.x;
  if (k >= Z) return;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (j >= Y) return;
  dst += k * (Y + y) * (X + x) + j * (X + x);
  src += k * Y * X + j * X;
  for (int i = 0; i < X; ++i) dst[i] = src[i];
}
void d_pad(float *dst, float *src, const int z, const int y, const int x, const int Z, const int Y,
           const int X) {
  HANDLE_ERROR(cudaMemset(dst, 0, (Z + z) * (Y + y) * (X + x) * sizeof(float)));
  dim3 BpG((Z + NIMPA_CU_THREADS / 32 - 1) / (NIMPA_CU_THREADS / 32), (Y + 31) / 32);
  dim3 TpB(NIMPA_CU_THREADS / 32, 32);
  pad<<<BpG, TpB>>>(dst, src, y, x, Z, Y, X);
  HANDLE_ERROR(cudaGetLastError());
}
/// y, z: how many slices to remove
__global__ void unpad(float *dst, float *src, const int y, const int x, const int Z, const int Y,
                      const int X) {
  int k = threadIdx.x + blockDim.x * blockIdx.x;
  if (k >= Z) return;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (j >= Y) return;
  dst += k * Y * X + j * X;
  src += k * (Y + y) * (X + x) + j * (X + x);
  for (int i = 0; i < X; ++i) dst[i] = src[i];
}
void d_unpad(float *dst, float *src, const int y, const int x, const int Z, const int Y,
             const int X) {
  dim3 BpG((Z + NIMPA_CU_THREADS / 32 - 1) / (NIMPA_CU_THREADS / 32), (Y + 31) / 32);
  dim3 TpB(NIMPA_CU_THREADS / 32, 32);
  unpad<<<BpG, TpB>>>(dst, src, y, x, Z, Y, X);
  HANDLE_ERROR(cudaGetLastError());
}

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
      sum += c_Kernel[2 * KERNEL_LENGTH + NIMPA_KERNEL_RADIUS - j] *
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

/// convolution function helper
void d_conv_pow2(float *dst, float *src, int Nvk, int Nvj, int Nvi) {
  assert(dst != src);
  assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= NIMPA_KERNEL_RADIUS);
  assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= NIMPA_KERNEL_RADIUS);

  assert(Nvk % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);
  assert(Nvj % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);
  assert(Nvi % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);

  assert(Nvj % ROWS_BLOCKDIM_Y == 0);
  assert(Nvi % COLUMNS_BLOCKDIM_X == 0);

  // temporary image for intermediate results
  float *d_buff;
  HANDLE_ERROR(cudaMalloc((void **)&d_buff, Nvk * Nvj * Nvi * sizeof(float)));
  // HANDLE_ERROR(cudaMemset(d_buff, 0, Nvk * Nvj * Nvi * sizeof(float)));

  // perform smoothing
  //------ ROWS -------
  dim3 blocks(Nvi / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), Nvj / ROWS_BLOCKDIM_Y);
  dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
  //----- COLUMNS ----
  dim3 blocks2(Nvi / COLUMNS_BLOCKDIM_X, Nvj / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
  dim3 threads2(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
  for (int k = 0; k < Nvk; k++) {
    cnv_rows<<<blocks, threads>>>(dst + k * Nvi * Nvj, src + k * Nvi * Nvj, Nvi, Nvj, Nvi);
    HANDLE_ERROR(cudaGetLastError());
    cnv_columns<<<blocks2, threads2>>>(d_buff + k * Nvi * Nvj, dst + k * Nvi * Nvj, Nvi, Nvj, Nvi,
                                       KERNEL_LENGTH);
    HANDLE_ERROR(cudaGetLastError());
  }
  //----- THIRD DIM ----
  dim3 blocks3(Nvi / COLUMNS_BLOCKDIM_X, Nvk / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
  dim3 threads3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
  for (int j = 0; j < Nvj; j++) {
    cnv_columns<<<blocks3, threads3>>>(dst + j * Nvi, d_buff + j * Nvi, Nvi, Nvk, Nvi * Nvj, 0);
    HANDLE_ERROR(cudaGetLastError());
  }

  HANDLE_ERROR(cudaFree(d_buff));
}

/// main convolution function
void d_conv(float *dst, float *src, int Nvk, int Nvj, int Nvi, bool _memset, bool _sync,
            int _log) {
  assert(dst != src);
  int Npk = ((COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) -
             Nvk % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)) %
            (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
  int Npj = ((COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) -
             Nvj % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)) %
            (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
  int Npi = ((ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) - Nvi % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)) %
            (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);

  float *d_src;
  float *d_dst;
  if (Npk | Npj | Npi) {
    if (_log <= LOGINFO)
      fprintf(stderr, "i> padding:(%d, %d, %d) since input:(%d, %d, %d)\n", Npi, Npj, Npk, Nvi,
              Nvj, Nvk);
    Nvi += Npi;
    Nvj += Npj;
    Nvk += Npk;
    HANDLE_ERROR(cudaMalloc((void **)&d_dst, Nvk * Nvj * Nvi * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_src, Nvk * Nvj * Nvi * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_dst, 0, Nvk * Nvj * Nvi * sizeof(float)));
    d_pad(d_src, src, Npk, Npj, Npi, Nvk - Npk, Nvj - Npj, Nvi - Npi);
  } else {
    d_dst = dst;
    d_src = src;
    if (_memset) memset(dst, 0, Nvk * Nvj * Nvi * sizeof(float));
  }

  d_conv_pow2(d_dst, d_src, Nvk, Nvj, Nvi);

  if (Npk | Npj | Npi) {
    d_unpad(dst, d_dst, Npj, Npi, Nvk - Npk, Nvj - Npj, Nvi - Npi);
    HANDLE_ERROR(cudaFree(d_dst));
    HANDLE_ERROR(cudaFree(d_src));
  }

  if (_sync) HANDLE_ERROR(cudaDeviceSynchronize()); // unified memcpy device2host
}
