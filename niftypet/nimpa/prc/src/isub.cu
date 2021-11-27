#include "cuhelpers.h"
#include "isub.h"
#include <cassert>

/// dst = src[idxs, 0:X] where J = len(indxs)
__global__ void isub(float *dst, const float *src, const int *idxs, int J, int X) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= J) return;
  dst += j * X;
  src += idxs[j] * X;
  for (size_t x = 0; x < X; ++x) dst[x] = src[x];
}

/// main 2D isub function
void d_isub(float *dst, const float *src, const int *idxs, int J, int X, bool _sync) {
  assert(dst != src);

  dim3 thrds(NIMPA_CU_THREADS, 1, 1);
  dim3 blcks((J + NIMPA_CU_THREADS - 1) / NIMPA_CU_THREADS, 1, 1);
  isub<<<blcks, thrds>>>(dst, src, idxs, J, X);
  HANDLE_ERROR(cudaGetLastError());

  if (_sync) HANDLE_ERROR(cudaDeviceSynchronize()); // unified memcpy device2host
}
