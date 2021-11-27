#include "cuhelpers.h"
#include "isub.h"
#include <cassert>

/// dst = src[idxs, 0:X] where J = len(indxs)
__global__ void isub2d(float *dst, const float *src, const int *idxs, int J, int X) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= X) return;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= J) return;
  dst[j * X + x] = src[idxs[j] * X + x];
}

/// main isub2d function
void d_isub2d(float *dst, const float *src, const int *idxs, int J, int X, bool _sync) {
  assert(dst != src);

  dim3 thrds(NIMPA_CU_THREADS / 32, 32);
  dim3 blcks((X + NIMPA_CU_THREADS / 32 - 1) / (NIMPA_CU_THREADS / 32), (J + 31) / 32);
  isub2d<<<blcks, thrds>>>(dst, src, idxs, J, X);
  HANDLE_ERROR(cudaGetLastError());

  if (_sync) HANDLE_ERROR(cudaDeviceSynchronize()); // unified memcpy device2host
}
