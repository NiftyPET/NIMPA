#include "cuhelpers.h"
#include "nlm.h"
#include <cassert>

/// 3D nonlocal means
__global__ void nlm_3d(float *dst, const float *src, const float *ref,
                       const float exp_norm, // = -1.0f / (2.0f * sigma * sigma)
                       int Z, int Y, int X, int HALF_WIDTH) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= X) return;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= Y) return;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z >= Z) return;
  int idx = z * Y * X + y * X + x;

  float result = 0;
  float norm = 0;
  float weight;
  for (int i = max(0, x - HALF_WIDTH); i < min(X, x + HALF_WIDTH); ++i) {
    for (int j = max(0, y - HALF_WIDTH); j < min(Y, y + HALF_WIDTH); ++j) {
      for (int k = max(0, z - HALF_WIDTH); k < min(Z, z + HALF_WIDTH); ++k) {
        weight = exp(pow(ref[k * Y * X + j * X + i] - ref[idx], 2) * exp_norm);
        /// distance weight - doesn't work as well for minimising NRMSE vs Truth
        /*
        weight /= (0.5 + sqrt(
            pow((i - x), 2) +
            pow((j - y), 2) +
            pow((k - z), 2)));
        */
        result += weight * src[k * Y * X + j * X + i];
        norm += weight;
      }
    }
  }

  dst[idx] = norm == 0 ? src[idx] : result / norm;
}

/// main NLM function
void d_nlm3d(float *dst, const float *src, const float *ref, float sigma, int Z, int Y, int X,
             int HALF_WIDTH, bool _sync) {
  assert(dst != src);

  dim3 thrds(32, 16, NIMPA_CU_THREADS / (32 * 16));
  dim3 blcks((X + 31) / 32, (Y + 15) / 16,
             (Z + NIMPA_CU_THREADS / (32 * 16) - 1) / (NIMPA_CU_THREADS / (32 * 16)));
  nlm_3d<<<blcks, thrds>>>(dst, src, ref, -1.0f / (2.0f * sigma * sigma), Z, Y, X, HALF_WIDTH);
  HANDLE_ERROR(cudaGetLastError());

  if (_sync) HANDLE_ERROR(cudaDeviceSynchronize()); // unified memcpy device2host
}
