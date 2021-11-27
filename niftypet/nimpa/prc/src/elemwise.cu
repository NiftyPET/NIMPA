#include "cuhelpers.h"
#include "elemwise.h"

/// dst = src_num / src_div
__global__ void div(float *dst, const float *src_num, const float *src_div, const size_t N,
                    float zeroDivDefault) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = (src_div[i] || zeroDivDefault == FLOAT_MAX) ? src_num[i] / src_div[i] : zeroDivDefault;
}

/// main div function
void d_div(float *dst, const float *src_num, const float *src_div, const size_t N,
           float zeroDivDefault, bool _sync) {
  dim3 thrds(NIMPA_CU_THREADS, 1, 1);
  dim3 blcks((N + NIMPA_CU_THREADS - 1) / NIMPA_CU_THREADS, 1, 1);
  div<<<blcks, thrds>>>(dst, src_num, src_div, N, zeroDivDefault);
  HANDLE_ERROR(cudaGetLastError());

  if (_sync) HANDLE_ERROR(cudaDeviceSynchronize()); // unified memcpy device2host
}
