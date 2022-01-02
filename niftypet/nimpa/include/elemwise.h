#ifndef _NIMPA_DIV_H_
#define _NIMPA_DIV_H_

#ifndef FLOAT_MAX
#include <limits>
const float FLOAT_MAX = std::numeric_limits<float>::infinity();
#endif

/// main div function
void d_div(float *dst, const float *src_num, const float *src_div, const size_t N,
           float zeroDivDefault = FLOAT_MAX, bool _sync = true);
/// main mul function
void d_mul(float *dst, const float *src_a, const float *src_b, const size_t N, bool _sync = true);
/// main add function
void d_add(float *dst, const float *src_a, const float *src_b, const size_t N, bool _sync = true);

#endif // _NIMPA_DIV_H_
