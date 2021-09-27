#ifndef _NIMPA_NLM_H_
#define _NIMPA_NLM_H_

/// main NLM function
void d_nlm3d(float *dst, const float *src, const float *ref, float sigma, int X, int Y, int Z,
             int HALF_WIDTH, bool _sync);

#endif // _NIMPA_NLM_H_