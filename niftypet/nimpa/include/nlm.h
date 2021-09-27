#ifndef _NIMPA_NLM_H_
#define _NIMPA_NLM_H_

/// main NLM function
void d_nlm3d(float *dst, const float *src, const float *ref, float sigma, int Z, int Y, int X,
             int HALF_WIDTH, bool _sync);

#endif // _NIMPA_NLM_H_
