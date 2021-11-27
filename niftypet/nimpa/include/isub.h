#ifndef _NIMPA_ISUB_H_
#define _NIMPA_ISUB_H_

/// main isub2d function
void d_isub2d(float *dst, const float *src, const int *idxs, int J, int X, bool _sync = true);

#endif // _NIMPA_ISUB_H_
