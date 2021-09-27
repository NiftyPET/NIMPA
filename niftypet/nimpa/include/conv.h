#ifndef _NIMPA_CONV_H_
#define _NIMPA_CONV_H_

// logging levels
#define LOGDEBUG 10
#define LOGINFO 20
#define LOGWARNING 30

// number of threads used for element-wise GPU calculations
#ifndef NIMPA_CU_THREADS
#define NIMPA_CU_THREADS 1024
#endif

/* separable convolution */
#ifndef NIMPA_KERNEL_RADIUS
#define NIMPA_KERNEL_RADIUS 8
#endif
#define KERNEL_LENGTH (2 * NIMPA_KERNEL_RADIUS + 1)

// Column convolution filter
#define COLUMNS_BLOCKDIM_X 8
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS 1

// Row convolution filter
#define ROWS_BLOCKDIM_X 8
#define ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1

struct Cnst {
  char DEVID; // device (GPU) ID.  allows choosing the device on which to perform calculations
  char LOG;   // logging
};

/// krnl: separable three kernels for x, y and z
void setConvolutionKernel(float *krnl, bool handle_errors = true);

/// sigma: Gaussian sigma
void setKernelGaussian(float sigma, bool handle_errors = true);

/// main convolution function
void d_conv(float *dst, float *src, int Nvk, int Nvj, int Nvi, bool _memset = true,
            bool _sync = true, int _log = LOGINFO);

#endif // _NIMPA_CONV_H_
