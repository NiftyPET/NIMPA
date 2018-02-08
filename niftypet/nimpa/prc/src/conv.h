#ifndef CONVOLUTIONSEPARABLE_H
#define CONVOLUTIONSEPARABLE_H

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2*KERNEL_RADIUS + 1)

// Column convolution filter
#define   COLUMNS_BLOCKDIM_X 8
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

// Row convolution filter
#define   ROWS_BLOCKDIM_X 8
#define   ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

struct Cnst {
	char DEVID; 	// device (GPU) ID.  allows choosing the device on which to perform calculations 
	bool VERBOSE;
};

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
void HandleError(cudaError_t err, const char *file, int line);


// GPU convolution
void setConvolutionKernel(float *hKrnl);

void gpu_cnv(float *imgout, float *imgint, int Nvk, int Nvj, int Nvi, Cnst Cnt);

#endif
