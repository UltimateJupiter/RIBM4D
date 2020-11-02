#pragma once
#include <cassert>
#include <iostream>
#include <vector>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <bm4d-gpu/kernels.cuh>

void sample_run(double *fftCoefR, double *fftCoefI,
                double *so3CoefR, double *so3CoefI,
                double *rdata, double *idata,
                double *workspace1, double *workspace2,
                double *d_cos_even,
                double **d_seminaive_naive_table,
                int bw, int degLim,
                int wsp1_bsize, int wsp2_bsize,
                int ridata_bsize,
                int so3Coef_bsize,
                int SNT_bsize,
                int fft_patch_size);

__global__ void k_sample_run(double *d_fftCoefR, double *d_fftCoefI,
                             double *d_so3CoefR, double *d_so3CoefI,
                             double *d_rdata, double *d_idata,
                             double *d_workspace1,
                             double *d_workspace2,
                             double *d_cos_even,
                             double **d_seminaive_naive_table,
                             int bw, int degLim,
                             int wsp1_bsize, int wsp2_bsize,
                             int ridata_bsize,
                             int so3Coef_bsize,
                             int SNT_bsize,
                             int fft_patch_size);

// Compute the best correlation and rotation angle
__device__ float4 soft_corr(double *sigCoefR, double *sigCoefI,
                            double *patCoefR, double *patCoefI,
                            double *so3CoefR, double *so3CoefI,
                            double *rdata, double *idata,
                            double *workspace1,
                            double *workspace2,
                            double *cos_even,
                            double **seminaive_naive_table,
                            int bw, int degLim);

