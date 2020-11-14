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

void sample_run(double *d_sigR, double *d_sigI,
    double *d_so3SigR, double *d_so3SigI,
    double *d_workspace1, double *d_workspace2,
    double *d_sigCoefR, double *d_sigCoefI,
    double *d_patCoefR, double *d_patCoefI,
    double *d_so3CoefR, double *d_so3CoefI,
    double *d_seminaive_naive_tablespace,
    double *d_cos_even,
    double **d_seminaive_naive_table,
    int bwIn, int bwOut, int degLim,
    int sig_patch_size,
    int wsp1_bsize,
    int wsp2_bsize,
    int sigpatCoef_bsize,
    int so3Coef_bsize,
    int so3Sig_bsize,
    int SNTspace_bsize,
    int SNT_bsize,
    int cos_even_bsize);

__global__ void k_sample_run(double *d_sigR, double *d_sigI,
    double *d_so3SigR, double *d_so3SigI,
    double *d_workspace1, double *d_workspace2,
    double *d_sigCoefR, double *d_sigCoefI,
    double *d_patCoefR, double *d_patCoefI,
    double *d_so3CoefR, double *d_so3CoefI,
    double *d_seminaive_naive_tablespace,
    double *d_cos_even,
    double **d_seminaive_naive_table,
    int bwIn, int bwOut, int degLim,
    int sig_patch_size,
    int wsp1_bsize,
    int wsp2_bsize,
    int sigpatCoef_bsize,
    int so3Coef_bsize,
    int so3Sig_bsize,
    int SNTspace_bsize,
    int SNT_bsize,
    int cos_even_bsize);

// Compute the best correlation and rotation angle
__device__ float4 soft_corr(double *sigR, double *sigI,
    double *patR, double *patI,
    double *so3SigR, double *so3SigI,
    double *workspace1, double *workspace2,
    double *sigCoefR, double *sigCoefI,
    double *patCoefR, double *patCoefI,
    double *so3CoefR, double *so3CoefI,
    double *seminaive_naive_tablespace,
    double *cos_even,
    double **seminaive_naive_table,
    int bwIn, int bwOut, int degLim);

