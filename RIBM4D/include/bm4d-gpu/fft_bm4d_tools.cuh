#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

#define FORWARD 1
#define BACKWORD -1

struct fft_bm4d_tools {
    /*
    * 
    * Apply fft related operations on a data patch
    * 
    */

    // patch size
    int N;

    // bandwidth
    int B;

    // mean of patch coordinates
    int xm, ym, zm;

    __device__ fft_bm4d_tools(int N_, int B_) {
        N = N_;
        xm = N_ / 2;
        ym = N_ / 2;
        zm = N_ / 2;
        B = B_;
    }

    // __device__ void init_patch_n(int N_);

    // __device__ void init_bandwidth(int B_);

    //In-place 3-d fftshift
    __device__ void fftshift_3d_in(float* data_re, float* data_im);

    //In-place 3-d ifftshift
    __device__ void ifftshift_3d_in(float* data_re, float* data_im);

    //Apply 3-d fft on cufftComplex data in place
    __device__ void fft_3d(float* data_re, float* data_im, float* buf_re, float* buf_im);

    __device__ void complex_abs(float* data_re, float* data_im);

    //Map volume data to spherical coordinates and do the integration
    __device__ void spherical_mapping(double* maga_re, double* maga_im, float* data_re);
};