/*
 * 2016, Vladislav Tananaev
 * tananaev@cs.uni-freiburg.de
 */
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
#include "helper_cuda.h"
#include "parameters.h"
#include "stdio.h"

#undef NDEBUG
#ifndef idx3
#define idx3(x, y, z, x_size, y_size) ((x) + ((y) + (y_size) * (z)) * (x_size))
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

struct uint3float1 {
    uint x;
    uint y;
    uint z;
    float val;
    __host__ __device__ uint3float1() : x(0), y(0), z(0), val(-1){};
    __host__ __device__ uint3float1(uint x, uint y, uint z, float val) : x(x), y(y), z(z), val(val) {}
};
inline uint3float1 make_uint3float1(uint x, uint y, uint z, float val) {
    return uint3float1(x, y, z, val);
}
inline uint3float1 make_uint3float1(uint3 c, float val) {
    return uint3float1(c.x, c.y, c.z, val);
}

// struct supporting message passing on Euler angles for rotations
struct uint3float4 {
    uint x;
    uint y;
    uint z;
    float val;
    float alpha;
    float beta;
    float gamma;
    __host__ __device__ uint3float4() : x(0), y(0), z(0), val(-1), alpha(0), beta(0), gamma(0){};
    __host__ __device__ uint3float4(uint x, uint y, uint z, float val, float alpha, float beta, float gamma) : x(x), y(y), z(z), val(val), alpha(alpha), beta(beta), gamma(gamma){}
};
inline uint3float4 make_uint3float4(uint x, uint y, uint z, float val, float alpha, float beta, float gamma) {
    return uint3float4(x, y, z, val, alpha, beta, gamma);
}
inline uint3float4 make_uint3float4(uint3 c, float val, float alpha, float beta, float gamma) {
    return uint3float4(c.x, c.y, c.z, val, alpha, beta, gamma);
}

// Referencing a patch with rotation (containing x, y, z, match score, and basis for optimal rotation)
struct rotateRef {
    uint x; // coords of reference point
    uint y;
    uint z;
    float val;

    // basis vectors after rotation
    // can also intepret as the rotation matrix since it is based on the standard basis
    float v1x;
    float v1y;
    float v1z;

    float v2x;
    float v2y;
    float v2z;

    float v3x;
    float v3y;
    float v3z;

    __host__ __device__ rotateRef() : x(0), y(0), z(0), val(-1), v1x(1), v1y(0), v1z(0), v2x(0), v2y(1), v2z(0), v3x(0), v3y(0), v3z(1){};
    __host__ __device__ rotateRef(uint x, uint y, uint z, float val, float v1x, float v1y, float v1z, float v2x, float v2y, float v2z, float v3x, float v3y, float v3z) : x(x), y(y), z(z), val(val), v1x(v1x), v1y(v1y), v1z(v1z), v2x(v2x), v2y(v2y), v2z(v2z), v3x(v3x), v3y(v3y), v3z(v3z){}
};


void run_fft_precomp(const uchar* __restrict d_noisy_volume,
                     const uint3 size,
                     const uint3 tshape,
                     const bm4d_gpu::Parameters params,
                     double *d_fftCoefR,
                     double *d_fftCoefI,
                     const cudaDeviceProp &d_prop);

void run_block_matching(const uchar* __restrict d_noisy_volume, const uint3 size, const uint3 tshape,
                        const bm4d_gpu::Parameters params, uint3float1* d_stacks, uint* d_nstacks,
                        const cudaDeviceProp& d_prop);

void run_block_matching_rot(const uchar* __restrict d_noisy_volume,
                            const double* __restrict d_fftCoefR,
                            const double* __restrict d_fftCoefI,
                            const uint3 size,
                            const uint3 tshape,
                            const bm4d_gpu::Parameters params,
                            rotateRef *d_stacks_rot,
                            uint *d_nstacks_rot,
                            int fft_patch_size,
                            const cudaDeviceProp &d_prop);
// Gather cubes together
void gather_cubes(const uchar* __restrict img, const uint3 size, const uint3 tshape,
                  const bm4d_gpu::Parameters params, uint3float1*& d_stacks, uint* d_nstacks,
                  float*& d_gathered4dstack, uint& gather_stacks_sum, const cudaDeviceProp& d_prop);
// Perform 3D DCT
void run_dct3d(float* d_gathered4dstack, uint gather_stacks_sum, int patch_size,
               const cudaDeviceProp& d_prop);
// Do WHT in 4th dim + Hard Thresholding + IWHT
void run_wht_ht_iwht(float* d_gathered4dstack, uint gather_stacks_sum, int patch_size,
                     uint* d_nstacks, const uint3 tshape, float*& d_group_weights,
                     const bm4d_gpu::Parameters params, const cudaDeviceProp& d_prop);
// Perform inverse 3D DCT
void run_idct3d(float* d_gathered4dstack, uint gather_stacks_sum, int patch_size,
                const cudaDeviceProp& d_prop);
// Aggregate
void run_aggregation(float* final_image, const uint3 size, const uint3 tshape,
                     const float* d_gathered4dstack, uint3float1* d_stacks, uint* d_nstacks,
                     float* group_weights, const bm4d_gpu::Parameters params, int gather_stacks_sum,
                     const cudaDeviceProp& d_prop);

void debug_kernel(float* tmp);

float normal_pdf_sqr(float std, float x);

void visualize_mask(float* mask, int k);

void load_noisy_3d(float tmp);

size_t checkGpuMem();
