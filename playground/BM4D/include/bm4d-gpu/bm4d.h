/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 */
#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

#include <bm4d-gpu/parameters.h>
#include <bm4d-gpu/kernels.cuh>
#include <bm4d-gpu/stopwatch.hpp>

class BM4D {
public:
    BM4D(bm4d_gpu::Parameters p, const std::vector<uchar>& in_noisy_volume, const int& width, const int& height, const int& depth)
            : params(p),
              width(width),
              height(height),
              depth(depth),
              d_gathered4dstack(NULL),
              d_stacks(NULL),
              d_nstacks(NULL) 
    {
        noisy_volume = in_noisy_volume;
        size = width * height * depth;
        psize = p.patch_size * p.patch_size * p.patch_size;
        pshift = ((float) p.patch_size - 1.0) / 2.0;
        int device;
        checkCudaErrors(cudaGetDevice(&device));
        checkCudaErrors(cudaGetDeviceProperties(&d_prop, device));

        twidth = std::floor((width - 1) / params.step_size + 1);
        theight = std::floor((height - 1) / params.step_size + 1);
        tdepth = std::floor((depth - 1) / params.step_size + 1);
        tsize = twidth * theight * tdepth;

        // TODO: Need change
        fft_patch_size = psize;
        
        // Copy noisy input array to the cuda device
        checkCudaErrors(cudaMalloc((void**)&d_noisy_volume, sizeof(uchar) * size));
        checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)noisy_volume.data(), sizeof(uchar) * size, cudaMemcpyHostToDevice));

        // allocating space for putting the input in patches (for fft computation)
        checkCudaErrors(cudaMalloc((void**)&d_noisy_stacks, sizeof(float) * tsize * psize));

        // uint3float1* tmp_arr = new uint3float1[params.maxN*tsize];
        checkCudaErrors(cudaMalloc((void**)&d_stacks, sizeof(uint3float1) * (params.maxN * tsize)));
        checkCudaErrors(cudaMalloc((void**)&d_stacks_rot, sizeof(rotateRef) * (params.maxN * tsize)));
        // std::cout << "Allocated " << sizeof(uint3float1)*(params.maxN*tsize) << "
        // bytes for d_stacks" << std::endl; checkCudaErrors(cudaMemcpy(d_stacks,
        // tmp_arr, sizeof(uint3float1)*params.maxN*tsize, cudaMemcpyHostToDevice));
        // delete[] tmp_arr;

        checkCudaErrors(cudaMalloc((void**)&d_nstacks, sizeof(uint) * (tsize)));
        checkCudaErrors(cudaMalloc((void**)&d_nstacks_rot, sizeof(uint) * (tsize)));
        checkCudaErrors(cudaMemset(d_nstacks, 0, sizeof(uint) * tsize));
        std::cout << "Allocated " << (tsize) << " elements for d_nstacks" << std::endl;

        checkCudaErrors(cudaMalloc((void**)&d_shfft_res, fft_patch_size * tsize * sizeof(float)));
        std::cout << "Allocated " << (fft_patch_size * tsize) << " elements for shfft" << std::endl;

        init_masks();
    }

    inline ~BM4D() {
        // Cleanup
        if (d_stacks) {
            checkCudaErrors(cudaFree(d_stacks));
            // std::cout << "Cleaned up d_stacks" << std::endl;
        }
        if (d_nstacks) {
            checkCudaErrors(cudaFree(d_nstacks));
            // std::cout << "Cleaned up d_nstacks" << std::endl;
        }
        if (d_gathered4dstack) {
            checkCudaErrors(cudaFree(d_gathered4dstack));
            // std::cout << "Cleaned up bytes of d_gathered4dstack" << std::endl;
        }
        if (d_shfft_res) {
            checkCudaErrors(cudaFree(d_shfft_res));
        }
        cudaDeviceReset();
    };
    
    void init_masks() {
        float std = pshift * 0.75; // std of gaussian TODO: modify this constant
        float sphere_tol = (pshift + 0.25) * (pshift + 0.25); // max distance to be included in the sphere
        int k = params.patch_size;

        Stopwatch t_init_mask(true);
        maskGaussian = (float*) malloc(psize * sizeof(float));
        maskSphere = (float*) malloc(psize * sizeof(float));
        // Odd size
        float dx, dy, dz, sqr_dist;
        int d;
        for (int z = 0; z < k; ++z)
            for (int y = 0; y < k; ++y)
                for (int x = 0; x < k; ++x) {
                    d = x + y * k + z * k * k;
                    dx = float(x) - pshift;
                    dy = float(y) - pshift;
                    dz = float(z) - pshift;
                    sqr_dist = dx*dx + dy*dy + dz*dz;
                    // Gaussian
                    maskGaussian[d] = normal_pdf_sqr(std, sqr_dist);
                    // Sphere
                    if (sqr_dist <= sphere_tol) maskSphere[d] = 1.0;
                    else maskSphere[d] = 0.0;
                }
        
        checkCudaErrors(cudaMalloc((void**)&d_maskGaussian, sizeof(float) * psize));
        checkCudaErrors(cudaMemcpy((void*)d_maskGaussian, (void*)maskGaussian, sizeof(float) * psize, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**)&d_maskSphere, sizeof(float) * psize));
        checkCudaErrors(cudaMemcpy((void*)d_maskSphere, (void*)maskSphere, sizeof(float) * psize, cudaMemcpyHostToDevice));
        
        t_init_mask.stop(); std::cout<<"Initialize masks took: " << t_init_mask.getSeconds() <<std::endl;
        visualize_mask(d_maskGaussian, params.patch_size);
        visualize_mask(d_maskSphere, params.patch_size);
    };

    void load_3d_array();
    void copy_to_stacks();
    std::vector<unsigned char> run_first_step();

 private:
    // Main variables
    std::vector<uchar> noisy_volume;
    cudaArray *d_noisy_volume_3d;
    cudaSurfaceObject_t noisy_volume_3d_surf;
    cudaTextureObject_t noisy_volume_3d_tex;
    uchar* d_noisy_volume;
    float* d_noisy_stacks;

    // Masking and Rotation
    float* d_rel_coords;
    float* d_maskGaussian;
    float* d_maskSphere;
    float* maskGaussian;
    float* maskSphere;

    // Previously Computed FFT arrays (Spherical Harmonics Representation)
    // float** shfftPatches;
    float* d_shfft_res;
    int fft_patch_size;

    // Device variables
    float* d_gathered4dstack;
    uint3float1* d_stacks;
    rotateRef* d_stacks_rot;
    uint* d_nstacks;
    uint* d_nstacks_rot;
    float* d_group_weights;
    int width, height, depth, size;
    int twidth, theight, tdepth, tsize;
    int psize;
    float pshift; //shift from geometric center to reference point

    // Parameters for launching kernels
    dim3 block;
    dim3 grid;

    cudaDeviceProp d_prop;
    bm4d_gpu::Parameters params;
};
