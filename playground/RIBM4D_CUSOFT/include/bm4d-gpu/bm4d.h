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
#include <cusoft/cusoft_kernels.cuh>

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
        checkCudaErrors(cudaMemset(d_nstacks_rot, 0, sizeof(uint) * tsize));
        std::cout << "Allocated " << (tsize) << " elements for d_nstacks" << std::endl;

        init_masks();
        init_rot_coords();

        // Allocate space for cusoft
        // TODO: Change after finalize
        bw = 8;
        degLim = 8;
        fft_patch_size = bw * bw;
        size_t free_mem = checkGpuMem();

        checkCudaErrors(cudaMalloc( (void**)&d_fftCoefR, sizeof(double) * bw * bw * tsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_fftCoefI, sizeof(double) * bw * bw * tsize ));
        std::cout << "Allocated " << (fft_patch_size * tsize * sizeof(double) * 2) << " elements for shfft" << std::endl;

        int soft_mem_size = sizeof(double) * (24*bw + 2*bw*bw)
                            + sizeof(double) * (16*bw*bw*bw)
                            + 2 * sizeof(double) * (8*bw*bw*bw)
                            + 2 * sizeof(double) * ((4*bw*bw*bw-bw)/3)
                            + sizeof(double) * bw
                            + sizeof(double*) * (bw + 1);

        batchsizeX = floor(float(free_mem) / float(soft_mem_size) * 0.9 / (theight * tdepth));
        if (batchsizeX > twidth)
            batchsizeX = twidth;
        batchsize = batchsizeX * theight * tdepth;
        printf("automatic batchsize (width) %d, batchsize %d, patch count %d\n", batchsizeX, batchsize, tsize);
        // Allocate space to CUSOFT Workspaces
        checkCudaErrors(cudaMalloc( (void**)&d_rdata, batchsize * sizeof(double) * (8*bw*bw*bw) ));
        checkCudaErrors(cudaMalloc( (void**)&d_idata, batchsize * sizeof(double) * (8*bw*bw*bw) ));
        checkCudaErrors(cudaMalloc( (void**)&d_workspace1, batchsize * sizeof(double) * (16*bw*bw*bw) ));
        checkCudaErrors(cudaMalloc( (void**)&d_workspace2, batchsize * sizeof(double) * (24*bw + 2*bw*bw) ));
        checkCudaErrors(cudaMalloc( (void**)&d_so3CoefR, batchsize * sizeof(double) * ((4*bw*bw*bw-bw)/3) ));
        checkCudaErrors(cudaMalloc( (void**)&d_so3CoefI, batchsize * sizeof(double) * ((4*bw*bw*bw-bw)/3) ));
        std::cout << "Allocated Memory to CUSOFT Workspaces" << std::endl;
        
        // Allocate space to CUSOFT lib Workspaces
        checkCudaErrors(cudaMalloc( (void**)&d_cos_even, batchsize * sizeof(double) * bw ));
        checkCudaErrors(cudaMalloc( (void**)&d_seminaive_naive_table, batchsize * sizeof(double*) * (bw + 1) )); // TODO: should be (double*)?
        std::cout << "Allocated Memory to CUSOFT Workspaces (lib)" << std::endl;
        
        checkGpuMem();
        wsp1_bsize = 16*bw*bw*bw;
        wsp2_bsize = 24*bw + 2*bw*bw;
        ridata_bsize = 8*bw*bw*bw;
        so3Coef_bsize = (4*bw*bw*bw-bw)/3;
        SNT_bsize = (bw + 1);
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
        if (d_fftCoefI) {
            checkCudaErrors(cudaFree(d_fftCoefI));
        }
        if (d_fftCoefR) {
            checkCudaErrors(cudaFree(d_fftCoefR));
        }
        if (d_rdata) {
            checkCudaErrors(cudaFree(d_rdata));
        }
        if (d_idata) {
            checkCudaErrors(cudaFree(d_idata));
        }
        // TODO: Add garbage collection for other malloced arrays
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
        // visualize_mask(d_maskGaussian, params.patch_size);
        // visualize_mask(d_maskSphere, params.patch_size);
    };

    void init_rot_coords() {
        Stopwatch t_rot_coords(true);
        int k = params.patch_size;
        rel_coords = (float*) malloc(psize * 3 * sizeof(float));
        int d;
        for (int z = 0; z < k; ++z)
            for (int y = 0; y < k; ++y)
                for (int x = 0; x < k; ++x) {
                    d = x + y * k + z * k * k;
                    rel_coords[3 * d] = (float) x - pshift;
                    rel_coords[3 * d + 1] = (float) y - pshift;
                    rel_coords[3 * d + 2] = (float) z - pshift;
                }
        
        checkCudaErrors(cudaMalloc((void**)&d_rel_coords, sizeof(float) * 3 * psize));
        checkCudaErrors(cudaMemcpy((void*)d_rel_coords, (void*)rel_coords, sizeof(float) * 3 * psize, cudaMemcpyHostToDevice));
        t_rot_coords.stop(); std::cout<<"Initialize reference coords took: " << t_rot_coords.getSeconds() <<std::endl;
    }

    void load_3d_array();
    std::vector<unsigned char> run_first_step();

 private:
    // Main variables
    std::vector<uchar> noisy_volume;
    cudaArray *d_noisy_volume_3d;
    cudaSurfaceObject_t noisy_volume_3d_surf;
    cudaTextureObject_t noisy_volume_3d_tex;

    // Masking and Rotation
    float* d_rel_coords;
    float* d_maskGaussian;
    float* d_maskSphere;
    float* rel_coords;
    float* maskGaussian;
    float* maskSphere;

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

    // CUSOFT configurations
    int bw;
    int degLim;
    int batchsize; // in case the memory space is not sufficient
    int batchsizeX;

    // Previously Computed FFT arrays (Spherical Harmonics Representation)
    double *d_fftCoefR, *d_fftCoefI;
    int fft_patch_size;
    
    // CUSOFT workspaces
    double *d_so3CoefR, *d_so3CoefI;
    double *d_rdata, *d_idata;
    double *d_workspace1, *d_workspace2;
    int wsp1_bsize, wsp2_bsize;
    int ridata_bsize;
    int so3Coef_bsize;

    // CUSOFT lib workspaces
    double *d_cos_even; //cos_even = (double *) malloc(sizeof(double) * bw);
    double **d_seminaive_naive_table; // seminaive_naive_table = (double **) malloc(sizeof(double) * (bw+1));
    int SNT_bsize; // size of d_seminaive_naive_table
    // Parameters for launching kernels
    dim3 block;
    dim3 grid;

    cudaDeviceProp d_prop;
    bm4d_gpu::Parameters params;
};
