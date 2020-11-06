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
// #include <cuda_runtime.h>
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

        checkCudaErrors(cudaMalloc((void**)&d_noisy_volume, sizeof(uchar) * size));

        // uint3float1* tmp_arr = new uint3float1[params.maxN*tsize];
        checkCudaErrors(cudaMalloc((void**)&d_stacks, sizeof(rotateRef) * (params.maxN * tsize)));

        checkCudaErrors(cudaMalloc((void**)&d_nstacks, sizeof(uint) * (tsize)));
        checkCudaErrors(cudaMemset(d_nstacks, 0, sizeof(uint) * tsize));
        std::cout << "Allocated " << (tsize) << " elements for d_nstacks" << std::endl;
        size_t free_mem = checkGpuMem();

        init_masks();

        // Allocate space for cusoft
        // TODO: Change after finalize fft shift algorithm
        bwIn = 8;
        bwOut = 8;
        degLim = 7;
        
        sig_n = bwIn * 2;
        sigpatSig_bsize = sig_n * sig_n;
        wsp1_bsize = 16*bwOut*bwOut*bwOut;
        wsp2_bsize = (14*bwIn*bwIn) + (48 * bwIn);
        sigpatCoef_bsize = bwIn * bwIn;
        so3Coef_bsize = (4*bwOut*bwOut*bwOut-bwOut)/3;
        so3Sig_bsize = 8 * bwOut * bwOut * bwOut;
        SNTspace_bsize = k_Reduced_Naive_TableSize(bwIn,bwIn) + k_Reduced_SpharmonicTableSize(bwIn,bwIn); // seminaive_naive_table
        SNT_bsize = (bwIn + 1);
        cos_even_bsize = bwIn;

        int soft_mem_size = sizeof(double) * (
            wsp1_bsize
            + wsp2_bsize
            + sigpatSig_bsize * 4
            + sigpatCoef_bsize * 4
            + so3Coef_bsize * 2
            + so3Sig_bsize * 2
            + SNTspace_bsize
            + SNT_bsize
            + cos_even_bsize
            + psize * 2
        );

        free_mem = checkGpuMem();
        batchsizeZ = floor(float(free_mem) / float(soft_mem_size) * 0.9 / (theight * twidth));
        if (batchsizeZ > tdepth)
            batchsizeZ = tdepth;
        batchsize = batchsizeZ * theight * twidth;
        printf("automatic batchsize (depth) %d, batchsize %d, patch count %d\n", batchsizeZ, batchsize, tsize);
        
        checkCudaErrors(cudaMalloc( (void**)&d_ref_patchs, batchsize * sizeof(float) * psize ));
        checkCudaErrors(cudaMalloc( (void**)&d_cmp_patchs, batchsize * sizeof(float) * psize ));

        checkCudaErrors(cudaMalloc( (void**)&d_sigR, sizeof(double) * sigpatSig_bsize * batchsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_sigI, sizeof(double) * sigpatSig_bsize * batchsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_patR, sizeof(double) * sigpatSig_bsize * batchsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_patI, sizeof(double) * sigpatSig_bsize * batchsize ));

        checkCudaErrors(cudaMalloc( (void**)&d_so3SigR, batchsize * sizeof(double) * so3Sig_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_so3SigI, batchsize * sizeof(double) * so3Sig_bsize ));
        
        checkCudaErrors(cudaMalloc( (void**)&d_workspace1, batchsize * sizeof(double) * wsp1_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_workspace2, batchsize * sizeof(double) * wsp2_bsize ));
        
        checkCudaErrors(cudaMalloc( (void**)&d_sigCoefR, batchsize * sizeof(double) * sigpatCoef_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_sigCoefI, batchsize * sizeof(double) * sigpatCoef_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_patCoefR, batchsize * sizeof(double) * sigpatCoef_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_patCoefI, batchsize * sizeof(double) * sigpatCoef_bsize ));
        
        checkCudaErrors(cudaMalloc( (void**)&d_so3CoefR, batchsize * sizeof(double) * so3Coef_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_so3CoefI, batchsize * sizeof(double) * so3Coef_bsize ));
        
        checkCudaErrors(cudaMalloc( (void**)&d_seminaive_naive_tablespace, batchsize * sizeof(double) * SNTspace_bsize ));
        std::cout << "Allocated Memory to CUSOFT Workspaces" << std::endl;
        
        // Allocate space to CUSOFT lib Workspaces
        checkCudaErrors(cudaMalloc( (void**)&d_cos_even, batchsize * sizeof(double) * cos_even_bsize ));
        checkCudaErrors(cudaMalloc( (void**)&d_seminaive_naive_table, batchsize * sizeof(double) * SNT_bsize )); // TODO: should be (double*)?
        std::cout << "Pre-Allocated Memory to CUSOFT Workspaces (lib)" << std::endl;
        
        checkGpuMem();
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
        // TODO: Add garbage collection for other malloced arrays
        cudaDeviceReset();
    };
    
    

    void free_cusoft_workspace()
    {
        std::cout << "freeing memory by block matching and cusoft" << std::endl;
        checkCudaErrors(cudaFree(d_ref_patchs));
        checkCudaErrors(cudaFree(d_cmp_patchs));

        checkCudaErrors(cudaFree(d_sigR));
        checkCudaErrors(cudaFree(d_sigI));
        checkCudaErrors(cudaFree(d_patR));
        checkCudaErrors(cudaFree(d_patI));

        checkCudaErrors(cudaFree(d_so3SigR));
        checkCudaErrors(cudaFree(d_so3SigI));

        checkCudaErrors(cudaFree(d_workspace1));
        checkCudaErrors(cudaFree(d_workspace2));

        checkCudaErrors(cudaFree(d_sigCoefR));
        checkCudaErrors(cudaFree(d_sigCoefI));
        checkCudaErrors(cudaFree(d_patCoefR));
        checkCudaErrors(cudaFree(d_patCoefI));

        checkCudaErrors(cudaFree(d_so3CoefR));
        checkCudaErrors(cudaFree(d_so3CoefI));

        checkCudaErrors(cudaFree(d_seminaive_naive_tablespace));

        checkCudaErrors(cudaFree(d_cos_even));
        checkCudaErrors(cudaFree(d_seminaive_naive_table));
        checkGpuMem();
    }

    void init_masks();
    void dev_volume2stack();
    void load_3d_array();
    std::vector<unsigned char> run_first_step();

 private:
    // Main variables
    std::vector<uchar> noisy_volume;

    uchar* d_noisy_volume;
    cudaArray *d_noisy_volume_3d;
    float* d_ref_patchs;
    float* d_cmp_patchs;

    cudaSurfaceObject_t noisy_volume_3d_surf;
    cudaTextureObject_t noisy_volume_3d_tex;

    // Masking and Rotation
    float* d_rel_coords;
    float* d_maskGaussian;
    float* d_maskSphere;
    float* maskGaussian;
    float* maskSphere;

    // Device variables
    float* d_gathered4dstack;
    rotateRef* d_stacks;
    uint* d_nstacks;
    
    float* d_group_weights;
    int width, height, depth, size;
    int twidth, theight, tdepth, tsize;
    int psize;
    float pshift; //shift from geometric center to reference point

    // CUSOFT configurations
    int bwIn, bwOut;
    int sig_n;
    int degLim;
    int batchsize; // in case the memory space is not sufficient
    int batchsizeZ;

    // buffer for real time computation of fftshift
    double *d_sigR, *d_sigI;
    double *d_patR, *d_patI;

    int sigpatSig_bsize;
    
    // CUSOFT workspaces
    double *d_so3SigR, *d_so3SigI;
    double *d_workspace1, *d_workspace2;
    double *d_sigCoefR, *d_sigCoefI;
    double *d_patCoefR, *d_patCoefI;
    double *d_so3CoefR, *d_so3CoefI;
    double *d_seminaive_naive_tablespace;

    int wsp1_bsize, wsp2_bsize;
    int sigpatCoef_bsize;
    int so3Sig_bsize;
    int so3Coef_bsize;
    int SNTspace_bsize;

    // CUSOFT lib workspaces
    double *d_cos_even; //cos_even = (double *) malloc(sizeof(double) * bw);
    double **d_seminaive_naive_table; // seminaive_naive_table = (double **) malloc(sizeof(double) * (bw+1));
    int SNT_bsize; // size of d_seminaive_naive_table
    int cos_even_bsize;

    // Parameters for launching kernels
    dim3 block;
    dim3 grid;

    cudaDeviceProp d_prop;
    bm4d_gpu::Parameters params;
};
