/*
* 2016, Vladislav Tananaev
* v.d.tananaev [at] gmail [dot] com
*/
#include <bm4d-gpu/kernels.cuh>
#include <bm4d-gpu/fft_bm4d_tools.cuh>
#include <cusoft/cusoft_kernels.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <math.h>
// #include "cublas_v2.h"

texture<uchar, 3, cudaReadModeNormalizedFloat> noisy_volume_3d_tex;

void bind_texture(cudaArray* d_noisy_volume_3d) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    noisy_volume_3d_tex.normalized = false;                     // access with normalized texture coordinates
    noisy_volume_3d_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    noisy_volume_3d_tex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    noisy_volume_3d_tex.addressMode[1] = cudaAddressModeWrap;
    noisy_volume_3d_tex.addressMode[2] = cudaAddressModeWrap;

    // --- Bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(noisy_volume_3d_tex, d_noisy_volume_3d, channelDesc));
    std::cout << "Texture Memory Binded" << std::endl;
    cudaDeviceSynchronize();
}

float normal_pdf_sqr(float std, float x) {
    // PDF of zero-mean gaussian (normalized)
    // input x: square of distance
    float xh = sqrt(x) / std;
    return exp(- (xh * xh) / 2.0);
}

// Get GPU RAM info
size_t checkGpuMem()
{
    float free_m,total_m,used_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(float)free_t/1048576.0 ;
    total_m=(float)total_t/1048576.0;
    used_m=total_m-free_m;
    printf ( "=== GPU RAM ===\n mem free  %lu \t- %f MB\n mem total %lu \t- %f MB\n mem used %f MB\n===============\n\n",free_t,free_m,total_t,total_m,used_m);
    return free_t;
}

// For allocating Spharmonic Tables
int k_Reduced_Naive_TableSize(int bw, int m)
{
  int i, sum;
  sum = 0;
  for (i=m; i<bw; i++)
    sum += ( 2 * bw * (bw - i));
  return sum;
}

int k_TableSize(int m, int bw)
{
    return (((bw/2) * ((bw/2) + 1)) - ((m/2)*((m/2)+1)) - ((bw/2) * (m % 2)));
}

int k_Reduced_SpharmonicTableSize(int bw, int m)
{
  int i, sum;
  sum = 0;
  for (i=0; i<m; i++)
    sum += k_TableSize(i,bw);
  return sum;
}

// ==========================================================================================
// ===================================== Copy image to stack ================================
// ==========================================================================================

// Copy the data to stacks (which are used for inplace fftshift)
// TODO: remove this if not needed in the end
__global__ void k_d_volume2stack(const uchar* __restrict d_noisy_volume,
                                 float* d_noisy_stacks,
                                 const uint3 size,
                                 const uint3 tshape,
                                 const bm4d_gpu::Parameters params)
{
    int psize;
    int vind, pind, pind_base;
    int px, py, pz;
    int vx, vy, vz;
    int k = params.patch_size;
    psize = k * k * k;
    for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tshape.z; Idz += blockDim.z*gridDim.z)
        for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tshape.y; Idy += blockDim.y*gridDim.y)
            for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tshape.x; Idx += blockDim.x*gridDim.x)
            {
                pind_base = Idx + Idy * tshape.x + Idz * tshape.x * tshape.y;
                // printf("%d\n", pind_base);
                
                int x = Idx * params.step_size;
                int y = Idy * params.step_size;
                int z = Idz * params.step_size;
                if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
                    return;
                for (pz = 0; pz < k; ++pz)
                    for (py = 0; py < k; ++py)
                        for (px = 0; px < k; ++px){
                            vx = max(0, min(x + px, size.x - 1));
                            vy = max(0, min(y + py, size.y - 1));
                            vz = max(0, min(z + pz, size.z - 1));
                            vind = vx + vy * size.x + vz * size.x * size.y;
                            pind = pind_base * psize + (px + py * k + pz * k * k);
                            d_noisy_stacks[pind] = (float) d_noisy_volume[vind];
                        }
            }
}

__global__ void k_d_volume2stack_debug(const uchar* __restrict d_noisy_volume,
    float* d_noisy_stacks,
    const uint3 size,
    const uint3 tshape,
    const bm4d_gpu::Parameters params)
{
    int k = params.patch_size;
    int psize = k * k * k;
    int vind, pind;
    int px, py, pz;
    for (int i = 0; i < 4; i++) {
        printf("Checking patch (%d, %d, %d)\n", i, i, i);
        for (pz = 0; pz < k; ++pz)
            for (py = 0; py < k; ++py)
                for (px = 0; px < k; ++px) {
                    vind = (px + i * params.step_size) + (py + i * params.step_size) * size.x + (pz + i * params.step_size) * size.x * size.y;
                    pind = i * (1 + tshape.x + tshape.x * tshape.y) * psize + px + py * k + pz * k * k;
                    printf("(%d, %d, %d): %d, %f\n", px, py, pz, d_noisy_volume[vind], d_noisy_stacks[pind]);
                }
    }
}

void d_volume2stack(uchar* d_noisy_volume, float* d_noisy_stacks, const uint3 size, const uint3 tshape, const bm4d_gpu::Parameters params, const cudaDeviceProp& d_prop)
{
    int threads = std::floor(sqrt(d_prop.maxThreadsPerBlock));
    dim3 block(threads, threads, 1);
    int bs_x = d_prop.maxGridSize[1] < tshape.x ? d_prop.maxGridSize[1] : tshape.x;
    int bs_y = d_prop.maxGridSize[1] < tshape.y ? d_prop.maxGridSize[1] : tshape.y;
    dim3 grid(bs_x, bs_y, 1);

    // Debug verification
    std::cout << "Copying all patches to stack (for fftshift) " << (tshape.x*tshape.y*tshape.z) << std::endl;

    k_d_volume2stack <<< grid, block >>>(d_noisy_volume,
                                         d_noisy_stacks,
                                         size,
                                         tshape,
                                         params);
    cudaDeviceSynchronize();
}

// =================== creating masks ========================
__global__ void vis_mask(float* mask, int k) {
    int d;
    for (int z = 0; z < k; ++z)
        for (int y = 0; y < k; ++y)
            for (int x = 0; x < k; ++x) {
                d = x + y * k + z * k * k;
                printf("%4f ", mask[d]);
                if (x == k-1 && y == k-1) printf("\nlayer%d\n", z);
                if (x == k-1) printf("\n");
            }
}
void visualize_mask(float* mask, int k) {
    vis_mask <<< 1, 1 >>> (mask, k);
}

// Make rotate reference object
__device__ rotateRef make_rotateRef_sim(uint x, uint y, uint z, float4 sim) {
    rotateRef rrf;
    rrf.x = x;
    rrf.y = y;
    rrf.z = z;
    rrf.val = sim.w;
    
    // we are referencing the coordinates, so apply the inverse rotation
    rrf.alpha = -sim.z;
    rrf.beta = -sim.y;
    rrf.gamma = -sim.x;
    // printf("%f, %f, %f\n", rrf.alpha, rrf.beta, rrf.gamma);

    // Compute rotation matrix
    float sa = sin(rrf.alpha), ca = cos(rrf.alpha);
    float sb = sin(rrf.beta), cb = cos(rrf.beta);
    float sg = sin(rrf.gamma), cg = cos(rrf.gamma);
    
    // Calculating the ZYZ rotation matrix
    rrf.v1x = ca * cb * cg - sa * sg;
    rrf.v1y = sa * cb * cg + ca * sg;
    rrf.v1z = -sb * cg;

    rrf.v2x = -ca * cb * sg - sa * cg;
    rrf.v2y = -sa * cb * sg + ca * cg;
    rrf.v2z = sb * sg;

    rrf.v3x = ca * sb;
    rrf.v3y = sa * sb;
    rrf.v3z = cb;

    return rrf;
}

// Make the inverse rrf object
__device__ rotateRef invert_rotateRef_sim(rotateRef rrf) {
    rotateRef rrf_inv;
    rrf_inv.x = rrf.x;
    rrf_inv.y = rrf.y;
    rrf_inv.z = rrf.z;
    rrf_inv.val = rrf.val;

    rrf_inv.alpha = -rrf.gamma;
    rrf_inv.beta  = -rrf.beta;
    rrf_inv.gamma = -rrf.alpha;

    // Compute rotation matrix
    float sa = sin(rrf_inv.alpha), ca = cos(rrf_inv.alpha);
    float sb = sin(rrf_inv.beta),  cb = cos(rrf_inv.beta);
    float sg = sin(rrf_inv.gamma), cg = cos(rrf_inv.gamma);
    
    // Calculating the ZYZ rotation matrix
    rrf_inv.v1x = ca * cb * cg - sa * sg;
    rrf_inv.v1y = sa * cb * cg + ca * sg;
    rrf_inv.v1z = -sb * cg;

    rrf_inv.v2x = -ca * cb * sg - sa * cg;
    rrf_inv.v2y = -sa * cb * sg + ca * cg;
    rrf_inv.v2z = sb * sg;

    rrf_inv.v3x = ca * sb;
    rrf_inv.v3y = sa * sb;
    rrf_inv.v3z = cb;

    return rrf_inv;
}


// inference within patch after rotation
__device__ float rotTex3D(rotateRef rrf, int px, int py, int pz, float pshift) {
    // must be called after the texture memory is binded with the noisy array.
    float fpx = (float)px - pshift;
    float fpy = (float)py - pshift;
    float fpz = (float)pz - pshift;
    
    float rpx = fpx * rrf.v1x + fpy * rrf.v2x + fpz * rrf.v3x + pshift + rrf.x;
    float rpy = fpx * rrf.v1y + fpy * rrf.v2y + fpz * rrf.v3y + pshift + rrf.y;
    float rpz = fpx * rrf.v1z + fpy * rrf.v2z + fpz * rrf.v3z + pshift + rrf.z;

    return tex3D(noisy_volume_3d_tex, rpx + 0.5f, rpy + 0.5f, rpz + 0.5f) * 255;
}


// ==========================================================================================
// =================================== compute fft shift ====================================
// ==========================================================================================

__device__ void DEBUG_show_spherical_buf(double* buf, int bw) {
    printf("start:\n");
    for (int i = 0; i < bw * 2; i++) {
        printf("[");
        for (int j = 0; j < bw * 2; j++) {
            printf("%lf, ", buf[2*bw*i + j]);
        }
        printf("]\n");
    }
    printf("\n");
}

__device__ void DEBUG_show_fft_res(float* data) {
    printf("fft: ");
    for (int i = 0; i < 64; i++) {
        printf("%f, ", data[i]);
    }
    printf("\n");
}

__device__ void DEBUG_show_buf(float* buf) {
    printf("buf: ");
    for (int i = 0; i < 16; i++) {
        printf("%f, ", buf[i]);
    }
    printf("\n");
}

// Update the bufferR and bufferI with the result computed based on patch (of size k*k*k)
__device__ void fft_shift_3d(float* patch, float* patch_im, float* patch_buf_re, float* patch_buf_im, double* bufR, double* bufI, int k, int bw)
{
    //printf("one fft-----------------------------------\n");
    struct fft_bm4d_tools fft_tools(k, bw);
    //DEBUG_show_fft_res(patch);
    //DEBUG_show_fft_res(patch_im);
    fft_tools.ifftshift_3d_in(patch, patch_im);
    //DEBUG_show_fft_res(patch);
    //DEBUG_show_fft_res(patch_im);
    fft_tools.fft_3d(patch, patch_im, patch_buf_re, patch_buf_im);
    // printf("Done2\n");
    //DEBUG_show_fft_res(patch);
    //DEBUG_show_fft_res(patch_im);
    fft_tools.fftshift_3d_in(patch, patch_im);
    //DEBUG_show_fft_res(patch);
    fft_tools.complex_abs(patch, patch_im);
    //DEBUG_show_fft_res(patch);
    fft_tools.spherical_mapping(bufR, bufI, patch);
    //DEBUG_show_spherical_buf(bufR, bw);
    //DEBUG_show_fft_res(patch);
    //printf("end-----------------------------------\n");
    return;
}

// ==========================================================================================
// ==================================== Debugging for stacks ================================
// ==========================================================================================

__global__ void k_debug_lookup_stacks(uint3float1 * d_stacks, int total_elements){
    int a = 345;
    for (int i = 0; i < 150; ++i){
        a += i;
        printf("%i: %d %d %d %f\n", i, d_stacks[i].x, d_stacks[i].y, d_stacks[i].z, d_stacks[i].val);
    }
}
__global__ void k_debug_lookup_4dgathered_stack(float* gathered_stack4d){
    for (int i = 0; i < 64 * 3; ++i){
        if (!(i % 4)) printf("\n");
        if (!(i % 16)) printf("------------\n");
        if (!(i % 64)) printf("------------\n");
        printf("%f ", gathered_stack4d[i]);
    }
}
__global__ void k_debug_lookup_int(int* gathered_stack4d){
    for (int i = 0; i < 64 * 3; ++i){
        if (!(i % 4)) printf("\n");
        if (!(i % 16)) printf("------------\n");
        if (!(i % 64)) printf("------------\n");
        printf("%d ", gathered_stack4d[i]);
    }
}
void debug_kernel_int(int* tmp){
    k_debug_lookup_int <<< 1, 1 >>>(tmp);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
void debug_kernel(float* tmp){
    k_debug_lookup_4dgathered_stack <<< 1, 1 >>>(tmp);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// Nearest lower power of 2
__device__ __inline__ uint flp2(uint x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

__global__ void k_check_texture_sync(const uchar* d_noisy_volume, uint3 imshape, int patchsize)
{
    float diff = 0.0;
    float pshift = ((float) patchsize - 1.0) / 2.0;
    for (int x = 0; x < imshape.x; x++)
        for (int y = 0; y < imshape.y-1; y++)
            for (int z = 0; z < imshape.z-2; z++) {
                float4 trivial_angle = make_float4(0., 0., 0., 0.);
                rotateRef rrf = make_rotateRef_sim(x, y, z, trivial_angle);
                int vind = (x) + (y+1) * imshape.x + (z+2) * imshape.x * imshape.y;
                diff += abs((float) d_noisy_volume[vind] - rotTex3D(rrf, 0, 1, 2, pshift));
            }
    printf("average difference in synchronization: %f\n", diff / (imshape.x * imshape.y * imshape.z));
}

void check_texture_sync(const uchar* d_noisy_volume, cudaArray* d_noisy_volume_3d, uint3 imshape, int patchsize)
{
    bind_texture(d_noisy_volume_3d);
    k_check_texture_sync <<< 1, 1 >>> (d_noisy_volume, imshape, patchsize);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// ==========================================================================================
// ======================================= Rotation Matching ================================
// ==========================================================================================

__device__ void clear_patch(float* patch, int n) {
    for (int i = 0; i < n * n * n; i++)
        patch[i] = 0.0;
}

__device__ void fill_patch(const uchar* __restrict d_noisy_volume, float* patch, const uint3 size, const uint3 coord, int k)
{
    //memset(patch, 0, k * k * k * sizeof(float));
    clear_patch(patch, k);
    int pind, vind;
    int px, py, pz;
    int vx, vy, vz;
    for (pz = 0; pz < k; ++pz)
        for (py = 0; py < k; ++py)
            for (px = 0; px < k; ++px){
                vx = max(0, min(coord.x + px, size.x - 1));
                vy = max(0, min(coord.y + py, size.y - 1));
                vz = max(0, min(coord.z + pz, size.z - 1));
                vind = vx + vy * size.x + vz * size.x * size.y;
                pind = px + py * k + pz * k * k;
                patch[pind] = (float) d_noisy_volume[vind];
            }
}

__device__ void apply_mask(float* patch, float* mask, int k)
{
    for (int i = 0; i < k * k * k; i++) {
        patch[i] *= mask[i];
    }
}

__device__ void add_stack_rot(rotateRef* d_stack_rot,
                              uint* d_nstack_rot, // keep track of current stack size
                              const rotateRef newval,
                              const int maxN)
{
    int k;
    uint num = (*d_nstack_rot);
    if (num < maxN) //add new value regardless
    {
        k = num++;
        while (k > 0 && newval.val > d_stack_rot[k-1].val)
        {
            d_stack_rot[k] = d_stack_rot[k - 1];
            --k;
        }
        d_stack_rot[k] = newval;
        *d_nstack_rot = num; // update stack count
    }
    else if (newval.val >= d_stack_rot[0].val) return;
    else //delete highest value and add new
    {
        k = 1;
        while (k < maxN && newval.val < d_stack_rot[k].val)
        {
            d_stack_rot[k - 1] = d_stack_rot[k];
            k++;
        }
        d_stack_rot[k - 1] = newval;
    }
}

__device__ rotateRef dist_rot(const uchar* __restrict d_noisy_volume, // the input patch
    const float* __restrict mask,
    const uint3 sig_ref, // reference point of the signal patch (base)
    const uint3 pat_ref, // reference point of the pattern patch (cmp)
    const uint3 size, // shape of image
    const int patch_width, // width of the patch
    double *sigR, double *sigI, // fft results of ref
    double *patR, double *patI, // fft results of cmp
    double *so3SigR, double *so3SigI, // workspaces for cusoft
    double *workspace1, double *workspace2,
    double *sigCoefR, double *sigCoefI,
    double *patCoefR, double *patCoefI,
    double *so3CoefR, double *so3CoefI,
    double *seminaive_naive_tablespace,
    double *cos_even,
    double **seminaive_naive_table,
    int bwIn, int bwOut, int degLim) // configuration of cusoft
{
    float4 soft_rot = soft_corr(sigR, sigI,
        patR, patI,
        so3SigR, so3SigI,
        workspace1, workspace2,
        sigCoefR, sigCoefI,
        patCoefR, patCoefI,
        so3CoefR, so3CoefI,
        seminaive_naive_tablespace,
        cos_even,
        seminaive_naive_table,
        bwIn, bwOut, degLim); // rotation for cmp (pattern) to match ref (signal)
        
    rotateRef cmpref = make_rotateRef_sim(pat_ref.x, pat_ref.y, pat_ref.z, soft_rot);
    // rotateRef cmpref = make_rotateRef_sim(pat_ref.x, pat_ref.y, pat_ref.z, make_float4(0, 0, 0, 0));

    // signal is the base patch, pattern is the new patch to be rotated.
    int k = patch_width;
    float pshift = (float(k) - 1) / 2.0;
    // printf("sigref: %d, %d, %d\n", sig_ref.x, sig_ref.y, sig_ref.z);
    // printf("patref: %d, %d, %d\n", pat_ref.x, pat_ref.y, pat_ref.z);

    int sig_vind;
    float diff = 0.0;
    for (int pz = 0; pz < k; ++pz)
        for (int py = 0; py < k; ++py)
            for (int px = 0; px < k; ++px) {
                // pind = px + py * k + pz * k * k;
                sig_vind = (px + sig_ref.x) + (py + sig_ref.y) * size.x + (pz + sig_ref.z) * size.x * size.y;
                float diff_pix = (float) d_noisy_volume[sig_vind] - rotTex3D(cmpref, px, py, pz, pshift);
                // diff_pix *= mask[pind];
                diff += diff_pix * diff_pix;
            }
    cmpref.val = diff / (k*k*k);
    return cmpref;
}

__global__ void k_block_matching_rot(const uchar* __restrict d_noisy_volume,
                                     const uint3 size,
                                     const uint3 tshape,
                                     int params_patch_size,
                                     int params_window_size,
                                     int params_step_size,
                                     int params_maxN,
                                     float params_sim_th,
                                     rotateRef* d_stacks_rot,
                                     uint* d_nstacks_rot,
                                     int batchsizeZ,
                                     float *mask_Gaussian,
                                     float *mask_Sphere,
                                     float *d_ref_patchs,
                                     float *d_cmp_patchs,
                                     float *d_patchs_im,
                                     float *d_buf_re,
                                     float *d_buf_im,
                                     double *d_sigR, double *d_sigI,
                                     double *d_patR, double *d_patI,
                                     double *d_so3SigR, double *d_so3SigI,
                                     double *d_workspace1, double *d_workspace2,
                                     double *d_sigCoefR, double *d_sigCoefI,
                                     double *d_patCoefR, double *d_patCoefI,
                                     double *d_so3CoefR, double *d_so3CoefI,
                                     double *d_seminaive_naive_tablespace,
                                     double *d_cos_even,
                                     double **d_seminaive_naive_table,
                                     int bwIn,
                                     int SNTspace_bsize)
{   
    int bwOut = bwIn;
    int degLim = bwIn - 1;
    
    int sig_n = bwIn * 2;
    int sigpatSig_bsize = sig_n * sig_n;
    int wsp1_bsize = 16*bwOut*bwOut*bwOut;
    int wsp2_bsize = (14*bwIn*bwIn) + (48 * bwIn);
    int sigpatCoef_bsize = bwIn * bwIn;
    int so3Coef_bsize = (4*bwOut*bwOut*bwOut-bwOut)/3;
    int so3Sig_bsize = 8 * bwOut * bwOut * bwOut;
    int SNT_bsize = (bwIn + 1);
    int cos_even_bsize = bwIn;
    int tsize = tshape.x * tshape.y * tshape.z;
    int log_offset = tsize / 20 + 1;

    int tx_ref, ty_ref, tz_ref, vx_ref, vy_ref, vz_ref;
    int vx_cmp, vy_cmp, vz_cmp;
    int tz_batchind;

    int patchsize = params_patch_size;
    int patch_bsize = patchsize * patchsize * patchsize; // size of the patch
    int patch_buf_bsize = patchsize * patchsize;

    for (tz_batchind = 0; tz_batchind < tshape.z; tz_batchind += batchsizeZ) {
        // printf("Processing %d layers among %d total\n", tz_batchind + batchsizeZ, tshape.z);
        for (tz_ref = tz_batchind + blockDim.z * blockIdx.z + threadIdx.z; tz_ref < min(tshape.z, batchsizeZ + tz_batchind); tz_ref += blockDim.z*gridDim.z)
            for (ty_ref = blockDim.y * blockIdx.y + threadIdx.y; ty_ref < tshape.y; ty_ref += blockDim.y*gridDim.y)
                for (tx_ref = blockDim.x * blockIdx.x + threadIdx.x; tx_ref < tshape.x; tx_ref += blockDim.x*gridDim.x) {
                    
                    vx_ref = tx_ref * params_step_size;
                    vy_ref = ty_ref * params_step_size;
                    vz_ref = tz_ref * params_step_size;

                    
                    if (vx_ref >= size.x || vy_ref >= size.y || vz_ref >= size.z || vx_ref < 0 || vy_ref < 0 || vz_ref < 0)
                        return;
                    
                    int tz_offset = tz_ref - tz_batchind;
                    int mem_offset = tz_offset * tshape.x * tshape.y + ty_ref * tshape.x + tx_ref; // memory location in the batch buffer
                    // if (mem_offset > 10) return;
                    // construct the reference patch
                    uint3 ref_coord = make_uint3(vx_ref, vy_ref, vz_ref);

                    float* ref_patch = &d_ref_patchs[mem_offset * patch_bsize];
                    double* sigR = &d_sigR[mem_offset * sigpatSig_bsize];
                    double* sigI = &d_sigI[mem_offset * sigpatSig_bsize];
                    
                    // pre-allocated workspace for cmp patch and fftshift of it
                    float* cmp_patch = &d_cmp_patchs[mem_offset * patch_bsize];
                    double* patR = &d_patR[mem_offset * sigpatSig_bsize];
                    double* patI = &d_patI[mem_offset * sigpatSig_bsize];

                    // pre-allocated workspace for buffers used in fft
                    float* patch_im = &d_patchs_im[mem_offset * patch_bsize];
                    float* buf_re = &d_buf_re[mem_offset * patch_buf_bsize];
                    float* buf_im = &d_buf_im[mem_offset * patch_buf_bsize];
                    
                    // pre-allocated workspace for cusoft
                    double* so3SigR = &d_so3SigR[mem_offset * so3Sig_bsize];
                    double* so3SigI = &d_so3SigI[mem_offset * so3Sig_bsize];
                    double* workspace1 = &d_workspace1[mem_offset * wsp1_bsize];
                    double* workspace2 = &d_workspace2[mem_offset * wsp2_bsize];
                    double* sigCoefR = &d_sigCoefR[mem_offset * sigpatCoef_bsize];
                    double* sigCoefI = &d_sigCoefI[mem_offset * sigpatCoef_bsize];
                    double* patCoefR = &d_patCoefR[mem_offset * sigpatCoef_bsize];
                    double* patCoefI = &d_patCoefI[mem_offset * sigpatCoef_bsize];
                    double* so3CoefR = &d_so3CoefR[mem_offset * so3Coef_bsize];
                    double* so3CoefI = &d_so3CoefI[mem_offset * so3Coef_bsize];
                    double* seminaive_naive_tablespace = &d_seminaive_naive_tablespace[mem_offset * SNTspace_bsize];
                    double* cos_even = &d_cos_even[mem_offset * cos_even_bsize];
                    double** seminaive_naive_table = &d_seminaive_naive_table[mem_offset * SNT_bsize];
                    
                    fill_patch(d_noisy_volume, ref_patch, size, ref_coord, patchsize); // fill the reference patch
                    apply_mask(ref_patch, mask_Gaussian, patchsize);
                    clear_patch(patch_im, patchsize); // clear patch_im for future fft
                    fft_shift_3d(ref_patch, patch_im, buf_re, buf_im, sigR, sigI, patchsize, bwIn); // compute fftshift for the reference patch

                    // range of searching
                    int wxb = fmaxf(0, vx_ref - params_window_size); // window x begin
                    int wyb = fmaxf(0, vy_ref - params_window_size); // window y begin
                    int wzb = fmaxf(0, vz_ref - params_window_size); // window z begin
                    int wxe = fminf(size.x - params_patch_size, vx_ref + params_window_size); // window x end
                    int wye = fminf(size.y - params_patch_size, vy_ref + params_window_size); // window y end
                    int wze = fminf(size.z - params_patch_size, vz_ref + params_window_size); // window z end
                    if ((mem_offset + 1) % 1000 == 0 || mem_offset == 100)
                        printf("Begin: patch %d / %d\n", mem_offset + 1, tsize);
                    for (vz_cmp = wzb; vz_cmp <= wze; vz_cmp++)
                        for (vy_cmp = wyb; vy_cmp <= wye; vy_cmp++)
                            for (vx_cmp = wxb; vx_cmp <= wxe; vx_cmp++){
                                // printf("One run started: %d, %d, %d offset %d\n", tx_ref, ty_ref, tz_ref, mem_offset);
                                // printf("%d, %d, %d\n", vx_cmp, vy_cmp, vz_cmp);
                                uint3 cmp_coord = make_uint3(vx_cmp, vy_cmp, vz_cmp);
                                fill_patch(d_noisy_volume, cmp_patch, size, cmp_coord, patchsize); // fill the reference patch
                                apply_mask(cmp_patch, mask_Gaussian, patchsize);
                                clear_patch(patch_im, patchsize); // clear patch_im for future fft
                                fft_shift_3d(cmp_patch, patch_im, buf_re, buf_im, patR, patI, patchsize, bwIn); // compute fftshift for the reference patch
                                // printf("One run fft done: %d, %d, %d offset %d\n", tx_ref, ty_ref, tz_ref, mem_offset);

                                rotateRef rrf = dist_rot(d_noisy_volume, 
                                                      mask_Sphere,
                                                      ref_coord, 
                                                      cmp_coord,
                                                      size,
                                                      params_patch_size,
                                                      sigR, sigI,
                                                      patR, patI,
                                                      so3SigR, so3SigI,
                                                      workspace1, workspace2,
                                                      sigCoefR, sigCoefI,
                                                      patCoefR, patCoefI,
                                                      so3CoefR, so3CoefI,
                                                      seminaive_naive_tablespace,
                                                      cos_even,
                                                      seminaive_naive_table,
                                                      bwIn, bwOut, degLim);
                                // printf("One run disrot done: %d, %d, %d offset %d\n", tx_ref, ty_ref, tz_ref, mem_offset);
                                // printf("%f\n", rrf.val);
                                if (rrf.val < params_sim_th){
                                    // printf("Dist %f\n", rrf.val);
                                    add_stack_rot(&d_stacks_rot[(tx_ref + (ty_ref + tz_ref* tshape.y)*tshape.x)*params_maxN],
                                                &d_nstacks_rot[tx_ref + (ty_ref + tz_ref* tshape.y)*tshape.x],
                                                rrf,
                                                params_maxN);
                                }
                                // printf("One run finished: %d, %d, %d offset %d\n", tx_ref, ty_ref, tz_ref, mem_offset);
                            }

                    if ((mem_offset + 1) % 1000 == 0 || mem_offset == 100)
                        printf("Finished: patch %d / %d\n", mem_offset + 1, tsize);
                }
    }
    
}

void run_block_matching_rot(const uchar* __restrict d_noisy_volume,
                            const uint3 size,
                            const uint3 tshape,
                            const bm4d_gpu::Parameters params,
                            rotateRef* d_stacks_rot,
                            uint* d_nstacks_rot,
                            int batchsizeZ,
                            float *mask_Gaussian,
                            float *mask_Sphere,
                            float *d_ref_patchs,
                            float *d_cmp_patchs,
                            float *d_patchs_im,
                            float *d_buf_re,
                            float *d_buf_im,
                            double *d_sigR, double *d_sigI,
                            double *d_patR, double *d_patI,
                            double *d_so3SigR, double *d_so3SigI,
                            double *d_workspace1, double *d_workspace2,
                            double *d_sigCoefR, double *d_sigCoefI,
                            double *d_patCoefR, double *d_patCoefI,
                            double *d_so3CoefR, double *d_so3CoefI,
                            double *d_seminaive_naive_tablespace,
                            double *d_cos_even,
                            double **d_seminaive_naive_table,
                            int bwIn, int bwOut, int degLim,
                            int SNTspace_bsize,
                            const cudaDeviceProp &d_prop)
{
    int threads = std::floor(sqrt(d_prop.maxThreadsPerBlock));
    dim3 block(threads, threads, 1);
    int bs_x = d_prop.maxGridSize[1] < tshape.x ? d_prop.maxGridSize[1] : tshape.x;
    int bs_y = d_prop.maxGridSize[1] < tshape.y ? d_prop.maxGridSize[1] : tshape.y;
    // int bs_z = d_prop.maxGridSize[2] < batchsizeZ ? d_prop.maxGridSize[2] : batchsizeZ;
    dim3 grid(bs_x, bs_y, 1);

    // Debug verification
    std::cout << "Total number of reference patches " << (tshape.x*tshape.y*tshape.z) << std::endl;

    dim3 blocks_test(tshape.x, tshape.y, batchsizeZ);
    dim3 grid_test(1, 1, 1);
    k_block_matching_rot <<< blocks_test, grid_test >>>(d_noisy_volume,
                                            size,
                                            tshape,
                                            params.patch_size,
                                            params.window_size,
                                            params.step_size,
                                            params.maxN,
                                            params.sim_th,
                                            d_stacks_rot,
                                            d_nstacks_rot,
                                            batchsizeZ,
                                            mask_Gaussian,
                                            mask_Sphere,
                                            d_ref_patchs,
                                            d_cmp_patchs,
                                            d_patchs_im,
                                            d_buf_re,
                                            d_buf_im,
                                            d_sigR, d_sigI,
                                            d_patR, d_patI,
                                            d_so3SigR, d_so3SigI,
                                            d_workspace1, d_workspace2,
                                            d_sigCoefR, d_sigCoefI,
                                            d_patCoefR, d_patCoefI,
                                            d_so3CoefR, d_so3CoefI,
                                            d_seminaive_naive_tablespace,
                                            d_cos_even,
                                            d_seminaive_naive_table,
                                            bwIn,
                                            SNTspace_bsize);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// ==========================================================================================
// =================================== Rot Gather Cubes =====================================
// ==========================================================================================

__global__ void k_nstack_rot_to_pow(rotateRef* d_stacks_rot, uint* d_nstacks_rot, const int elements, const uint maxN){
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < elements; i += blockDim.x*gridDim.x){
        if (i >= elements) return;

        uint inGroupId = i % maxN;
        uint groupId = i/maxN;

        uint n = d_nstacks_rot[groupId];
        uint tmp = flp2(n);
        uint diff = d_nstacks_rot[groupId] - tmp;

        __syncthreads();
        d_nstacks_rot[groupId] = tmp;

        if (inGroupId < diff || inGroupId >= n)
            d_stacks_rot[i].val = -1;
    }
}

__global__ void k_gather_cubes_rot(cudaArray* d_noisy_volume_3d,
                               const uint3 size,
                               const bm4d_gpu::Parameters params,
                               const rotateRef* __restrict d_stacks_rot,
                               const uint array_size,
                               float* d_gathered4dstack_rot)
{
    int cube_size = params.patch_size*params.patch_size*params.patch_size;
    int pshift = ((float) params.patch_size - 1.0) / 2.0;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < array_size; i += blockDim.x*gridDim.x){
        if (i >= array_size) return;
        rotateRef rrf = d_stacks_rot[i];

        for (int z = 0; z < params.patch_size; ++z)
            for (int y = 0; y < params.patch_size; ++y)
                for (int x = 0; x < params.patch_size; ++x){
                    int stack_idx = i*cube_size + (x)+(y + z*params.patch_size)*params.patch_size;
                    d_gathered4dstack_rot[stack_idx] = rotTex3D(rrf, x, y, z, pshift);
                    // printf("%f\n", d_gathered4dstack_rot[stack_idx]);
                }
    }
}

struct is_not_empty_rot
{
    __host__ __device__
    bool operator()(const rotateRef x)
    {
        return (x.val != -1);
    }
};

void gather_cubes_rot(cudaArray* d_noisy_volume_3d,
                      const uint3 size,
                      const uint3 tshape,
                      const bm4d_gpu::Parameters params,
                      rotateRef* &d_stacks_rot,
                      uint* d_nstacks_rot,
                      float* &d_gathered4dstack_rot,
                      uint &gather_stacks_sum_rot,
                      const cudaDeviceProp &d_prop)
{
    // Convert all the numbers in d_nstacks to the lowest power of two
    bind_texture(d_noisy_volume_3d);
    uint array_size = (tshape.x*tshape.y*tshape.z);
    int threads = d_prop.maxThreadsPerBlock / 2;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(params.maxN*array_size / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(params.maxN*array_size / threads);
    if (bs_x == 0) bs_x++;

    printf("arraysize:%d, %d, %d\n", array_size, bs_x, threads);
    
    k_nstack_rot_to_pow <<< bs_x, threads >>>(d_stacks_rot, d_nstacks_rot, params.maxN*array_size, params.maxN);
    checkCudaErrors(cudaGetLastError());
    thrust::device_ptr<uint> dt_nstacks_rot = thrust::device_pointer_cast(d_nstacks_rot);
    gather_stacks_sum_rot = thrust::reduce(dt_nstacks_rot, dt_nstacks_rot + array_size);
    std::cout << "Sum of patches: "<< gather_stacks_sum_rot << std::endl;
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Make a compaction
    rotateRef * d_stacks_compacted_rot;
    checkCudaErrors(cudaMalloc((void**)&d_stacks_compacted_rot, sizeof(rotateRef)*gather_stacks_sum_rot));
    thrust::device_ptr<rotateRef> dt_stacks_rot = thrust::device_pointer_cast(d_stacks_rot);
    thrust::device_ptr<rotateRef> dt_stacks_compacted_rot = thrust::device_pointer_cast(d_stacks_compacted_rot);
    thrust::copy_if(dt_stacks_rot, dt_stacks_rot + params.maxN *tshape.x*tshape.y*tshape.z, dt_stacks_compacted_rot, is_not_empty_rot());
    d_stacks_compacted_rot = thrust::raw_pointer_cast(dt_stacks_compacted_rot);
    rotateRef* tmp = d_stacks_rot;
    d_stacks_rot = d_stacks_compacted_rot;
    checkCudaErrors(cudaFree(tmp));
    //k_debug_lookup_stacks <<< 1, 1 >>>(d_stacks, tshape.x*tshape.y*tshape.z);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Allocate memory for gathered stacks uchar
    checkCudaErrors(cudaMalloc((void**)&d_gathered4dstack_rot, sizeof(float)*(gather_stacks_sum_rot*params.patch_size*params.patch_size*params.patch_size)));
    //std::cout << "Allocated " << sizeof(float)*(gather_stacks_sum*params.patch_size*params.patch_size*params.patch_size) << " bytes for gathered4dstack" << std::endl;

    bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum_rot / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum_rot / threads);
    if (bs_x == 0) bs_x++;
    printf("gather_stacks_sum_rot:%d, %d, %d\n", gather_stacks_sum_rot, bs_x, threads);
    k_gather_cubes_rot <<<  bs_x, threads >>> (d_noisy_volume_3d, size, params, d_stacks_rot, gather_stacks_sum_rot, d_gathered4dstack_rot);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// ==========================================================================================
// ===================================== 4D Filtering =======================================
// ==========================================================================================

// support 4x4x4 patchsize
__global__ void dct3d(float* d_gathered4dstack, int patch_size, uint gather_stacks_sum){
    for(int cuIdx=blockIdx.x; cuIdx < gather_stacks_sum; cuIdx+=blockDim.x*gridDim.x){
        if( cuIdx >= gather_stacks_sum) return;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    //int cuIdx = blockIdx.x;
    int stride = patch_size*patch_size*patch_size;
    // DCT 4x4 matrix
    const float dct_coeff[4][4] =
    {
        { 0.500000000000000f,  0.500000000000000f,  0.500000000000000f,  0.500000000000000f },
        { 0.653281482438188f,  0.270598050073099f, -0.270598050073099f, -0.653281482438188f },
        { 0.500000000000000f, -0.500000000000000f, -0.500000000000000f,  0.500000000000000f },
        { 0.270598050073099f, -0.653281482438188f,  0.653281482438188f, -0.270598050073099f }
    };
    const float dct_coeff_T[4][4] =
    {
        { 0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f },
        { 0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f },
        { 0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f },
        { 0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f }
    };
    // Load corresponding cube to the shared memory
    __shared__ float cube[4][4][4];
    int idx = (cuIdx*stride)+(x + y*patch_size + z*patch_size*patch_size);
    cube[z][y][x] = d_gathered4dstack[idx];
    __syncthreads();
    // Do 2d dct for rows (by taking slices along z direction)
    float tmp = dct_coeff[y][0] * cube[z][0][x] + dct_coeff[y][1] * cube[z][1][x] + dct_coeff[y][2] * cube[z][2][x] + dct_coeff[y][3] * cube[z][3][x];
    __syncthreads();
    cube[z][y][x] = tmp;
    __syncthreads();
    tmp = dct_coeff_T[0][x] * cube[z][y][0] + dct_coeff_T[1][x] * cube[z][y][1] + dct_coeff_T[2][x] * cube[z][y][2] + dct_coeff_T[3][x] * cube[z][y][3];
    __syncthreads();
    cube[z][y][x] = tmp;
    __syncthreads();
    // Grab Z vector
    float z_vec[4];
    for (int i = 0; i < 4; ++i){
        z_vec[i] = cube[i][y][x];
    }
    __syncthreads();
    cube[z][y][x] = z_vec[0] * dct_coeff[z][0] + z_vec[1] * dct_coeff[z][1] + z_vec[2] * dct_coeff[z][2] + z_vec[3] * dct_coeff[z][3];
    __syncthreads();
    d_gathered4dstack[idx] = cube[z][y][x];
    }
}

void run_dct3d(float* d_gathered4dstack, uint gather_stacks_sum, int patch_size, const cudaDeviceProp &d_prop){
    int threads = patch_size*patch_size*patch_size;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum / threads);
    if (bs_x == 0) bs_x = 1;
    
    dct3d <<< bs_x, dim3(patch_size, patch_size, patch_size) >>> (d_gathered4dstack, patch_size, gather_stacks_sum);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

__global__ void idct3d(float* d_gathered4dstack, int patch_size, uint gather_stacks_sum){
    for(int cuIdx=blockIdx.x; cuIdx < gather_stacks_sum; cuIdx+=blockDim.x*gridDim.x){
        if( cuIdx >= gather_stacks_sum) return;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    //int cuIdx = blockIdx.x;
    int stride = patch_size*patch_size*patch_size;
    // DCT 4x4 matrix
    const float dct_coeff[4][4] =
    {
        { 0.500000000000000f, 0.500000000000000f, 0.500000000000000f, 0.500000000000000f },
        { 0.653281482438188f, 0.270598050073099f, -0.270598050073099f, -0.653281482438188f },
        { 0.500000000000000f, -0.500000000000000f, -0.500000000000000f, 0.500000000000000f },
        { 0.270598050073099f, -0.653281482438188f, 0.653281482438188f, -0.270598050073099f }
    };
    const float dct_coeff_T[4][4] =
    {
        { 0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f },
        { 0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f },
        { 0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f },
        { 0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f }
    };
    // Load corresponding cube to the shared memory
    __shared__ float cube[4][4][4];
    int idx = (cuIdx*stride) + (x + y*patch_size + z*patch_size*patch_size);
    cube[z][y][x] = d_gathered4dstack[idx];
    __syncthreads();
    float z_vec[4];
    for (int i = 0; i < 4; ++i){
        z_vec[i] = cube[i][y][x];
    }
    __syncthreads();
    cube[z][y][x] = z_vec[0] * dct_coeff_T[z][0] + z_vec[1] * dct_coeff_T[z][1] + z_vec[2] * dct_coeff_T[z][2] + z_vec[3] * dct_coeff_T[z][3];
    __syncthreads();
    float tmp = dct_coeff_T[y][0] * cube[z][0][x] + dct_coeff_T[y][1] * cube[z][1][x] + dct_coeff_T[y][2] * cube[z][2][x] + dct_coeff_T[y][3] * cube[z][3][x];
    __syncthreads();
    cube[z][y][x] = tmp;
    tmp = dct_coeff[0][x] * cube[z][y][0] + dct_coeff[1][x] * cube[z][y][1] + dct_coeff[2][x] * cube[z][y][2] + dct_coeff[3][x] * cube[z][y][3];
    __syncthreads();
    cube[z][y][x] = tmp;
    __syncthreads();
    d_gathered4dstack[idx] = cube[z][y][x];
    }
}

void run_idct3d(float* d_gathered4dstack, uint gather_stacks_sum, int patch_size, const cudaDeviceProp &d_prop){
    int threads = patch_size*patch_size*patch_size;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum / threads);
    if (bs_x == 0) bs_x = 1;
    idct3d <<< bs_x, dim3(patch_size, patch_size, patch_size) >>> (d_gathered4dstack, patch_size, gather_stacks_sum);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// (a,b) -> (a+b,a-b) without overflow
__device__ __host__ void whrotate(float& a, float& b)
{
    float t;
    t = a;
    a = a + b;
    b = t - b;
}

// Integer log2
__device__ __host__ long ilog2(long x)
{
    long l2 = 0;
    for (; x; x >>= 1) ++l2;
    return l2;
}

/**
* Fast Walsh-Hadamard transform
*/
__device__ __host__ void fwht(float* data, int size)
{
    const long l2 = ilog2(size) - 1;
    for (long i = 0; i < l2; ++i)
    {
        for (long j = 0; j < (1 << l2); j += 1 << (i + 1))
            for (long k = 0; k < (1 << i); ++k)
                whrotate(data[j + k], data[j + k + (1 << i)]);
    }
}

__global__ void k_run_wht_ht_iwht(float* d_gathered4dstack,
                                  uint groups,
                                  int patch_size,
                                  uint* d_nstacks,
                                  uint* accumulated_nstacks,
                                  float* d_group_weights,
                                  const float hard_th)
{

    for (uint cuIdx = blockIdx.x; cuIdx < groups; cuIdx += gridDim.x){
        if (cuIdx >= groups) return;

        int x = threadIdx.x;
        int y = threadIdx.y;
        int z = threadIdx.z;
        //int cuIdx = blockIdx.x;
        int stride = patch_size*patch_size*patch_size;
        float group_vector[16];
        int size = d_nstacks[cuIdx];
        int group_start = accumulated_nstacks[cuIdx];
        //printf("\nSize: %d Group start: %d \n", size, group_start);

        for (int i = 0; i < size; i++){
            long long int gl_idx = (group_start*stride) + (x + y*patch_size + z*patch_size*patch_size + i*stride);
            group_vector[i] = d_gathered4dstack[gl_idx];
        }

        fwht(group_vector, size);
        //// Threshold
        float threshold = hard_th * sqrtf((float)size);
        d_group_weights[cuIdx*stride + x + y*patch_size + z*patch_size*patch_size] = 0;
        for (int i = 0; i < size; i++){
            group_vector[i] /= size; // normalize
            if (fabs(group_vector[i]) > threshold)
            {
                d_group_weights[cuIdx*stride + x + y*patch_size + z*patch_size*patch_size] += 1;
            }
            else {
                group_vector[i] = 0;
            }
        }
        //// Inverse fwht
        fwht(group_vector, size);
        for (int i = 0; i < size; i++){
            long long int gl_idx = (group_start*stride) + (x + y*patch_size + z*patch_size*patch_size + i*stride);
            d_gathered4dstack[gl_idx] = group_vector[i];
        }
    }
}
__global__ void k_sum_group_weights(float* d_group_weights, uint* d_accumulated_nstacks, uint* d_nstacks, uint groups, int patch_size){
    for (int cuIdx = blockIdx.x; cuIdx < groups; cuIdx += gridDim.x){
        if (cuIdx >= groups) return;
        int stride = patch_size*patch_size*patch_size;
        float counter = 0;
        for (int i=0;i<stride; ++i)
        {
            int idx = cuIdx*stride + i;
            counter+=d_group_weights[idx];
        }
        __syncthreads();
        d_group_weights[cuIdx*stride] = 1.0/(float)counter;
    }
}

struct is_computed_weight
{
    __host__ __device__
        bool operator()(const float x)
    {
        return (x < 1.0);
    }
};

void run_wht_ht_iwht(float* d_gathered4dstack,
                     uint gather_stacks_sum,
                     int patch_size,
                     uint* d_nstacks,
                     const uint3 tshape,
                     float* &d_group_weights,
                     const bm4d_gpu::Parameters params,
                     const cudaDeviceProp &d_prop)
{
    int groups = tshape.x*tshape.y*tshape.z;
    // Accumulate nstacks through sum
    uint* d_accumulated_nstacks;
    checkCudaErrors(cudaMalloc((void **)&d_accumulated_nstacks, sizeof(uint)*groups));
    thrust::device_ptr<uint> dt_accumulated_nstacks = thrust::device_pointer_cast(d_accumulated_nstacks);
    thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
    thrust::exclusive_scan(dt_nstacks, dt_nstacks + groups, dt_accumulated_nstacks);
    d_accumulated_nstacks = thrust::raw_pointer_cast(dt_accumulated_nstacks);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMalloc((void **)&d_group_weights, sizeof(float)*groups*patch_size*patch_size*patch_size)); // Cubes with weights for each group
    checkCudaErrors(cudaMemset(d_group_weights, 0.0, sizeof(float)*groups*patch_size*patch_size*patch_size));
    int threads = params.patch_size*params.patch_size*params.patch_size;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
    if (bs_x == 0) bs_x = 1;
    k_run_wht_ht_iwht <<< bs_x, dim3(params.patch_size, params.patch_size, params.patch_size) >>> (d_gathered4dstack, groups, patch_size, d_nstacks, d_accumulated_nstacks, d_group_weights, params.hard_th);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    threads = 1;
    bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
    k_sum_group_weights <<< bs_x, threads >>>(d_group_weights, d_accumulated_nstacks, d_nstacks, groups, patch_size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_accumulated_nstacks));
}

// ==========================================================================================
// ==================================== Rot Aggregation =====================================
// ==========================================================================================

__device__ float trilinear_interp(const float* data, float x, float y, float z, int N) {
    float cap = (float) N - 1.0;
    x = max(0.0, min(x, cap));
    y = max(0.0, min(y, cap));
    z = max(0.0, min(z, cap));
    
    int x0 = int(x);
    int y0 = int(y);
    int z0 = int(z);
 
    float xd = x - floor(x);
    float yd = y - floor(y);
    float zd = z - floor(z);
 
    int x1 = int(ceil(x));
    int y1 = int(ceil(y));
    int z1 = int(ceil(z));
 
    // c000, c001
    float c00 = data[x0 * N * N + y0 * N + z0] * (1 - xd) +
                data[x1 * N * N + y0 * N + z0] * xd;
 
    // c001, c101
    float c01 = data[x0 * N * N + y0 * N + z1] * (1 - xd) +
                data[x1 * N * N + y0 * N + z1] * xd;
 
    // c010, c110
    float c10 = data[x0 * N * N + y1 * N + z0] * (1 - xd) +
                data[x1 * N * N + y1 * N + z0] * xd;
 
    // c011, c111
    float c11 = data[x0 * N * N + y1 * N + z1] * (1 - xd) +
                data[x1 * N * N + y1 * N + z1] * xd;
 
    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;
 
    // result of trilinear interpolation
    float c = c0 * (1 - zd) + c1 * zd;
 
    return c;
}

__device__ float rot_inference_inverse(const float* data, rotateRef rrf_inv, int x, int y, int z, int N, float pshift)
{
    float px, py, pz;
    float u = (float)N - 0.75f;
    float l = -0.25f;
    px = (float)x - pshift;
    py = (float)y - pshift;
    pz = (float)z - pshift;

    float rpx = px * rrf_inv.v1x + py * rrf_inv.v2x + pz * rrf_inv.v3x + pshift;
    float rpy = px * rrf_inv.v1y + py * rrf_inv.v2y + pz * rrf_inv.v3y + pshift;
    float rpz = px * rrf_inv.v1z + py * rrf_inv.v2z + pz * rrf_inv.v3z + pshift;

    if ((rpx > u || rpx < l) || (rpy > u || rpy < l) || (rpz > u || rpz < l)) {
        return -1;
    }
    else {
        return trilinear_interp(data, rpx, rpy, rpz, N);
    }
}

__global__ void k_aggregation_rot(float* d_denoised_volume,
                             float* d_weights_volume,
                             const uint3 size,
                             const uint3 tshape,
                             const float* d_gathered4dstack,
                             rotateRef* d_stacks,
                             uint* d_nstacks,
                             float* group_weights,
                             const bm4d_gpu::Parameters params,
                             const uint* d_accumulated_nstacks)
{
    uint groups = (tshape.x*tshape.y*tshape.z);
    int stride = params.patch_size*params.patch_size*params.patch_size;
    float pshift = (float)params.patch_size / 2.0 - 0.5;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < groups; i += blockDim.x*gridDim.x){

        if (i >= groups) return;

        float weight = group_weights[i*stride];
        int patches = d_nstacks[i];
        int group_beginning = d_accumulated_nstacks[i];
        // printf("group %d, weight %f\n", i, weight);
        //if (i > 15) return;
        for (int p = 0; p < patches; ++p){
            rotateRef ref = d_stacks[group_beginning + p];
            rotateRef ref_inv = invert_rotateRef_sim(ref);

            int edge_correction = floor((((float)params.patch_size - 1.0) / 2) * 1.5);

            for (int z = -edge_correction; z < params.patch_size + edge_correction; ++z)
                for (int y = -edge_correction; y < params.patch_size + edge_correction; ++y)
                    for (int x = -edge_correction; x < params.patch_size + edge_correction; ++x){
                        int rx = x + ref.x;
                        int ry = y + ref.y;
                        int rz = z + ref.z;

                        if (rx < 0 || rx >= size.x) continue;
                        if (ry < 0 || ry >= size.y) continue;
                        if (rz < 0 || rz >= size.z) continue;

                        int img_idx = (rx)+(ry)*size.x + (rz)*size.x*size.y;
                        long long int stack_start_idx = group_beginning*stride + p*stride;
                        const float* denoised_patch = &d_gathered4dstack[stack_start_idx];

                        float tmp = rot_inference_inverse(denoised_patch, ref_inv, x, y, z, params.patch_size, pshift);
                        
                        if (tmp != -1.0) {
                            // printf("%f\n", tmp);
                            __syncthreads();
                            atomicAdd(&d_denoised_volume[img_idx], tmp*weight);
                            atomicAdd(&d_weights_volume[img_idx], weight);
                        }
                    }
        }
    }
}

__global__ void k_normalizer_rot(float* d_denoised_volume,
                                const float* __restrict d_weights_volume,
                                const uint3 size)
{
    int im_size = size.x*size.y*size.z;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < im_size; i += blockDim.x*gridDim.x)
            {
                if (i >= im_size) return;
                float tmp = d_denoised_volume[i];
                __syncthreads();
                d_denoised_volume[i] = tmp / d_weights_volume[i];
            }
}

void run_aggregation_rot(float* final_image,
                     const uint3 size,
                     const uint3 tshape,
                     const float* d_gathered4dstack,
                     rotateRef* d_stacks,
                     uint* d_nstacks,
                     float* d_group_weights,
                     const bm4d_gpu::Parameters params,
                     int gather_stacks_sum,
                     const cudaDeviceProp &d_prop)
{
    int im_size = size.x*size.y*size.z;
    int groups = tshape.x*tshape.y*tshape.z;


    // Accumulate nstacks through sum
    uint* d_accumulated_nstacks;
    checkCudaErrors(cudaMalloc((void **)&d_accumulated_nstacks, sizeof(uint)*groups));
    thrust::device_ptr<uint> dt_accumulated_nstacks = thrust::device_pointer_cast(d_accumulated_nstacks);
    thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
    thrust::exclusive_scan(dt_nstacks, dt_nstacks + groups, dt_accumulated_nstacks);
    d_accumulated_nstacks = thrust::raw_pointer_cast(dt_accumulated_nstacks);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    float* d_denoised_volume, *d_weights_volume;
    checkCudaErrors(cudaMalloc((void **)&d_denoised_volume, sizeof(float)*size.x*size.y*size.z));
    checkCudaErrors(cudaMalloc((void **)&d_weights_volume, sizeof(float)*size.x*size.y*size.z));
    checkCudaErrors(cudaMemset(d_denoised_volume, 0.0, sizeof(float)*size.x*size.y*size.z));
    checkCudaErrors(cudaMemset(d_weights_volume, 0.0, sizeof(float)*size.x*size.y*size.z));
    int threads = 128;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
    if (bs_x == 0) bs_x++;
    
    printf("Begin Aggregation Kernel Call\n");
    k_aggregation_rot <<< bs_x, threads >>>(d_denoised_volume, d_weights_volume, size, tshape, d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params, d_accumulated_nstacks);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    threads = 128;
    bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(im_size / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(im_size / threads);
    if (bs_x == 0) bs_x++;
    printf("Begin Normalization\n");
    k_normalizer_rot <<< bs_x, threads >>>(d_denoised_volume, d_weights_volume, size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(final_image, d_denoised_volume, sizeof(float)*im_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_denoised_volume));
    checkCudaErrors(cudaFree(d_weights_volume));
    checkCudaErrors(cudaFree(d_accumulated_nstacks));
}
