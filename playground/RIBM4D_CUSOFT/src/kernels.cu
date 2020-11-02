/*
* 2016, Vladislav Tananaev
* v.d.tananaev [at] gmail [dot] com
*/
#include <bm4d-gpu/kernels.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <math.h>
// #include "cublas_v2.h"

texture<uchar, 3, cudaReadModeNormalizedFloat> noisy_volume_3d_tex;

float normal_pdf_sqr(float std, float x) {
    // PDF of zero-mean gaussian (normalized)
    // input x: square of distance
    float xh = sqrt(x) / std;
    return exp(- (xh * xh) / 2.0);
}

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

__device__ rotateRef make_rotateRef_sim(uint x, uint y, uint z, float4 sim) {
    rotateRef rrf;
    rrf.x = x;
    rrf.y = y;
    rrf.z = z;
    rrf.val = sim.w;

    // Compute rotation matrix
    float sa = sin(sim.x), ca = cos(sim.x);
    float sb = sin(sim.y), cb = cos(sim.y);
    float sg = sin(sim.z), cg = cos(sim.z);
    
    rrf.v1x = cb * cg;
    rrf.v1y = cb * sg;
    rrf.v1z = -sb;

    rrf.v2x = sa * sb * cg - ca * sb;
    rrf.v2y = sa * sb * sg + ca * cb;
    rrf.v2z = sa * cb;

    rrf.v3x = ca * sb * cg + sa * sg;
    rrf.v3y = ca * sb * sg - sa * cg;
    rrf.v3z = ca * cb;

    return rrf;
}

__device__ float rotTex3D(rotateRef rrf, int px, int py, int pz, float pshift) {
    // must be called after the texture memory is binded with the noisy array.
    float fpx = (float)px - pshift;
    float fpy = (float)py - pshift;
    float fpz = (float)pz - pshift;
    
    float rpx = fpx * rrf.v1x + fpx * rrf.v2x + fpx * rrf.v3x + pshift + rrf.x;
    float rpy = fpy * rrf.v1y + fpy * rrf.v2y + fpy * rrf.v3y + pshift + rrf.y;
    float rpz = fpz * rrf.v1z + fpz * rrf.v2z + fpz * rrf.v3z + pshift + rrf.z;

    return tex3D(noisy_volume_3d_tex, rpx + 0.5f, rpy + 0.5f, rpz + 0.5f) * 255;
}

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
}


__global__ void k_run_fft_precomp(const uchar* __restrict img,
                                  const uint3 size,
                                  const uint3 tshape,
                                  const bm4d_gpu::Parameters params,
                                  double *d_fftCoefR,
                                  double *d_fftCoefI)
{

    for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tshape.z; Idz += blockDim.z*gridDim.z)
        for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tshape.y; Idy += blockDim.y*gridDim.y)
            for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tshape.x; Idx += blockDim.x*gridDim.x)
            {
                int x = Idx * params.step_size;
                int y = Idy * params.step_size;
                int z = Idz * params.step_size;
                if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
                    return;
                uint3 ref = make_uint3(x, y, z);
                // TODO: compute the fft and update d_shfft_res
            }
}

void run_fft_precomp(const uchar* __restrict d_noisy_volume,
                     const uint3 size,
                     const uint3 tshape,
                     const bm4d_gpu::Parameters params,
                     double *d_fftCoefR,
                     double *d_fftCoefI,
                     const cudaDeviceProp &d_prop)
{
    int threads = std::floor(sqrt(d_prop.maxThreadsPerBlock));
    dim3 block(threads, threads, 1);
    int bs_x = d_prop.maxGridSize[1] < tshape.x ? d_prop.maxGridSize[1] : tshape.x;
    int bs_y = d_prop.maxGridSize[1] < tshape.y ? d_prop.maxGridSize[1] : tshape.y;
    dim3 grid(bs_x, bs_y, 1);

    // Debug verification
    std::cout << "Pre compute fft on patches " << (tshape.x*tshape.y*tshape.z) << std::endl;

    k_run_fft_precomp <<< grid, block >>>(d_noisy_volume,
                                         size,
                                         tshape,
                                         params,
                                         d_fftCoefR,
                                         d_fftCoefI);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

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

// ==========================================================================================
// ======================================= Standard Matching ================================
// ==========================================================================================

__device__ void add_stack(uint3float1* d_stacks,
                          uint* d_nstacks,
                          const uint3float1 val,
                          const int maxN)
{
    int k;
    uint num = (*d_nstacks);
    if (num < maxN) //add new value
    {
        k = num++;
        while (k > 0 && val.val > d_stacks[k-1].val)
        {
            d_stacks[k] = d_stacks[k - 1];
            --k;
        }

        d_stacks[k] = val;
        *d_nstacks = num;
    }
    else if (val.val >= d_stacks[0].val) return;
    else //delete highest value and add new
    {
        k = 1;
        while (k < maxN && val.val < d_stacks[k].val)
        {
            d_stacks[k - 1] = d_stacks[k];
            k++;
        }
        d_stacks[k - 1] = val;
    }
}
// Distance between patches
__device__ float dist(const uchar* __restrict img, const uint3 size, const uint3 ref, const uint3 cmp, const int k){
    float diff(0);
    for (int z = 0; z < k; ++z)
        for (int y = 0; y < k; ++y)
            for (int x = 0; x < k; ++x){
                int rx = max(0, min(x + ref.x, size.x - 1));
                int ry = max(0, min(y + ref.y, size.y - 1));
                int rz = max(0, min(z + ref.z, size.z - 1));
                int cx = max(0, min(x + cmp.x, size.x - 1));
                int cy = max(0, min(y + cmp.y, size.y - 1));
                int cz = max(0, min(z + cmp.z, size.z - 1));
                float tmp = (img[(rx) + (ry)*size.x + (rz)*size.x*size.y] - img[(cx) + (cy)*size.x + (cz)*size.x*size.y]);
                diff += tmp*tmp;
         }
    return diff/(k*k*k);
}

__global__ void k_block_matching(const uchar* __restrict img,
                                 const uint3 size,
                                 const uint3 tshape,
                                 const bm4d_gpu::Parameters params,
                                 uint3float1* d_stacks,
                                 uint* d_nstacks)
{

    for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tshape.z; Idz += blockDim.z*gridDim.z)
        for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tshape.y; Idy += blockDim.y*gridDim.y)
            for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tshape.x; Idx += blockDim.x*gridDim.x)
            {

            int x = Idx * params.step_size;
            int y = Idy * params.step_size;
            int z = Idz * params.step_size;
            if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
                return;

            int wxb = fmaxf(0, x - params.window_size); // window x begin
            int wyb = fmaxf(0, y - params.window_size); // window y begin
            int wzb = fmaxf(0, z - params.window_size); // window z begin
            int wxe = fminf(size.x - 1, x + params.window_size); // window x end
            int wye = fminf(size.y - 1, y + params.window_size); // window y end
            int wze = fminf(size.z - 1, z + params.window_size); // window z end

            uint3 ref = make_uint3(x, y, z);

            for (int wz = wzb; wz <= wze; wz++)
                for (int wy = wyb; wy <= wye; wy++)
                    for (int wx = wxb; wx <= wxe; wx++){
                        float w = dist(img, size, ref, make_uint3(wx, wy, wz), params.patch_size);
                        // printf("Dist %f\n", w);

                        if (w < params.sim_th){
                            add_stack(&d_stacks[(Idx + (Idy + Idz* tshape.y)*tshape.x)*params.maxN],
                                      &d_nstacks[Idx + (Idy + Idz* tshape.y)*tshape.x],
                                      uint3float1(wx, wy, wz, w),
                                      params.maxN);
                        }
                    }
        }
}

// linear memory implementation
void run_block_matching(const uchar* __restrict d_noisy_volume,
                        const uint3 size,
                        const uint3 tshape,
                        const bm4d_gpu::Parameters params,
                        uint3float1 *d_stacks,
                        uint *d_nstacks,
                        const cudaDeviceProp &d_prop)
{
    int threads = std::floor(sqrt(d_prop.maxThreadsPerBlock));
    dim3 block(threads, threads, 1);
    int bs_x = d_prop.maxGridSize[1] < tshape.x ? d_prop.maxGridSize[1] : tshape.x;
    int bs_y = d_prop.maxGridSize[1] < tshape.y ? d_prop.maxGridSize[1] : tshape.y;
    dim3 grid(bs_x, bs_y, 1);

    // Debug verification
    std::cout << "Total number of reference patches " << (tshape.x*tshape.y*tshape.z) << std::endl;

    k_block_matching <<< grid, block >>>(d_noisy_volume,
                                         size,
                                         tshape,
                                         params,
                                         d_stacks,
                                         d_nstacks);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// ==========================================================================================
// ======================================= Rotation Matching ================================
// ==========================================================================================

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

// TODO: Ziyi: dock with API for Euler Angle computation, return a float4 vector (similarity, alpha, beta, gamma)
__device__ float4 dist_rot(const float* __restrict d_shfft_res, const uint3 ref, const uint3 cmp, const int k, const int fft_patch_size){
    // d_shfft_res is the array of precomputed fft result
    // img_surf is the cuda array surface data (a byproduct of texture)
    float diff(0);
    float4 info;
    info.x = 0;
    info.y = 0;
    info.z = 0;
    info.w = diff;
    // TODO: Complete this thing
    return info;
}

__global__ void k_block_matching_rot(const float* __restrict d_shfft_res,
                                     const uint3 size,
                                     const uint3 tshape,
                                     const bm4d_gpu::Parameters params,
                                     rotateRef* d_stacks_rot,
                                     uint* d_nstacks_rot,
                                     int fft_patch_size)
{
    for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tshape.z; Idz += blockDim.z*gridDim.z)
        for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tshape.y; Idy += blockDim.y*gridDim.y)
            for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tshape.x; Idx += blockDim.x*gridDim.x)
            {

            int x = Idx * params.step_size;
            int y = Idy * params.step_size;
            int z = Idz * params.step_size;
            if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
                return;

            int wxb = fmaxf(0, x - params.window_size); // window x begin
            int wyb = fmaxf(0, y - params.window_size); // window y begin
            int wzb = fmaxf(0, z - params.window_size); // window z begin
            int wxe = fminf(size.x - 1, x + params.window_size); // window x end
            int wye = fminf(size.y - 1, y + params.window_size); // window y end
            int wze = fminf(size.z - 1, z + params.window_size); // window z end

            uint3 ref = make_uint3(x, y, z);

            for (int wz = wzb; wz <= wze; wz++)
                for (int wy = wyb; wy <= wye; wy++)
                    for (int wx = wxb; wx <= wxe; wx++){

                        uint3 cmp = make_uint3(wx, wy, wz);
                        float4 sim = dist_rot(d_shfft_res, ref, cmp, params.patch_size, fft_patch_size);
                        
                        if (sim.w < params.sim_th){
                            // printf("Dist %f\n", sim.w);
                            add_stack_rot(&d_stacks_rot[(Idx + (Idy + Idz* tshape.y)*tshape.x)*params.maxN],
                                          &d_nstacks_rot[Idx + (Idy + Idz* tshape.y)*tshape.x],
                                          make_rotateRef_sim(wx, wy, wz, sim),
                                          params.maxN);
                        }
                    }
        }
}

void run_block_matching_rot(const uchar* __restrict d_noisy_volume,
                            const float* __restrict d_shfft_res,
                            const uint3 size,
                            const uint3 tshape,
                            const bm4d_gpu::Parameters params,
                            rotateRef *d_stacks_rot,
                            uint *d_nstacks_rot,
                            int fft_patch_size,
                            const cudaDeviceProp &d_prop)
{
    int threads = std::floor(sqrt(d_prop.maxThreadsPerBlock));
    dim3 block(threads, threads, 1);
    int bs_x = d_prop.maxGridSize[1] < tshape.x ? d_prop.maxGridSize[1] : tshape.x;
    int bs_y = d_prop.maxGridSize[1] < tshape.y ? d_prop.maxGridSize[1] : tshape.y;
    dim3 grid(bs_x, bs_y, 1);

    // Debug verification
    std::cout << "Total number of reference patches " << (tshape.x*tshape.y*tshape.z) << std::endl;

    k_block_matching_rot <<< grid, block >>>(d_shfft_res,
                                             size,
                                             tshape,
                                             params,
                                             d_stacks_rot,
                                             d_nstacks_rot,
                                             fft_patch_size);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

// ==========================================================================================
// ========================================== Filtering =====================================
// ==========================================================================================

__global__ void k_nstack_to_pow(uint3float1* d_stacks, uint* d_nstacks, const int elements, const uint maxN){
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < elements; i += blockDim.x*gridDim.x){
        if (i >= elements) return;

        uint inGroupId = i % maxN;
        uint groupId = i/maxN;

        uint n = d_nstacks[groupId];
        uint tmp = flp2(n);
        uint diff = d_nstacks[groupId] - tmp;

        __syncthreads();
        d_nstacks[groupId] = tmp;

        if (inGroupId < diff || inGroupId >= n)
            d_stacks[i].val = -1;

    }
}

__global__ void k_gather_cubes(const uchar* __restrict img,
                               const uint3 size,
                               const bm4d_gpu::Parameters params,
                               const uint3float1* __restrict d_stacks,
                               const uint array_size,
                               float* d_gathered4dstack)
{
    int cube_size = params.patch_size*params.patch_size*params.patch_size;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < array_size; i += blockDim.x*gridDim.x){
        if (i >= array_size) return;
        uint3float1 ref = d_stacks[i];

        for (int z = 0; z < params.patch_size; ++z)
            for (int y = 0; y < params.patch_size; ++y)
                for (int x = 0; x < params.patch_size; ++x){

                    int rx = max(0, min(x + ref.x, size.x - 1));
                    int ry = max(0, min(y + ref.y, size.y - 1));
                    int rz = max(0, min(z + ref.z, size.z - 1));

                    int img_idx = (rx) + (ry)*size.x + (rz)*size.x*size.y;
                    int stack_idx = i*cube_size + (x)+(y + z*params.patch_size)*params.patch_size;

                    d_gathered4dstack[stack_idx] = img[img_idx];
                }

    }
}

struct is_not_empty
{
    __host__ __device__
    bool operator()(const uint3float1 x)
    {
        return (x.val != -1);
    }
};

void gather_cubes(const uchar* __restrict img,
                  const uint3 size,
                  const uint3 tshape,
                  const bm4d_gpu::Parameters params,
                  uint3float1* &d_stacks,
                  uint* d_nstacks,
                  float* &d_gathered4dstack,
                  uint &gather_stacks_sum,
                  const cudaDeviceProp &d_prop)
{
    // Convert all the numbers in d_nstacks to the lowest power of two
    uint array_size = (tshape.x*tshape.y*tshape.z);
    int threads = d_prop.maxThreadsPerBlock;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(params.maxN*array_size / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(params.maxN*array_size / threads);
    k_nstack_to_pow <<< bs_x, threads >>>(d_stacks, d_nstacks, params.maxN*array_size, params.maxN);
    checkCudaErrors(cudaGetLastError());
    thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
    gather_stacks_sum = thrust::reduce(dt_nstacks, dt_nstacks + array_size);
    //std::cout << "Sum of pathces: "<< gather_stacks_sum << std::endl;
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    // Make a compaction
    uint3float1 * d_stacks_compacted;
    checkCudaErrors(cudaMalloc((void**)&d_stacks_compacted, sizeof(uint3float1)*gather_stacks_sum));
    thrust::device_ptr<uint3float1> dt_stacks = thrust::device_pointer_cast(d_stacks);
    thrust::device_ptr<uint3float1> dt_stacks_compacted = thrust::device_pointer_cast(d_stacks_compacted);
    thrust::copy_if(dt_stacks, dt_stacks + params.maxN *tshape.x*tshape.y*tshape.z, dt_stacks_compacted, is_not_empty());
    d_stacks_compacted = thrust::raw_pointer_cast(dt_stacks_compacted);
    uint3float1* tmp = d_stacks;
    d_stacks = d_stacks_compacted;
    checkCudaErrors(cudaFree(tmp));
    //k_debug_lookup_stacks <<< 1, 1 >>>(d_stacks, tshape.x*tshape.y*tshape.z);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Allocate memory for gathered stacks uchar
    checkCudaErrors(cudaMalloc((void**)&d_gathered4dstack, sizeof(float)*(gather_stacks_sum*params.patch_size*params.patch_size*params.patch_size)));
    //std::cout << "Allocated " << sizeof(float)*(gather_stacks_sum*params.patch_size*params.patch_size*params.patch_size) << " bytes for gathered4dstack" << std::endl;

    bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum / threads);
    k_gather_cubes <<<  bs_x, threads >>> (img, size, params, d_stacks, gather_stacks_sum, d_gathered4dstack);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

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

void aggregation_cpu(float* image_vol,
                     float* weights_vol,
                     float* group_weights,
                     uint3 size,
                     uint3 tshape,
                     int gather_stacks_sum,
                     uint3float1* stacks,
                     uint* nstacks,
                     float* gathered_stacks,
                     int patch_size)
{
    int all_cubes = gather_stacks_sum;
    int stride = patch_size*patch_size*patch_size;

    int cubes_so_far = 0;
    int groupId = 0; // Iterator over group numbers

    for (int i = 0; i < all_cubes; ++i){
        uint3float1 ref = stacks[i];

        uint cubes_in_group = nstacks[groupId];
        if ((i - cubes_so_far)==cubes_in_group) {
            cubes_so_far += cubes_in_group;
            //std::cout << "cubes in a grouo " << cubes_in_group << std::endl;
            groupId++;
        }

        float weight = group_weights[groupId*stride];
        //std::cout << "Weight: " << weight << std::endl;

        for (int z = 0; z < patch_size; ++z)
            for (int y = 0; y < patch_size; ++y)
                for (int x = 0; x < patch_size; ++x){
                    int rx = x + ref.x;
                    int ry = y + ref.y;
                    int rz = z + ref.z;
                    if (rx < 0 || rx >= size.x) continue;
                    if (ry < 0 || ry >= size.y) continue;
                    if (rz < 0 || rz >= size.z) continue;
                    //std::cout << image_vol[rx + ry*size.x + rz*size.x*size.y] << std::endl;
                    float tmp = gathered_stacks[i*stride + x + y*patch_size + z*patch_size*patch_size];
                    image_vol[rx + ry*size.x + rz*size.x*size.y] += tmp*weight;
                    weights_vol[rx + ry*size.x + rz*size.x*size.y] += weight;
                }
    }

    for (int i = 0; i < size.x*size.y*size.z; ++i){
        image_vol[i] /= weights_vol[i];
    }

}

__global__ void k_aggregation(float* d_denoised_volume,
                             float* d_weights_volume,
                             const uint3 size,
                             const uint3 tshape,
                             const float* d_gathered4dstack,
                             uint3float1* d_stacks,
                             uint* d_nstacks,
                             float* group_weights,
                             const bm4d_gpu::Parameters params,
                             const uint* d_accumulated_nstacks)
{
    uint groups = (tshape.x*tshape.y*tshape.z);
    int stride = params.patch_size*params.patch_size*params.patch_size;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < groups; i += blockDim.x*gridDim.x){

        if (i >= groups) return;

        float weight = group_weights[i*stride];
        int patches = d_nstacks[i];
        int group_beginning = d_accumulated_nstacks[i];
        //printf("Weight for the group %d is %f\n", i, weight);
        //printf("Num of patches %d\n", patches);
        //printf("Group beginning %d\n", group_beginning);
        //if (i > 15) return;
        for (int p = 0; p < patches; ++p){
            uint3float1 ref = d_stacks[group_beginning + p];

            for (int z = 0; z < params.patch_size; ++z)
                for (int y = 0; y < params.patch_size; ++y)
                    for (int x = 0; x < params.patch_size; ++x){
                        int rx = x + ref.x;
                        int ry = y + ref.y;
                        int rz = z + ref.z;

                        if (rx < 0 || rx >= size.x) continue;
                        if (ry < 0 || ry >= size.y) continue;
                        if (rz < 0 || rz >= size.z) continue;

                        int img_idx = (rx)+(ry)*size.x + (rz)*size.x*size.y;
                        long long int stack_idx = group_beginning*stride + (x)+(y + z*params.patch_size)*params.patch_size + p*stride;
                        float tmp = d_gathered4dstack[stack_idx];
                        __syncthreads();
                        atomicAdd(&d_denoised_volume[img_idx], tmp*weight);
                        atomicAdd(&d_weights_volume[img_idx], weight);
                    }
        }
    }
}

__global__ void k_normalizer(float* d_denoised_volume,
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

void run_aggregation(float* final_image,
                     const uint3 size,
                     const uint3 tshape,
                     const float* d_gathered4dstack,
                     uint3float1* d_stacks,
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
    int threads = d_prop.maxThreadsPerBlock;
    int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
    k_aggregation <<< bs_x, threads >>>(d_denoised_volume, d_weights_volume, size, tshape, d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params, d_accumulated_nstacks);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    threads = d_prop.maxThreadsPerBlock;
    bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(im_size / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(im_size / threads);
    k_normalizer <<< bs_x, threads >>>(d_denoised_volume, d_weights_volume, size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(final_image, d_denoised_volume, sizeof(float)*im_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_denoised_volume));
    checkCudaErrors(cudaFree(d_weights_volume));
    checkCudaErrors(cudaFree(d_accumulated_nstacks));

}
