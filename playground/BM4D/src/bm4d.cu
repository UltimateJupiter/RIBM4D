/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 */

#include <bm4d-gpu/bm4d.h>
texture<uchar, 3, cudaReadModeNormalizedFloat> noisy_volume_3d_tex;

void BM4D::load_3d_array() {
    Stopwatch copyingtodevice(true);
    const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    checkCudaErrors(cudaMalloc3DArray(&d_noisy_volume_3d, &channelDesc, volumeSize));

    // Copy data to 3D array (host to device)
    uchar *volume_tmp = &noisy_volume[0];
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)volume_tmp, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_noisy_volume_3d;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
    copyingtodevice.stop(); std::cout << "Copying to device (3d tex) took:" << copyingtodevice.getSeconds() << std::endl;
    
    // Create the surface object
    struct cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(surfRes));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = d_noisy_volume_3d;
    checkCudaErrors(cudaCreateSurfaceObject(&noisy_volume_3d_surf, &surfRes));
    std::cout << "Binded with surface reference" << std::endl;

    // Create the texture object
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = d_noisy_volume_3d;
    cudaTextureDesc  texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&noisy_volume_3d_tex, &texRes, &texDesc, NULL));
}

std::vector<uchar> BM4D::run_first_step() {
    // Stopwatch copyingtodevice(true);
    uchar* d_noisy_volume;
    Stopwatch copyingtodevice(true);
    assert(size == noisy_volume.size());

    load_3d_array();

    // Old Memory Copy
    checkCudaErrors(cudaMalloc((void**)&d_noisy_volume, sizeof(uchar) * size));
    checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)noisy_volume.data(), sizeof(uchar) * size, cudaMemcpyHostToDevice));
    copyingtodevice.stop(); std::cout<<"Copying to device took:" << copyingtodevice.getSeconds() <<std::endl;

    uint3 im_size = make_uint3(width, height, depth);
    uint3 tr_size = make_uint3(twidth, theight, tdepth);    // Truncated size, with some step for ref patches
    
    // Do block matching
    Stopwatch blockmatching(true);
    run_block_matching(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_prop);
    run_block_matching_3d(d_noisy_volume_3d, im_size, tr_size, params, d_stacks, d_nstacks, d_prop);
    blockmatching.stop();
    std::cout << "Blockmatching took: " << blockmatching.getSeconds() << std::endl;

    // Gather cubes together
    uint gather_stacks_sum;
    Stopwatch gatheringcubes(true);
    gather_cubes(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_gathered4dstack, gather_stacks_sum, d_prop);
    // std::cout << "Acquied size " << gather_stacks_sum << std::endl;
    gatheringcubes.stop();
    std::cout << "Gathering cubes took: " << gatheringcubes.getSeconds() << std::endl;
    // debug_kernel(d_gathered4dstack);
    checkCudaErrors(cudaFree(d_noisy_volume));

    // Perform 3D DCT
    Stopwatch dct_forward(true);
    run_dct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);
    dct_forward.stop();
    std::cout << "3D DCT forwards took: " << dct_forward.getSeconds() << std::endl;
    // debug_kernel(d_gathered4dstack);

    // Do WHT in 4th dim + Hard Thresholding + IWHT
    float* d_group_weights;
    Stopwatch wht_t(true);
    run_wht_ht_iwht(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_nstacks, tr_size, d_group_weights, params, d_prop);
    wht_t.stop();
    std::cout << "WHT took: " << wht_t.getSeconds() << std::endl;

    // Perform inverse 3D DCT
    Stopwatch dct_backward(true);
    run_idct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);
    dct_backward.stop();
    std::cout << "3D DCT backwards took: " << dct_backward.getSeconds() << std::endl;
    // debug_kernel(d_gathered4dstack);

    // Aggregate
    float* final_image = new float[width * height * depth];
    memset(final_image, 0.0, sizeof(float) * width * height * depth);
    Stopwatch aggregation_t(true);
    run_aggregation(final_image, im_size, tr_size, d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params, gather_stacks_sum, d_prop);
    aggregation_t.stop();
    std::cout << "Aggregation took: " << aggregation_t.getSeconds() << std::endl;
    for (int i = 0; i < size; i++) {
        noisy_volume[i] = static_cast<uchar>(final_image[i]);
    }
    delete[] final_image;
    return noisy_volume;
}
