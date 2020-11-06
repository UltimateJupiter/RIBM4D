/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 * 2020, Xingyu Zhu
 * jupiter.zhuxingyu [at] gmail [dot] com
 */

#include <bm4d-gpu/bm4d.h>
// texture<uchar, 3, cudaReadModeNormalizedFloat> noisy_volume_3d_tex;

void BM4D::load_3d_array() {
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
    std::cout << "Copying to device (3d tex)" << std::endl;
}

void BM4D::init_masks() {
    float std = pshift * 0.75; // std of gaussian 
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
};

std::vector<uchar> BM4D::run_first_step() {
    assert(size == noisy_volume.size());
    uint3 imshape = make_uint3(width, height, depth);
    uint3 tshape = make_uint3(twidth, theight, tdepth);    // Truncated size, with some step for ref patches

    checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)noisy_volume.data(), sizeof(uchar) * size, cudaMemcpyHostToDevice));
    load_3d_array();
    check_texture_sync(d_noisy_volume, d_noisy_volume_3d, imshape, params.patch_size);
    
    // Do block matching
    // Stopwatch blockmatching(true);
    // std::cout << "\nStart blockmatching" << std::endl;
    // run_block_matching(d_noisy_volume, imshape, tshape, params, d_stacks, d_nstacks, d_prop);
    // blockmatching.stop();
    // std::cout << "Blockmatching took: " << blockmatching.getSeconds() << std::endl;
    bind_texture(d_noisy_volume_3d);
    Stopwatch blockmatching_rot(true);
    std::cout << "\nStart blockmatching (rot)" << std::endl;
    run_block_matching_rot(d_noisy_volume,
                            imshape,
                            tshape,
                            params,
                            d_stacks,
                            d_nstacks,
                            batchsizeZ,
                            d_maskGaussian,
                            d_maskSphere,
                            d_ref_patchs,
                            d_cmp_patchs,
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
                            bwIn, bwOut, degLim,
                            SNTspace_bsize,
                            d_prop);
    blockmatching_rot.stop();
    std::cout << "Blockmatching rot took: " << blockmatching_rot.getSeconds() << std::endl;
    free_cusoft_workspace();

    // // Gather cubes together
    // uint gather_stacks_sum;
    // Stopwatch gatheringcubes(true);
    // gather_cubes(d_noisy_volume, imshape, tshape, params, d_stacks, d_nstacks, d_gathered4dstack, gather_stacks_sum, d_prop);
    // // std::cout << "Acquied size " << gather_stacks_sum << std::endl;
    // gatheringcubes.stop();
    // std::cout << "Gathering cubes took: " << gatheringcubes.getSeconds() << std::endl;
    // // debug_kernel(d_gathered4dstack);

    // Gather cubes together
    uint gather_stacks_sum;
    Stopwatch gatheringcubes(true);
    gather_cubes_rot(d_noisy_volume_3d, imshape, tshape, params, d_stacks, d_nstacks, d_gathered4dstack, gather_stacks_sum, d_prop);
    // std::cout << "Acquied size " << gather_stacks_sum << std::endl;
    gatheringcubes.stop();
    std::cout << "Gathering cubes took: " << gatheringcubes.getSeconds() << std::endl;
    // debug_kernel(d_gathered4dstack);
    return noisy_volume;
    
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
    run_wht_ht_iwht(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_nstacks, tshape, d_group_weights, params, d_prop);
    wht_t.stop();
    std::cout << "WHT took: " << wht_t.getSeconds() << std::endl;

    // Perform inverse 3D DCT
    Stopwatch dct_backward(true);
    run_idct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);
    dct_backward.stop();
    std::cout << "3D DCT backwards took: " << dct_backward.getSeconds() << std::endl;
    // debug_kernel(d_gathered4dstack);


    // Aggregate
    // float* final_image = new float[width * height * depth];
    // memset(final_image, 0.0, sizeof(float) * width * height * depth);
    // Stopwatch aggregation_t(true);
    // run_aggregation(final_image, imshape, tshape, d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params, gather_stacks_sum, d_prop);
    // aggregation_t.stop();
    // std::cout << "Aggregation took: " << aggregation_t.getSeconds() << std::endl;
    // for (int i = 0; i < size; i++) {
    //     noisy_volume[i] = static_cast<uchar>(final_image[i]);
    // }
    // delete[] final_image;
    return noisy_volume;
}
