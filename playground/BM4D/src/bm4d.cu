/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 * 2020, Xingyu Zhu
 * jupiter.zhuxingyu [at] gmail [dot] com
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
}

void BM4D::init_rot_coords() {
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
    
    checkCudaErrors(cudaMalloc((void**)&dev_rel_coords, sizeof(float) * 3 * psize));
    checkCudaErrors(cudaMemcpy((void*)dev_rel_coords, (void*)rel_coords, sizeof(float) * 3 * psize, cudaMemcpyHostToDevice));
    t_rot_coords.stop(); std::cout<<"Initialize reference coords took: " << t_rot_coords.getSeconds() <<std::endl;
}

void BM4D::init_masks() {
    // Initialize the Spherical Gaussian Mask and Strict Spherical Mask
    // Need to initialize rot coords before this step

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
                dx = rel_coords[3 * d];
                dy = rel_coords[3 * d + 1];
                dz = rel_coords[3 * d + 2];
                sqr_dist = dx*dx + dy*dy + dz*dz;
                
                // Gaussian
                maskGaussian[d] = normal_pdf_sqr(std, sqr_dist);
                
                // Sphere
                if (sqr_dist <= sphere_tol) maskSphere[d] = 1.0;
                else maskSphere[d] = 0.0;
            }
    
    checkCudaErrors(cudaMalloc((void**)&dev_maskGaussian, sizeof(float) * psize));
    checkCudaErrors(cudaMemcpy((void*)dev_maskGaussian, (void*)maskGaussian, sizeof(float) * psize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&dev_maskSphere, sizeof(float) * psize));
    checkCudaErrors(cudaMemcpy((void*)dev_maskSphere, (void*)maskSphere, sizeof(float) * psize, cudaMemcpyHostToDevice));
    
    t_init_mask.stop(); std::cout<<"Initialize masks took: " << t_init_mask.getSeconds() <<std::endl;

    // for (int z = 0; z < k; ++z)
    //     for (int y = 0; y < k; ++y)
    //         for (int x = 0; x < k; ++x) {
    //             d = x + y * k + z * k * k;
    //             printf("%4f ", maskGaussian[d]);
    //             if (x == k-1 && y == k-1) printf("\n");
    //             if (x == k-1) printf("\n");
    //         }

    // for (int z = 0; z < k; ++z)
    //     for (int y = 0; y < k; ++y)
    //         for (int x = 0; x < k; ++x) {
    //             d = x + y * k + z * k * k;
    //             printf("%4f ", maskSphere[d]);
    //             if (x == k-1 && y == k-1) printf("\n");
    //             if (x == k-1) printf("\n");
    //         }
}

std::vector<uchar> BM4D::run_first_step() {
    // Stopwatch copyingtodevice(true);
    uchar* d_noisy_volume;
    Stopwatch copyingtodevice(true);
    assert(size == noisy_volume.size());

    load_3d_array();
    init_rot_coords();
    init_masks();

    return noisy_volume;

    // Old Memory Copy
    // TODO: Remove after surface reference is complete
    checkCudaErrors(cudaMalloc((void**)&d_noisy_volume, sizeof(uchar) * size));
    checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)noisy_volume.data(), sizeof(uchar) * size, cudaMemcpyHostToDevice));
    copyingtodevice.stop(); std::cout<<"Copying to device took:" << copyingtodevice.getSeconds() <<std::endl;

    uint3 im_size = make_uint3(width, height, depth);
    uint3 tr_size = make_uint3(twidth, theight, tdepth);    // Truncated size, with some step for ref patches
    
    // Do block matching
    Stopwatch blockmatching(true);
    run_block_matching(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_prop);
    run_block_matching_3d(noisy_volume_3d_surf, im_size, tr_size, params, d_stacks, d_nstacks, d_prop);
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
