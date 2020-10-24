// Simple copy kernel
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                           cudaSurfaceObject_t outputSurfObj,
                           int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfObj, x * 4, y);
        // Write to output surface
        surf2Dwrite(data, outputSurfObj, x * 4, y);
    }
}

// Host code
int main()
{
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // Copy to device memory some data located at address h_data
    // in host memory 
    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    copyKernel<<<dimGrid, dimBlock>>>(inputSurfObj,
                                      outputSurfObj,
                                      width, height);


    // Destroy surface objects
    cudaDestroySurfaceObject(inputSurfObj);
    cudaDestroySurfaceObject(outputSurfObj);

    // Free device memory
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);

    return 0;
}

#include <helper_cuda.h>
#include <curand.h>
#define NUM_TEX 4

const int SizeNoiseTest = 32;
const int cubeSizeNoiseTest = SizeNoiseTest*SizeNoiseTest*SizeNoiseTest;
static cudaTextureObject_t texNoise[NUM_TEX];

__global__ void AccesTexture(cudaTextureObject_t my_tex)
{
    float test = tex3D<float>(my_tex,(float)threadIdx.x,(float)threadIdx.y,(float)threadIdx.z);//by using this the error occurs
    printf("thread: %d,%d,%d, value: %f\n", threadIdx.x, threadIdx.y, threadIdx.z, test);
}

void CreateTexture()
{

    float *d_NoiseTest;//Device Array with random floats
    cudaMalloc((void **)&d_NoiseTest, cubeSizeNoiseTest*sizeof(float));//Allocation of device Array
    for (int i = 0; i < NUM_TEX; i++){
    //curand Random Generator (needs compiler link -lcurand)
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1235ULL+i);
    curandGenerateUniform(gen, d_NoiseTest, cubeSizeNoiseTest);//writing data to d_NoiseTest
    curandDestroyGenerator(gen);

    //cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cuda Array
    cudaArray *d_cuArr;
    checkCudaErrors(cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(SizeNoiseTest*sizeof(float),SizeNoiseTest,SizeNoiseTest), 0));
    cudaMemcpy3DParms copyParams = {0};


    //Array creation
    copyParams.srcPtr   = make_cudaPitchedPtr(d_NoiseTest, SizeNoiseTest*sizeof(float), SizeNoiseTest, SizeNoiseTest);
    copyParams.dstArray = d_cuArr;
    copyParams.extent   = make_cudaExtent(SizeNoiseTest,SizeNoiseTest,SizeNoiseTest);
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
    //Array creation End

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = d_cuArr;
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&texNoise[i], &texRes, &texDescr, NULL));}
}

int main(int argc, char **argv)
{
    CreateTexture();
    AccesTexture<<<1,dim3(2,2,2)>>>(texNoise[0]);
    AccesTexture<<<1,dim3(2,2,2)>>>(texNoise[1]);
    AccesTexture<<<1,dim3(2,2,2)>>>(texNoise[2]);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}