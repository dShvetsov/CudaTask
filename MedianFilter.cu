#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <cuda.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "SOIL.h"

texture<uchar4, 2, cudaReadModeElementType> texImage;

void checkGPUOperation()
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess){
        fprintf(stderr, "Cuda Error : %s\n", cudaGetErrorString(code));
        exit(-1);
    }
}


__global__ void median_filter(uchar4 *pDst,
                              int width,
                              int height,
                              int radius)
{
    // compute idx
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    // arrays for region pixels
    unsigned char xArr[512], yArr[512], zArr[512], wArr[512]; //TODO : unhardore
    int arraySize = 0;

    // prevent texture out of range
    int fromx = thrust::max(0, tidx - radius);
    int tox = thrust::min(width, tidx + radius);
    int fromy = thrust::max(0, tidy - radius);
    int toy = thrust::min(height, tidy + radius);

    // collect region pixels
    for (int i = fromx; i < tox; i++){
        for (int j = fromy; j < toy; j++){
            uchar4 tmp = tex2D(texImage, i + 0.5f, j + 0.5f);
            xArr[arraySize] = tmp.x;
            yArr[arraySize] = tmp.y;
            zArr[arraySize] = tmp.z;
            wArr[arraySize] = tmp.w;
            arraySize++;
        }
    }

    //sort pixels
    thrust::sort(thrust::seq, xArr, xArr + arraySize);
    thrust::sort(thrust::seq, yArr, yArr + arraySize);
    thrust::sort(thrust::seq, zArr, zArr + arraySize);
    thrust::sort(thrust::seq, wArr, wArr + arraySize);

    // take the median. Middle pixels in sort array
    uchar4 res;
    res.x = xArr[arraySize / 2];
    res.y = yArr[arraySize / 2];;
    res.z = zArr[arraySize / 2];
    res.w = wArr[arraySize / 2];
    pDst[tidx + tidy * width] = res;
}

int main(int argc, char ** argv)
{

    //parsing arguments
    const char *srcImagePath = argc >= 2 ? argv[1] : "Lenna.png";
    const char *dstImagePath = argc >= 3 ? argv[2] : "result.tga";
    int radius  = argc >= 4 ? atoi(argv[3]) : 1;

    // loading image
    int width, height, channels;
    unsigned char* srcImage =
         SOIL_load_image( srcImagePath,
                          &width, &height, &channels,
                          SOIL_LOAD_RGBA /*unhardcode*/ );

    if (srcImage == NULL) {
        fprintf(stderr, "failed loading image");
        return -1;
    }
    int size = width * height * 4;
    printf("image loaded. width : %d, height %d\n", width, height);

    //create cuda array
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    checkGPUOperation();
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    checkGPUOperation();
    // copy image to texture
    cudaMemcpyToArray(cuArray, 0, 0, srcImage, size, cudaMemcpyHostToDevice);
    checkGPUOperation();

    // free unused memery
    SOIL_free_image_data(srcImage);

    // bind texture
    cudaBindTextureToArray(texImage, cuArray);
    checkGPUOperation();


    //allocate memory to result image
    uchar4 *devResult;
    cudaMalloc((void **)&devResult, size);

    // run kernel
    dim3 block(32, 8);
    dim3 grid( width / block.x + ((width % block.x) ? 1: 0),
            height / block.y + ((height % block.y) ? 1: 0) );
    median_filter<<<grid, block>>>(devResult, width, height, radius);
    cudaDeviceSynchronize();


    // copy result
    unsigned char* dstImage = (unsigned char *)malloc(size);
    cudaMemcpy(dstImage, devResult, size, cudaMemcpyDeviceToHost);

    SOIL_save_image( dstImagePath,
                     SOIL_SAVE_TYPE_TGA,
                     width, height, SOIL_LOAD_RGBA,
                     dstImage);

    //free memory
    cudaFreeArray(cuArray);
    cudaFree(devResult);
    free(dstImage);
    printf("Done\n");
    return 0;
}

