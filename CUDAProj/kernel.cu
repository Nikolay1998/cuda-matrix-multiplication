
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <ctime>

#define BLOCK_SIZE 16
#define BASE_TYPE double 


cudaError_t addWithCuda(BASE_TYPE *c, const BASE_TYPE *a, const BASE_TYPE *b, unsigned int size);

__global__ void addKernel(BASE_TYPE *c, const BASE_TYPE *a, const BASE_TYPE *b, int size)
{
    int i0 = size * (blockDim.y * blockIdx.y +
        threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;
    BASE_TYPE sum = 0;
    for (int k = 0; k < size; k++) {
        sum += a[i0 + k] * b[k * size + j0];
    }
    int ind = size * (blockDim.y * blockIdx.y +
        threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    c[ind] = sum;
}

int cons_mult(BASE_TYPE* a, BASE_TYPE* b, BASE_TYPE* c, int size) {
    unsigned int start_time2 = clock();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            BASE_TYPE local_result = 0;
            for (int t = 0; t < size; t++) {
                local_result += a[i * size + t] * b[t * size + j];
            }
            c[i * size + j] = local_result;
        }
    }
    unsigned int end_time2 = clock();
    return end_time2 - start_time2 + 1;
}

int main()
{
    const int mat_size = 1600;
    BASE_TYPE a[mat_size * mat_size];
    BASE_TYPE b[mat_size * mat_size];

    for (int i = 0; i < mat_size * mat_size; i++) {
        *(a + i) = i;
        *(b + i) = i;
    }

    BASE_TYPE c[mat_size * mat_size] = { 0 };

    // Add vectors in parallel.
    
    int start_time = clock();
    cudaError_t cudaStatus = addWithCuda(c, a, b, mat_size);
    int end_time = clock();
    int search_time = end_time - start_time;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    std::cout << "\nParallel multiplication time:" << search_time << "\nParallel multiplication result:\n";
    std::cout << '\n' << c[800 * mat_size + 800];
    /*
    for (int i = 0; i < mat_size; i++) {
        std::cout << '\n';
        for (int j = 0; j < mat_size; j++) {
            std::cout << c[i* mat_size +j] << ' ';
        }
    }
    */
     // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    

    int l = cons_mult(a, b, c, mat_size);
    std::cout << "\nConsicutive multiplication time:" << l << "\nConsicutive multiplication result:\n";
    std::cout << '\n' << c[800 * mat_size + 800];
    /*
    for (int i = 0; i < mat_size; i++) {
        std::cout << '\n';
        for (int j = 0; j < mat_size; j++) {
            std::cout << c[i * mat_size + j] << ' ';
        }
    }
    */
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(BASE_TYPE *c, const BASE_TYPE *a, const BASE_TYPE *b, unsigned int size)
{
    BASE_TYPE *dev_a = 0;
    BASE_TYPE *dev_b = 0;
    BASE_TYPE *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(BASE_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(BASE_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(BASE_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(size / BLOCK_SIZE, size / BLOCK_SIZE);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
