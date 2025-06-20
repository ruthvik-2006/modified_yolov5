#ifndef SEQUENTIAL_CU
#define SEQUENTIAL_CU

#include <stdio.h>

__device__ void sequential_dummy_kernel(float* output, const float* input, long long num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx];
    }
}

#endif