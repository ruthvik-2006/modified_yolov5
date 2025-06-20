#ifndef SILU_CU
#define SILU_CU

#include<float.h>
#include<math.h>

__device__ void Silu(const float* input,float* output,long long num_elements){
    long long index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=num_elements) return;
    float sigmoid_val=1.0f/(1.0f+expf(-input[index]));
    output[index]=sigmoid_val*input[index];
}

#endif