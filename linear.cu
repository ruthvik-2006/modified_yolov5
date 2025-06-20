#ifndef LINEAR_CU
#define LINEAR_CU

#include<stdio.h>
__device__ void linear(float* output,const float* input,const float* weights,const float* bias,
    int batch_size,int input_features,int output_features){
        int batch_index=blockIdx.y*blockDim.y+threadIdx.y;
        int out_index=blockIdx.x*blockDim.x+threadIdx.x;
        if(batch_index>=batch_size || out_index>=output_features)return;
        float result=bias ? bias[out_index]:0.0f; // check if bias is 0 or !0 assigning acccordingly.

        //m*n and n*k --> m*k we have m and k...so traversing on n -> in_features.
        for(int i=0;i<input_features;i++){
            result=result+(input[batch_index*input_features+i]*weights[out_index*input_features+i]);
        }

        output[batch_index*output_features+out_index]=result;

}

#endif
