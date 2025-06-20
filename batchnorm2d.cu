#ifndef BATCHNORM2D_CU
#define BATCHNORM2D_CU

#include <stdio.h>

__device__ void batchnorm2d(float* output,const float* input,const float* gamma,const float* beta,const float* mean,
    const float* var,int N, int C, int H, int W,float eps){
    int n = blockIdx.x;
    int c = threadIdx.x;
    
    if (n >= N || c >= C) return;

    int hw = H * W;
    float mean_val = mean[c];
    float var_val = var[c];
    float gamma_val = gamma ? gamma[c] : 1.0f;
    float beta_val = beta ? beta[c] : 0.0f;

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            float norm = (input[idx] - mean_val) / sqrtf(var_val + eps);
            output[idx] = norm * gamma_val + beta_val;
        }
    }
}
#endif
