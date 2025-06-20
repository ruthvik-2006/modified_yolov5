#ifndef CONV2D_CU
#define CONV2D_CU
__device__ void convolution_kernel(float* output, const float* input, const float* weights, const float* bias,int N, int C_in, 
    int H_in, int W_in,int C_out, int H_out, int W_out,int K_h,
    int K_w,int stride_h, int stride_w,int pad_h, int pad_w,int dilate_h, int dilate_w) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    int n = blockIdx.z / C_out;
    int c_out = blockIdx.z % C_out;

    if (w_out >= W_out || h_out >= H_out || n >= N) {
        return;
    }


    float accumulator = bias[c_out];

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh * dilate_h;
                int w_in = w_out * stride_w - pad_w + kw * dilate_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                    int weight_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + kh * K_w + kw;
                    accumulator += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    int output_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
    output[output_idx] = accumulator;
}

#endif