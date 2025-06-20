#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "conv2d.cu"
#include "Silu.cu"
#include "batchnorm2d.cu"
#include "linear.cu"
#include "maxpool2d.cu"
#include "upsample.cu"
#include "sequential.cu" 

using namespace std;

__device__ pair<int, int> calculate_conv_output_dims(int H_in, int W_in, int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilate_h, int dilate_w) {
    int H_out = (H_in + 2 * pad_h - dilate_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilate_w * (K_w - 1) - 1) / stride_w + 1;
    return {H_out, W_out};
}

__device__ void device_convolution(float* output, const float* input, const float* weights, const float* bias,
                                   int N, int C_in, int H_in, int W_in,
                                   int C_out, int H_out, int W_out,
                                   int K_h, int K_w,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dilate_h, int dilate_w) {
    convolution_kernel(output, input, weights, bias, N, C_in, H_in, W_in,
                C_out, H_out, W_out, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate_h, dilate_w);
}

__device__ void device_silu(const float* input, float* output, long long num_elements) {
    Silu(input, output, num_elements);
}

__device__ void device_batchnorm2d(float* output, const float* input, const float* gamma, const float* beta,
    const float* mean, const float* var, int N, int C, int H, int W, float eps) {
    batchnorm2d(output, input, gamma, beta, mean, var, N, C, H, W, eps);
}

__device__ void device_linear(float* output,const float* input,const float* weights,const float* bias,
    int batch_size,int input_features,int output_features){
    linear(output, input, weights, bias, batch_size, input_features, output_features);
}

__device__ void device_maxpool2d(const float* input, float* output,
                                 int N, int C, int H_in, int W_in,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w,
                                 int pad_h, int pad_w,
                                 int dilation_h, int dilation_w,
                                 int H_out, int W_out) {
    maxpool2d(input, output, N, C, H_in, W_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
              dilation_h, dilation_w, H_out, W_out);
}

__device__ void device_upsample(const float* input, float* output,
                                int N, int C, int H_in, int W_in,
                                int H_out, int W_out) {
    upsample_nearest(input, output, N, C, H_in, W_in, H_out, W_out);
}

__device__ void device_sequential(float* output, const float* input, long long num_elements) {
    sequential_dummy_kernel(output, input, num_elements);
}

extern "C" void convolution_layer(float* output_host, const float* input_host, const float* weights_host, const float* bias_host,
                                  int N, int C_in, int H_in, int W_in,
                                  int C_out, int H_out, int W_out,
                                  int K_h, int K_w,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dilate_h, int dilate_w) {
    size_t input_size = (size_t)N * C_in * H_in * W_in * sizeof(float);
    size_t output_size = (size_t)N * C_out * H_out * W_out * sizeof(float);
    size_t weights_size = (size_t)C_out * C_in * K_h * K_w * sizeof(float);
    size_t bias_size = (size_t)C_out * sizeof(float);

    float *d_input, *d_output, *d_weights, *d_bias;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);
    cudaMalloc((void**)&d_weights, weights_size);
    cudaMalloc((void**)&d_bias, bias_size);

    cudaMemcpy(d_input, input_host, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_host, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias_host, bias_size, cudaMemcpyHostToDevice );

    int total_elements = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    device_convolution<<<num_blocks, threads_per_block>>>(
        d_output, d_input, d_weights, d_bias, N, C_in, H_in, W_in,
        C_out, H_out, W_out, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate_h, dilate_w
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_bias);
}

extern "C" void silu_layer(const float* input_host, float* output_host, long long num_elements) {
    size_t data_size = num_elements * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, data_size);
    cudaMalloc((void**)&d_output, data_size);

    cudaMemcpy(d_input, input_host, data_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    device_silu<<<num_blocks, threads_per_block>>>(d_input, d_output, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

extern "C" void batchnorm2d_layer(float* output_host, const float* input_host, const float* gamma_host, const float* beta_host,
                                  const float* mean_host, const float* var_host, int N, int C, int H, int W, float eps) {
    size_t size = (size_t)N * C * H * W * sizeof(float);
    size_t ch_size = (size_t)C * sizeof(float);

    float *d_input, *d_output, *d_gamma, *d_beta, *d_mean, *d_var;

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_gamma, ch_size);
    cudaMalloc((void**)&d_beta, ch_size);
    cudaMalloc((void**)&d_mean, ch_size);
    cudaMalloc((void**)&d_var, ch_size);

    cudaMemcpy(d_input, input_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma_host, ch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta_host, ch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean_host, ch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_var, var_host, ch_size, cudaMemcpyHostToDevice);

    // For Batchnorm2d, thread configuration based on total elements
    int total_threads = N * C * H * W;
    int blocksize = 256;
    int nblocks = (total_threads + blocksize - 1) / blocksize;

    device_batchnorm2d<<<nblocks, blocksize>>>(
        d_output, d_input, d_gamma, d_beta, d_mean, d_var, N, C, H, W, eps
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_mean);
    cudaFree(d_var);
}

extern "C" void linear_layer(float* output_host,const float* input_host,const float* weights_host,const float* bias_host,
    int batch_size,int input_features,int output_features) {
    size_t input_size = (size_t)batch_size * input_features * sizeof(float);
    size_t output_size = (size_t)batch_size * output_features * sizeof(float);
    size_t weights_size = (size_t)input_features * output_features * sizeof(float);
    size_t bias_size = (size_t)output_features * sizeof(float);

    float *d_input, *d_output, *d_weights, *d_bias;
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);
    cudaMalloc((void**)&d_weights, weights_size);
    cudaMalloc((void**)&d_bias, bias_size);

    cudaMemcpy(d_input, input_host, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_host, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias_host, bias_size, cudaMemcpyHostToDevice);

    int total_threads = batch_size * output_features;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    device_linear<<<num_blocks, threads_per_block>>>(
        d_output, d_input, d_weights, d_bias, batch_size, input_features, output_features
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_bias);
}

extern "C" void maxpool2d_layer(const float* input_host, float* output_host,
                                int N, int C, int H_in, int W_in,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int H_out, int W_out) {
    size_t input_size = (size_t)N * C * H_in * W_in * sizeof(float);
    size_t output_size = (size_t)N * C * H_out * W_out * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, input_host, input_size, cudaMemcpyHostToDevice);

    int total_elements = N * C * H_out * W_out;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    device_maxpool2d<<<num_blocks, threads_per_block>>>(
        d_input, d_output, N, C, H_in, W_in,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, H_out, W_out
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

extern "C" void upsample_layer(const float* input_host, float* output_host,
                                int N, int C, int H_in, int W_in,
                                int H_out, int W_out) {
    size_t input_size = (size_t)N * C * H_in * W_in * sizeof(float);
    size_t output_size = (size_t)N * C * H_out * W_out * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, input_host, input_size, cudaMemcpyHostToDevice);

    int total_threads = N * C * H_out * W_out;
    int blocksize = 256;
    int nblocks = (total_threads + blocksize - 1) / blocksize;

    device_upsample<<<nblocks, blocksize>>>(d_input, d_output,
                                            N, C, H_in, W_in, H_out, W_out);
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

extern "C" void sequential_layer_dummy(float* output_host, const float* input_host, long long num_elements) {
    size_t data_size = num_elements * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, data_size);
    cudaMalloc((void**)&d_output, data_size);

    cudaMemcpy(d_input, input_host, data_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    device_sequential<<<num_blocks, threads_per_block>>>(d_output, d_input, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(output_host, d_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


__device__ void apply_conv_block(const vector<float>& input, vector<float>& output,
                      int N, int C_in, int H_in, int W_in,
                      int C_out, int K_h, int K_w,
                      int stride_h, int stride_w, int pad_h, int pad_w,
                      int dilate_h, int dilate_w) {

    vector<float> weights(C_out * C_in * K_h * K_w, 1.0f);
    vector<float> bias(C_out, 0.0f);
    
    pair<int, int> output_dims = calculate_conv_output_dims(H_in, W_in, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate_h, dilate_w);
    int H_out = output_dims.first;
    int W_out = output_dims.second;

    vector<float> conv_tmp_output(N * C_out * H_out * W_out);

    convolution_layer(conv_tmp_output.data(), input.data(), weights.data(), bias.data(),
                      N, C_in, H_in, W_in,
                      C_out, H_out, W_out,
                      K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate_h, dilate_w);

    vector<float> gamma(C_out, 1.0f);
    vector<float> beta(C_out, 0.0f);
    vector<float> mean(C_out, 0.5f);
    vector<float> var(C_out, 1.0f);
    float eps = 1e-5f;

    vector<float> bn_tmp_output(conv_tmp_output.size());
    batchnorm2d_layer(bn_tmp_output.data(), conv_tmp_output.data(), gamma.data(), beta.data(),
                      mean.data(), var.data(), N, C_out, H_out, W_out, eps);

    output.resize(bn_tmp_output.size());
    silu_layer(bn_tmp_output.data(), output.data(), bn_tmp_output.size());
}



__device__ void apply_bottleneck_block(const vector<float>& input, vector<float>& output,
                            int N, int C_in, int H_in, int W_in,
                            bool shortcut, int g) {

    int c_bottleneck_internal_hidden = C_in; 

    vector<float> cv1_output;
    apply_conv_block(input, cv1_output,
                     N, C_in, H_in, W_in,
                     c_bottleneck_internal_hidden, 1, 1, 1, 1, 0, 0, 1, 1);

    vector<float> cv2_output;
    apply_conv_block(cv1_output, cv2_output,
                     N, c_bottleneck_internal_hidden, H_in, W_in,
                     C_in, 3, 3, 1, 1, 1, 1, 1, 1);

    if (shortcut && (C_in == C_in)) {
        output.resize(cv2_output.size());
        for (size_t i = 0; i < cv2_output.size(); ++i) {
            output[i] = cv2_output[i] + input[i];
        }
    } else {
        output = cv2_output;
    }
}

__device__ void apply_sppf_block(const vector<float>& input, vector<float>& output,
                      int N, int C_in, int H_in, int W_in,
                      int C_out, int k_maxpool) {

    int c_hidden = C_in / 2;
    int k_pad = k_maxpool / 2; // For 'same' padding with stride 1

    vector<float> cv1_output;
    apply_conv_block(input, cv1_output,
                     N, C_in, H_in, W_in,
                     c_hidden, 1, 1, 1, 1, 0, 0, 1, 1);
    
    // MaxPool output dimensions will be the same as input when stride is 1 and padding is k/2
    int H_mp_out = (H_in + 2 * k_pad - 1 * (k_maxpool - 1) - 1) / 1 + 1;
    int W_mp_out = (W_in + 2 * k_pad - 1 * (k_maxpool - 1) - 1) / 1 + 1;

    vector<float> mp1_output(N * c_hidden * H_mp_out * W_mp_out);
    maxpool2d_layer(cv1_output.data(), mp1_output.data(),
                    N, c_hidden, H_in, W_in, // input dimensions
                    k_maxpool, k_maxpool, 1, 1, k_pad, k_pad, 1, 1, // kernel, stride, pad, dilate
                    H_mp_out, W_mp_out); // output dimensions

    vector<float> mp2_output(N * c_hidden * H_mp_out * W_mp_out);
    maxpool2d_layer(mp1_output.data(), mp2_output.data(),
                    N, c_hidden, H_mp_out, W_mp_out, // input dimensions are previous output
                    k_maxpool, k_maxpool, 1, 1, k_pad, k_pad, 1, 1,
                    H_mp_out, W_mp_out);

    vector<float> mp3_output(N * c_hidden * H_mp_out * W_mp_out);
    maxpool2d_layer(mp2_output.data(), mp3_output.data(),
                    N, c_hidden, H_mp_out, W_mp_out, // input dimensions are previous output
                    k_maxpool, k_maxpool, 1, 1, k_pad, k_pad, 1, 1,
                    H_mp_out, W_mp_out);

    int concated_channels = c_hidden * 4;
    vector<float> concatenated_sppf_output(N * concated_channels * H_in * W_in);

    // Concatenation on CPU
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_in; ++h) {
            for (int w = 0; w < W_in; ++w) {
                for (int c = 0; c < c_hidden; ++c) {
                    concatenated_sppf_output[((n * concated_channels + c) * H_in + h) * W_in + w] =
                        cv1_output[((n * c_hidden + c) * H_in + h) * W_in + w];
                }
                for (int c = 0; c < c_hidden; ++c) {
                    concatenated_sppf_output[((n * concated_channels + c_hidden + c) * H_in + h) * W_in + w] =
                        mp1_output[((n * c_hidden + c) * H_in + h) * W_in + w];
                }
                for (int c = 0; c < c_hidden; ++c) {
                    concatenated_sppf_output[((n * concated_channels + 2 * c_hidden + c) * H_in + h) * W_in + w] =
                        mp2_output[((n * c_hidden + c) * H_in + h) * W_in + w];
                }
                for (int c = 0; c < c_hidden; ++c) {
                    concatenated_sppf_output[((n * concated_channels + 3 * c_hidden + c) * H_in + h) * W_in + w] =
                        mp3_output[((n * c_hidden + c) * H_in + h) * W_in + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_sppf_output, output,
                     N, concated_channels, H_in, W_in,
                     C_out, 1, 1, 1, 1, 0, 0, 1, 1);
}




// ====================================================================
//calls !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// ====================================================================
__global__ void full_inference(float* input, float* output, int N, int H, int W, int C) {
    extern __shared__ float shared[];
    float* buffer1 = shared;
    float* buffer2 = buffer1 + N * 2048 * H * W;
    float* current_input = input;
    float* current_output = buffer1;
    int N_batch=1;
    int C_current = C;
    int H_current = H;
    int W_current = W;

    int K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate;
    int C_out;

    K_h = 6; K_w = 6; stride_h = 2; stride_w = 2; pad_h = 2; pad_w = 2; dilate = 1;
    C_out = 64;

    // Layer 0 (Conv)
    K_h = 6; K_w = 6; stride_h = 2; stride_w = 2; pad_h = 2; pad_w = 2; dilate = 1;
    C_out = 64;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L0 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L0.first;
    W_current = dims_L0.second;
    cout << "Layer 0 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 1 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 128;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L1 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L1.first;
    W_current = dims_L1.second;
    cout << "Layer 1 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 2 (C3)
    int c1_c3 = C_current;
    int c2_c3 = 128;
    int n_bottlenecks = 3;
    bool shortcut_c3 = true;
    int g_c3 = 1;
    float e_c3 = 0.5f;

    int c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    vector<float> cv1_output_c3;
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    vector<float> cv2_output_c3;
    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    vector<float> m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    int m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    int cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    int concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    vector<float> concatenated_output_c3(N_batch * concated_C_c3 * H_current * W_current);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 2 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 3 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 256;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L3 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L3.first;
    W_current = dims_L3.second;
    cout << "Layer 3 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 4 (C3)
    c1_c3 = C_current;
    c2_c3 = 256;
    n_bottlenecks = 6;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;

    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    cv1_output_c3.clear(); cv2_output_c3.clear(); m_output_c3.clear();
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3.assign(N_batch * concated_C_c3 * H_current * W_current, 0.0f);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 4 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 5 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 512;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L5 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L5.first;
    W_current = dims_L5.second;
    cout << "Layer 5 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 6 (C3)
    c1_c3 = C_current;
    c2_c3 = 512;
    n_bottlenecks = 9;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;

    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    cv1_output_c3.clear(); cv2_output_c3.clear(); m_output_c3.clear();
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3.assign(N_batch * concated_C_c3 * H_current * W_current, 0.0f);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 6 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 7 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 1024;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L7 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L7.first;
    W_current = dims_L7.second;
    cout << "Layer 7 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 8 (C3)
    c1_c3 = C_current;
    c2_c3 = 1024;
    n_bottlenecks = 3;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;

    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    cv1_output_c3.clear(); cv2_output_c3.clear(); m_output_c3.clear();
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3.assign(N_batch * concated_C_c3 * H_current * W_current, 0.0f);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 8 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 9: SPPF [1024, 5]
    int C_in9 = C_current; // Should be 1024 from Layer 8
    int C_out9 = 1024;
    int k_sppf = 5;
    apply_sppf_block(current_input, current_output,
                     N_batch, C_in9, H_current, W_current,
                     C_out9, k_sppf);
    C_current = C_out9;
    // H_current and W_current remain unchanged due to stride 1 and padding k//2 in MaxPool
    cout << "Layer 9 (SPPF) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;


    cout << "\nAll backbone layers have been processed from all inference device function" << endl;
    cout << "Final output of backbone: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;

    output=current_output;
    

    cout << "Sample of final output ****(after Layer 9 SPPF block) ***** from device all_inference:" << endl;
    for (int i = 0; i < min((int)current_output.size(), 10); ++i) {
        cout << current_output[i] << " ";
    }
    cout << endl;


    

}




int main() {
    int N = 1, H = 5, W = 5, C = 1;
    int total_elements = N * C * H * W;
    vector<float> host_input(total_elements, 1.0f);
    vector<float> host_output(N * 1024 * H * W);

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, total_elements * sizeof(float));
    cudaMalloc(&d_output, host_output.size() * sizeof(float));

    cudaMemcpy(d_input, host_input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    size_t shared_size = 2 * 2048 * H * W * sizeof(float) + 2048 * 2048 * sizeof(float);
    full_inference<<<1, 1, shared_size>>>(d_input, d_output, N, H, W, C);
    cudaMemcpy(host_output.data(), d_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Output sample:" << endl;
    for (int i = 0; i < min(10, (int)host_output.size()); ++i) cout << host_output[i] << " ";
    cout << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}




/*int main() {
    int N_batch = 1;

    int H_current = 5;
    int W_current = 5;
    int C_current = 1;
    vector<float> current_input(N_batch * C_current * H_current * W_current, 1.0f);
    vector<float> current_output;

    int K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate;
    int C_out;

    // Layer 0 (Conv)
    K_h = 6; K_w = 6; stride_h = 2; stride_w = 2; pad_h = 2; pad_w = 2; dilate = 1;
    C_out = 64;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L0 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L0.first;
    W_current = dims_L0.second;
    cout << "Layer 0 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 1 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 128;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L1 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L1.first;
    W_current = dims_L1.second;
    cout << "Layer 1 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 2 (C3)
    int c1_c3 = C_current;
    int c2_c3 = 128;
    int n_bottlenecks = 3;
    bool shortcut_c3 = true;
    int g_c3 = 1;
    float e_c3 = 0.5f;

    int c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    vector<float> cv1_output_c3;
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    vector<float> cv2_output_c3;
    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    vector<float> m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    int m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    int cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    int concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    vector<float> concatenated_output_c3(N_batch * concated_C_c3 * H_current * W_current);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 2 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 3 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 256;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L3 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L3.first;
    W_current = dims_L3.second;
    cout << "Layer 3 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 4 (C3)
    c1_c3 = C_current;
    c2_c3 = 256;
    n_bottlenecks = 6;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;

    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    cv1_output_c3.clear(); cv2_output_c3.clear(); m_output_c3.clear();
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3.assign(N_batch * concated_C_c3 * H_current * W_current, 0.0f);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 4 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 5 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 512;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L5 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L5.first;
    W_current = dims_L5.second;
    cout << "Layer 5 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 6 (C3)
    c1_c3 = C_current;
    c2_c3 = 512;
    n_bottlenecks = 9;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;

    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    cv1_output_c3.clear(); cv2_output_c3.clear(); m_output_c3.clear();
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3.assign(N_batch * concated_C_c3 * H_current * W_current, 0.0f);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 6 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 7 (Conv)
    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 1024;
    apply_conv_block(current_input, current_output,
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate);
    C_current = C_out;
    pair<int, int> dims_L7 = calculate_conv_output_dims(H_current, W_current, K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate, dilate);
    H_current = dims_L7.first;
    W_current = dims_L7.second;
    cout << "Layer 7 (Conv) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 8 (C3)
    c1_c3 = C_current;
    c2_c3 = 1024;
    n_bottlenecks = 3;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;

    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3);

    cv1_output_c3.clear(); cv2_output_c3.clear(); m_output_c3.clear();
    apply_conv_block(current_input, cv1_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    apply_conv_block(current_input, cv2_output_c3,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1);

    m_output_c3 = cv1_output_c3;

    for (int i = 0; i < n_bottlenecks; ++i) {
        vector<float> current_m_input = m_output_c3; 
        apply_bottleneck_block(current_m_input, m_output_c3,
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3);
    }
    
    m_output_channels_c3 = m_output_c3.size() / (N_batch * H_current * W_current);
    cv2_output_channels_c3 = cv2_output_c3.size() / (N_batch * H_current * W_current);
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3.assign(N_batch * concated_C_c3 * H_current * W_current, 0.0f);

    for (int n = 0; n < N_batch; ++n) {
        for (int h = 0; h < H_current; ++h) {
            for (int w = 0; w < W_current; ++w) {
                for (int c = 0; c < m_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + c) * H_current + h) * W_current + w] =
                        m_output_c3[((n * m_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
                for (int c = 0; c < cv2_output_channels_c3; ++c) {
                    concatenated_output_c3[((n * concated_C_c3 + m_output_channels_c3 + c) * H_current + h) * W_current + w] =
                        cv2_output_c3[((n * cv2_output_channels_c3 + c) * H_current + h) * W_current + w];
                }
            }
        }
    }

    apply_conv_block(concatenated_output_c3, current_output,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1);
    C_current = c2_c3;
    cout << "Layer 8 (C3) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    // Layer 9: SPPF [1024, 5]
    int C_in9 = C_current; // Should be 1024 from Layer 8
    int C_out9 = 1024;
    int k_sppf = 5;
    apply_sppf_block(current_input, current_output,
                     N_batch, C_in9, H_current, W_current,
                     C_out9, k_sppf);
    C_current = C_out9;
    // H_current and W_current remain unchanged due to stride 1 and padding k//2 in MaxPool
    cout << "Layer 9 (SPPF) output: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;
    current_input = current_output;

    cout << "\nAll backbone layers have been processed." << endl;
    cout << "Final output of backbone: C=" << C_current << ", H=" << H_current << ", W=" << W_current << endl;

    cout << "Sample of final output (after Layer 9 SPPF block):" << endl;
    for (int i = 0; i < min((int)current_output.size(), 10); ++i) {
        cout << current_output[i] << " ";
    }
    cout << endl;

    return 0;
}*/
