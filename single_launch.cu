#include <cuda_runtime.h>
#include <stdio.h> 
#include <algorithm> 
#include <utility> 
#include <math.h>
#include <cmath>
#include <cfloat>  
#include <float.h> 
#include <random>
using namespace std;

__device__ float compute_convolution_element(const float* input, const float* weights, const float* bias,
                                             int n, int c_out, int h_out, int w_out,
                                             int C_in, int H_in, int W_in,
                                             int K_h, int K_w,
                                             int stride_h, int stride_w, int pad_h, int pad_w,
                                             int dilate_h, int dilate_w) {
    float accumulator = bias[c_out]; 

    for (int c_in_idx = 0; c_in_idx < C_in; ++c_in_idx) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh * dilate_h;
                int w_in = w_out * stride_w - pad_w + kw * dilate_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    
                    int input_idx = n * C_in * H_in * W_in + c_in_idx * H_in * W_in + h_in * W_in + w_in;
                    int weight_idx = c_out * C_in * K_h * K_w + c_in_idx * K_h * K_w + kh * K_w + kw;
                    accumulator += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    return accumulator;
}


__device__ void Silu_forward_elementwise(const float* input, float* output, int total_elements) {
    long long index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=total_elements) return;
    float x = input[index];
    x = fmaxf(fminf(x, 80.0f), -80.0f);  // Clamp input to prevent overflow (to prevent nan)
    float sigmoid_val=1.0f/(1.0f+expf(-x));
    output[index]=sigmoid_val*input[index];
}

__device__ float Silu_element(float x) {
    return x / (1.0f + expf(-x));
}



__device__ void batchnorm2d_forward_elementwise(const float* input, float* output,
                                                const float* gamma, const float* beta,
                                                const float* mean, const float* var,
                                                int N, int C, int H, int W, float eps) {
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
            // float norm = (input[idx] - mean_val) / sqrtf(var_val + eps);
            // output[idx] = norm * gamma_val + beta_val;
            float denom = sqrtf(var_val + eps); //changed because of nan values
            if (denom < 1e-8f || isnan(denom) || isinf(denom)) denom = 1.0f;
            float norm = (input[idx] - mean_val) / denom;
            output[idx] = norm * gamma_val + beta_val;
        }
    }
    
}

__device__ float batchnorm2d_element(float val, float gamma, float beta, float mean, float var, float eps) {
    // return gamma * ((val - mean) / sqrtf(var + eps)) + beta;
    float denom = sqrtf(var + eps);
    if (denom < 1e-8f || isnan(denom) || isinf(denom)) denom = 1.0f;
    return gamma * ((val - mean) / denom) + beta;

}

__device__ float maxpool2d_element(const float* input,
                                   int n, int c, int h_out, int w_out,
                                   int C_in, int H_in, int W_in,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w) { 
    float max_val = -FLT_MAX; 

    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_start + kh; 
            int w_in = w_start + kw; 

            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                
                int input_idx = n * C_in * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                }
            }
        }
    }
    return max_val;
}

pair<int, int> calculate_conv_output_dims(int H_in, int W_in, int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilate_h, int dilate_w) {
    int H_out = (H_in + 2 * pad_h - dilate_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilate_w * (K_w - 1) - 1) / stride_w + 1;
    return {H_out, W_out};
}



__device__ void apply_conv_block(const float* input_ptr, float* output_ptr,
                                 const float* weights_ptr, const float* bias_ptr,
                                 const float* gamma_ptr, const float* beta_ptr,
                                 const float* mean_ptr, const float* var_ptr,
                                 int N, int C_in, int H_in, int W_in,
                                 int C_out, int K_h, int K_w,
                                 int stride_h, int stride_w, int pad_h, int pad_w,
                                 int dilate_h, int dilate_w,
                                 int H_out, int W_out, float eps,
                                 float* temp_buffer_conv, float* temp_buffer_bn) {
    int total_output_elements = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Simplified for 1D launch in main, adjust if 3D
    int c_out_idx=0;
    if (idx < total_output_elements) {
        
        int w_idx = idx % W_out;
        int h_idx = (idx / W_out) % H_out;
        int c_out_idx = (idx / (W_out * H_out)) % C_out;
        int n_idx = idx / (C_out * H_out * W_out);
        float conv_val = compute_convolution_element(input_ptr, weights_ptr, bias_ptr,
                                                    n_idx, c_out_idx, h_idx, w_idx,
                                                    C_in, H_in, W_in,
                                                    K_h, K_w,
                                                    stride_h, stride_w, pad_h, pad_w,
                                                    dilate_h, dilate_w);
        temp_buffer_conv[idx] = conv_val;
    }
    __syncthreads(); 

    if (idx < total_output_elements) {
        float bn_val = batchnorm2d_element(temp_buffer_conv[idx],
                                           gamma_ptr[c_out_idx], beta_ptr[c_out_idx],
                                           mean_ptr[c_out_idx], var_ptr[c_out_idx], eps);
        temp_buffer_bn[idx] = bn_val;
    }
    __syncthreads();

    if (idx < total_output_elements) {
        output_ptr[idx] = Silu_element(temp_buffer_bn[idx]);
    }
    __syncthreads();
}



__device__ void apply_bottleneck_block(const float* input_ptr, float* output_ptr,
                                       const float* weights_cv1_ptr, const float* bias_cv1_ptr,
                                       const float* gamma_cv1_ptr, const float* beta_cv1_ptr, const float* mean_cv1_ptr, const float* var_cv1_ptr,
                                       const float* weights_cv2_ptr, const float* bias_cv2_ptr,
                                       const float* gamma_cv2_ptr, const float* beta_cv2_ptr, const float* mean_cv2_ptr, const float* var_cv2_ptr,
                                       int N, int C_in, int H_in, int W_in,
                                       bool shortcut, int g,
                                       int c_hidden_conv1, int c_hidden_conv2, float eps,
                                       float* temp_conv1_out_ptr, float* temp_conv1_bn_ptr,
                                       float* temp_conv2_out_ptr, float* temp_conv2_bn_ptr) {

   
    int H_intermediate = H_in;
    int W_intermediate = W_in;

    
    apply_conv_block(input_ptr, temp_conv1_out_ptr,
                     weights_cv1_ptr, bias_cv1_ptr,
                     gamma_cv1_ptr, beta_cv1_ptr, mean_cv1_ptr, var_cv1_ptr,
                     N, C_in, H_in, W_in,
                     c_hidden_conv1, 1, 1, 
                     1, 1, 0, 0, 
                     1, 1, 
                     H_intermediate, W_intermediate, eps,
                     temp_conv1_bn_ptr,
                     temp_conv1_bn_ptr + N * c_hidden_conv1 * H_intermediate * W_intermediate // A separate temp buffer for BN to SiLU
                     );
    __syncthreads(); 

    
    apply_conv_block(temp_conv1_out_ptr, temp_conv2_out_ptr, 
                     weights_cv2_ptr, bias_cv2_ptr,
                     gamma_cv2_ptr, beta_cv2_ptr, mean_cv2_ptr, var_cv2_ptr,
                     N, c_hidden_conv1, H_intermediate, W_intermediate,
                     C_in, 3, 3, 
                     1, 1, 1, 1, 
                     1, 1, 
                     H_intermediate, W_intermediate, eps,
                     temp_conv2_bn_ptr,
                     temp_conv2_bn_ptr + N * C_in * H_intermediate * W_intermediate // A separate temp buffer for BN to SiLU
                     );
    __syncthreads();

    
    int total_elements = N * C_in * H_in * W_in;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < total_elements) {
        if (shortcut && (C_in == C_in)) { 
            output_ptr[idx] = temp_conv2_out_ptr[idx] + input_ptr[idx];
        } else {
            output_ptr[idx] = temp_conv2_out_ptr[idx];
        }
    }
    __syncthreads();
}

__device__ void apply_sppf_block(const float* input_ptr, float* output_ptr,
                                 const float* weights_cv1_ptr, const float* bias_cv1_ptr,
                                 const float* gamma_cv1_ptr, const float* beta_cv1_ptr, const float* mean_cv1_ptr, const float* var_cv1_ptr,
                                 const float* weights_final_conv_ptr, const float* bias_final_conv_ptr,
                                 const float* gamma_final_conv_ptr, const float* beta_final_conv_ptr, const float* mean_final_conv_ptr, const float* var_final_conv_ptr,
                                 int N, int C_in, int H_in, int W_in,
                                 int C_out, int k_maxpool, float eps,
                                 float* temp_cv1_out_ptr,
                                 float* temp_mp1_out_ptr, float* temp_mp2_out_ptr, float* temp_mp3_out_ptr,
                                 float* temp_sppf_concat_out_ptr,
                                 float* temp_conv_bn_buffer1, float* temp_conv_bn_buffer2) {

    int c_hidden = C_in / 2;
    int k_pad = k_maxpool / 2;
    apply_conv_block(input_ptr, temp_cv1_out_ptr, 
                     weights_cv1_ptr, bias_cv1_ptr,
                     gamma_cv1_ptr, beta_cv1_ptr, mean_cv1_ptr, var_cv1_ptr,
                     N, C_in, H_in, W_in,
                     c_hidden, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_in, W_in, eps,
                     temp_conv_bn_buffer1, temp_conv_bn_buffer2); 
    __syncthreads();

    int H_mp_out = (H_in + 2 * k_pad - 1 * (k_maxpool - 1) - 1) / 1 + 1;
    int W_mp_out = (W_in + 2 * k_pad - 1 * (k_maxpool - 1) - 1) / 1 + 1;

    int total_mp_elements = N * c_hidden * H_mp_out * W_mp_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    
    if (idx < total_mp_elements) {
        int w_mp_idx = idx % W_mp_out;
        int h_mp_idx = (idx / W_mp_out) % H_mp_out;
        int c_mp_idx = (idx / (W_mp_out * H_mp_out)) % c_hidden;
        int n_mp_idx = idx / (c_hidden * H_mp_out * W_mp_out);

        
        temp_mp1_out_ptr[idx] = maxpool2d_element(temp_cv1_out_ptr,
                                                  n_mp_idx, c_mp_idx, h_mp_idx, w_mp_idx,
                                                  c_hidden, H_in, W_in, 
                                                  k_maxpool, k_maxpool, 1, 1, k_pad, k_pad);
    }
    __syncthreads(); // Sync after mp1

    if (idx < total_mp_elements) {
        // MaxPool 2
        int n_mp_idx = idx / (c_hidden * H_mp_out * W_mp_out);
        int w_mp_idx = idx % W_mp_out;
        int h_mp_idx = (idx / W_mp_out) % H_mp_out;
        int c_mp_idx = (idx / (W_mp_out * H_mp_out)) % c_hidden;
        temp_mp2_out_ptr[idx] = maxpool2d_element(temp_mp1_out_ptr,
                                                  n_mp_idx, c_mp_idx, h_mp_idx, w_mp_idx,
                                                  c_hidden, H_mp_out, W_mp_out, // C_in here refers to channel dimension of temp_mp1_out_ptr
                                                  k_maxpool, k_maxpool, 1, 1, k_pad, k_pad);
    }
    __syncthreads(); // Sync after mp2

    if (idx < total_mp_elements) {
        // MaxPool 3
        int n_mp_idx = idx / (c_hidden * H_mp_out * W_mp_out);
        int w_mp_idx = idx % W_mp_out;
        int h_mp_idx = (idx / W_mp_out) % H_mp_out;
        int c_mp_idx = (idx / (W_mp_out * H_mp_out)) % c_hidden;
        temp_mp3_out_ptr[idx] = maxpool2d_element(temp_mp2_out_ptr,
                                                  n_mp_idx, c_mp_idx, h_mp_idx, w_mp_idx,
                                                  c_hidden, H_mp_out, W_mp_out, // C_in here refers to channel dimension of temp_mp2_out_ptr
                                                  k_maxpool, k_maxpool, 1, 1, k_pad, k_pad);
    }
    __syncthreads(); // Sync after mp3

    // Concatenation. Each thread writes to its specific part of the concatenated output.
    int concated_channels = c_hidden * 4;
    int total_concatenated_elements = N * concated_channels * H_in * W_in;

    if (idx < total_concatenated_elements) {
        // Decompose concatenated linear index to NCHW for concatenation
        int w_concat = idx % W_in;
        int h_concat = (idx / W_in) % H_in;
        int c_concat = (idx / (W_in * H_in)) % concated_channels;
        int n_concat = idx / (concated_channels * H_in * W_in);

        if (c_concat < c_hidden) { // First block (cv1_output)
            temp_sppf_concat_out_ptr[idx] = temp_cv1_out_ptr[n_concat * c_hidden * H_in * W_in + c_concat * H_in * W_in + h_concat * W_in + w_concat];
        } else if (c_concat < 2 * c_hidden) { // Second block (mp1_output)
            temp_sppf_concat_out_ptr[idx] = temp_mp1_out_ptr[n_concat * c_hidden * H_in * W_in + (c_concat - c_hidden) * H_in * W_in + h_concat * W_in + w_concat];
        } else if (c_concat < 3 * c_hidden) { // Third block (mp2_output)
            temp_sppf_concat_out_ptr[idx] = temp_mp2_out_ptr[n_concat * c_hidden * H_in * W_in + (c_concat - 2 * c_hidden) * H_in * W_in + h_concat * W_in + w_concat];
        } else { // Fourth block (mp3_output)
            temp_sppf_concat_out_ptr[idx] = temp_mp3_out_ptr[n_concat * c_hidden * H_in * W_in + (c_concat - 3 * c_hidden) * H_in * W_in + h_concat * W_in + w_concat];
        }
    }
    __syncthreads(); // Sync after concatenation

    // Final convolution block after concatenation
    apply_conv_block(temp_sppf_concat_out_ptr, output_ptr,
                     weights_final_conv_ptr, bias_final_conv_ptr,
                     gamma_final_conv_ptr, beta_final_conv_ptr, mean_final_conv_ptr, var_final_conv_ptr,
                     N, concated_channels, H_in, W_in,
                     C_out, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_in, W_in, eps,
                     temp_conv_bn_buffer1, temp_conv_bn_buffer2); // Re-use temp buffers
    __syncthreads();
}


// ====================================================================
// full_inference Global Kernel (use global ---> (host to global) (global->device) (device->device) <--- use only these)
// ====================================================================


__global__ void full_inference(float* input, float* output,
                               const float* d_weights, const float* d_biases,
                               const float* d_bn_gamma, const float* d_bn_beta,
                               const float* d_bn_mean, const float* d_bn_var,
                               int N, int H, int W, int C,
                               float* buffer1, float* buffer2,
                               float* d_apply_conv_block_temp_conv, float* d_apply_conv_block_temp_bn, 
                               float* d_temp_c3_cv1_output, float* d_temp_c3_cv2_output, float* d_temp_c3_m_output
                               ) {


    float* current_input = input;
    float* current_output;

    int N_batch = N; // Batch size
    int C_current = C; // Current channels
    int H_current = H; // Current height
    int W_current = W; // Current width

    int K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilate;
    int C_out; 
    float eps = 1e-5f; 

    const float* w_ptr = d_weights;
    const float* b_ptr = d_biases;
    const float* g_ptr = d_bn_gamma;
    const float* bt_ptr = d_bn_beta;
    const float* m_ptr = d_bn_mean;
    const float* v_ptr = d_bn_var;

    
    int total_elements_in_max_layer = N_batch * 1024 * H * W; 
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;


    // --- Layer 0 (Conv) ---
    K_h = 6; K_w = 6; stride_h = 2; stride_w = 2; pad_h = 2; pad_w = 2; dilate = 1;
    C_out = 64;

    int H_out_L0 = (H_current + 2 * pad_h - dilate * (K_h - 1) - 1) / stride_h + 1;
    int W_out_L0 = (W_current + 2 * pad_w - dilate * (K_w - 1) - 1) / stride_w + 1;

    current_output = buffer1; // Output of Layer 0 goes to buffer1
    apply_conv_block(current_input, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for Layer 0
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate,
                     H_out_L0, W_out_L0,
                     eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads(); // Sync after layer operation

    C_current = C_out;
    H_current = H_out_L0;
    W_current = W_out_L0;
    if (thread_idx == 0) { // Only one thread prints
        printf("Layer 0 (Conv) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output; // Input for next layer is output of current


    

    // --- Layer 1 (Conv) ---
    // Update parameter pointers by the size of parameters used in Layer 0
    // (C_out_prev * C_in_prev * K_h * K_w) for weights
    w_ptr += (size_t)C_out * (C_current) * 6 * 6; // Example size for L0 weights
    b_ptr += C_out;
    g_ptr += C_out; bt_ptr += C_out; m_ptr += C_out; v_ptr += C_out;

    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 128;

    int H_out_L1 = (H_current + 2 * pad_h - dilate * (K_h - 1) - 1) / stride_h + 1;
    int W_out_L1 = (W_current + 2 * pad_w - dilate * (K_w - 1) - 1) / stride_w + 1;

    current_output = buffer2; // Ping-pong output to buffer2
    apply_conv_block(current_input, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for Layer 1
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate,
                     H_out_L1, W_out_L1,
                     eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = C_out;
    H_current = H_out_L1;
    W_current = W_out_L1;
    if (thread_idx == 0) {
        printf("Layer 1 (Conv) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 2 (C3) ---
    // Update parameter pointers for Layer 2's internal convs.
    w_ptr += (size_t)C_out * (C_current) * 3 * 3; // Example size for L1 weights
    b_ptr += C_out;
    g_ptr += C_out; bt_ptr += C_out; m_ptr += C_out; v_ptr += C_out;

    int c1_c3 = C_current;
    int c2_c3 = 128;
    int n_bottlenecks = 3;
    bool shortcut_c3 = true;
    int g_c3 = 1;
    float e_c3 = 0.5f;
    int c_hidden_c3 = static_cast<int>(c2_c3 * e_c3); // 64

    // Use specific temporary buffers for C3's internal processing
    float* cv1_output_c3_ptr = d_temp_c3_cv1_output;
    float* cv2_output_c3_ptr = d_temp_c3_cv2_output;
    float* m_output_c3_ptr = d_temp_c3_m_output; // This buffer will be iteratively updated

    // Conv 1x1 for first branch of C3 (cv1_output_c3)
    // C_out for this conv is c_hidden_c3
    apply_conv_block(current_input, cv1_output_c3_ptr,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    // Advance pointers for parameters used by this conv (C_in * C_out * 1 * 1)
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1;
    b_ptr += c_hidden_c3;
    g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;

    // Conv 1x1 for second branch of C3 (cv2_output_c3)
    // C_out for this conv is c_hidden_c3
    apply_conv_block(current_input, cv2_output_c3_ptr,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current,
                     c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    // Advance pointers for parameters used by this conv
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1;
    b_ptr += c_hidden_c3;
    g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;

    // Initialize m_output_c3 (This is for the 'main' branch of the C3 bottleneck chain)
    int total_c_hidden_c3_elements = N_batch * c_hidden_c3 * H_current * W_current;
    if (thread_idx < total_c_hidden_c3_elements) {
        m_output_c3_ptr[thread_idx] = cv1_output_c3_ptr[thread_idx];
    }
    __syncthreads();

    // Bottleneck chain (n_bottlenecks times)
    for (int i = 0; i < n_bottlenecks; ++i) {
        // Pointers for the two internal convs of each bottleneck block
        // First conv: C_in = c_hidden_c3, C_out = c_hidden_c3
        // Second conv: C_in = c_hidden_c3, C_out = C_in (of bottleneck block, i.e., c_hidden_c3)

        // apply_bottleneck_block uses 2 conv blocks internally, so advance pointers twice per iteration
        apply_bottleneck_block(m_output_c3_ptr, m_output_c3_ptr, // Input/Output for bottleneck are the same buffer
                               w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // conv1 params
                               w_ptr + (size_t)c_hidden_c3 * c_hidden_c3 * 1 * 1, // conv2 weights
                               b_ptr + c_hidden_c3, // conv2 biases
                               g_ptr + c_hidden_c3, bt_ptr + c_hidden_c3, m_ptr + c_hidden_c3, v_ptr + c_hidden_c3, // conv2 bn params
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3, c_hidden_c3, c_hidden_c3, eps,
                               buffer1, d_apply_conv_block_temp_conv, // Use ping-pong for bottleneck internal temps
                               buffer2, d_apply_conv_block_temp_bn);
        __syncthreads();

        // Advance pointers past the two convs within this bottleneck block
        w_ptr += (size_t)c_hidden_c3 * c_hidden_c3 * (1 * 1 + 3 * 3); // C1*C_hidden*K1*K1 + C_hidden*C_in*K2*K2
        b_ptr += c_hidden_c3 * 2;
        g_ptr += c_hidden_c3 * 2; bt_ptr += c_hidden_c3 * 2; m_ptr += c_hidden_c3 * 2; v_ptr += c_hidden_c3 * 2;
    }

    // Concatenate m_output_c3 and cv2_output_c3
    int m_output_channels_c3 = c_hidden_c3;
    int cv2_output_channels_c3 = c_hidden_c3;
    int concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;

    float* concatenated_output_c3_ptr = buffer1; // Use buffer1 for the concatenated result

    int total_concatenated_elements_c3 = N_batch * concated_C_c3 * H_current * W_current;
    if (thread_idx < total_concatenated_elements_c3) {
        int w_concat = thread_idx % W_current;
        int h_concat = (thread_idx / W_current) % H_current;
        int c_concat = (thread_idx / (W_current * H_current)) % concated_C_c3;
        int n_concat = thread_idx / (concated_C_c3 * H_current * W_current);

        if (c_concat < m_output_channels_c3) {
            concatenated_output_c3_ptr[thread_idx] =
                m_output_c3_ptr[n_concat * m_output_channels_c3 * H_current * W_current + c_concat * H_current * W_current + h_concat * W_current + w_concat];
        } else {
            concatenated_output_c3_ptr[thread_idx] =
                cv2_output_c3_ptr[n_concat * cv2_output_channels_c3 * H_current * W_current + (c_concat - m_output_channels_c3) * H_current * W_current + h_concat * W_current + w_concat];
        }
    }
    __syncthreads();

    // Final Conv after concatenation (part of C3 module)
    // C_in for this conv is concated_C_c3, C_out is c2_c3
    apply_conv_block(concatenated_output_c3_ptr, current_output, // Output to current_output buffer (buffer2)
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for final C3 conv
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = c2_c3;
    // H_current and W_current remain unchanged for C3 block with 1x1 convs and 3x3 with same padding
    if (thread_idx == 0) {
        printf("Layer 2 (C3) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output; // Input for next layer is output of current


    // --- Layer 3 (Conv) ---
    w_ptr += (size_t)c2_c3 * concated_C_c3 * 1 * 1; // Advance params past C3 final conv
    b_ptr += c2_c3;
    g_ptr += c2_c3; bt_ptr += c2_c3; m_ptr += c2_c3; v_ptr += c2_c3;

    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 256;


    int H_out_L3 = (H_current + 2 * pad_h - dilate * (K_h - 1) - 1) / stride_h + 1;
    int W_out_L3 = (W_current + 2 * pad_w - dilate * (K_w - 1) - 1) / stride_w + 1;

    current_output = buffer1; // Ping-pong output to buffer1
    apply_conv_block(current_input, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for Layer 3
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate,
                     H_out_L3, W_out_L3,
                     eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = C_out;
    H_current = H_out_L3;
    W_current = W_out_L3;
    if (thread_idx == 0) {
        printf("Layer 3 (Conv) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 4 (C3) ---
    w_ptr += (size_t)C_out * C_current * 3 * 3; // Advance params past L3 conv
    b_ptr += C_out;
    g_ptr += C_out; bt_ptr += C_out; m_ptr += C_out; v_ptr += C_out;

    c1_c3 = C_current;
    c2_c3 = 256;
    n_bottlenecks = 6;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;
    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3); // 128

    // Reuse C3 internal buffers
    cv1_output_c3_ptr = d_temp_c3_cv1_output;
    cv2_output_c3_ptr = d_temp_c3_cv2_output;
    m_output_c3_ptr = d_temp_c3_m_output;

    // First branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv1_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;

    // Second branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv2_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;
    
    // Initialize m_output_c3
    if (thread_idx < total_c_hidden_c3_elements) {
        m_output_c3_ptr[thread_idx] = cv1_output_c3_ptr[thread_idx];
    }
    __syncthreads();

    // Bottleneck chain
    for (int i = 0; i < n_bottlenecks; ++i) {
        // Pass m_output_c3_ptr as both input and output for in-place update or ping-pong
        apply_bottleneck_block(m_output_c3_ptr, m_output_c3_ptr,
                               w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // conv1 params
                               w_ptr + (size_t)c_hidden_c3 * c_hidden_c3 * 1 * 1, b_ptr + c_hidden_c3, g_ptr + c_hidden_c3, bt_ptr + c_hidden_c3, m_ptr + c_hidden_c3, v_ptr + c_hidden_c3, // conv2 params
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3, c_hidden_c3, c_hidden_c3, eps,
                               buffer1, d_apply_conv_block_temp_conv, buffer2, d_apply_conv_block_temp_bn);
        __syncthreads();
        // Advance parameters past the two internal convs of this bottleneck block
        w_ptr += (size_t)c_hidden_c3 * c_hidden_c3 * (1 * 1 + 3 * 3);
        b_ptr += c_hidden_c3 * 2;
        g_ptr += c_hidden_c3 * 2; bt_ptr += c_hidden_c3 * 2; m_ptr += c_hidden_c3 * 2; v_ptr += c_hidden_c3 * 2;
    }
    
    // Concatenate m_output_c3 and cv2_output_c3
     m_output_channels_c3 = c_hidden_c3;
    cv2_output_channels_c3 = c_hidden_c3; // Corrected to use c_hidden_c3 for second branch
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
     concatenated_output_c3_ptr = buffer1; // Use buffer1 for concatenated output

    int total_concatenated_elements = N_batch * concated_C_c3 * H_current * W_current;
    if (thread_idx < total_concatenated_elements) {
        int w_concat = thread_idx % W_current;
        int h_concat = (thread_idx / W_current) % H_current;
        int c_concat = (thread_idx / (W_current * H_current)) % concated_C_c3;
        int n_concat = thread_idx / (concated_C_c3 * H_current * W_current);

        if (c_concat < m_output_channels_c3) {
            concatenated_output_c3_ptr[thread_idx] =
                m_output_c3_ptr[n_concat * m_output_channels_c3 * H_current * W_current + c_concat * H_current * W_current + h_concat * W_current + w_concat];
        } else {
            concatenated_output_c3_ptr[thread_idx] =
                cv2_output_c3_ptr[n_concat * cv2_output_channels_c3 * H_current * W_current + (c_concat - m_output_channels_c3) * H_current * W_current + h_concat * W_current + w_concat];
        }
    }
    __syncthreads();

    // Final Conv of C3 block
    current_output = buffer2; // Ping-pong output to buffer2
    apply_conv_block(concatenated_output_c3_ptr, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = c2_c3;
    if (thread_idx == 0) {
        printf("Layer 2 (C3) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 3 (Conv) ---
    w_ptr += (size_t)c2_c3 * concated_C_c3 * 1 * 1; // Advance params past C3 final conv
    b_ptr += c2_c3;
    g_ptr += c2_c3; bt_ptr += c2_c3; m_ptr += c2_c3; v_ptr += c2_c3;

    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 256;

    H_out_L3 = (H_current + 2 * pad_h - dilate * (K_h - 1) - 1) / stride_h + 1;
    W_out_L3 = (W_current + 2 * pad_w - dilate * (K_w - 1) - 1) / stride_w + 1;

    current_output = buffer1; // Ping-pong output to buffer1
    apply_conv_block(current_input, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for Layer 3
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate,
                     H_out_L3, W_out_L3,
                     eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = C_out;
    H_current = H_out_L3;
    W_current = W_out_L3;
    if (thread_idx == 0) {
        printf("Layer 3 (Conv) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 4 (C3) ---
    w_ptr += (size_t)C_out * C_current * 3 * 3; // Advance params past L3 conv
    b_ptr += C_out;
    g_ptr += C_out; bt_ptr += C_out; m_ptr += C_out; v_ptr += C_out;

    c1_c3 = C_current;
    c2_c3 = 256;
    n_bottlenecks = 6;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;
    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3); // 128

    // Reuse C3 internal buffers
    cv1_output_c3_ptr = d_temp_c3_cv1_output;
    cv2_output_c3_ptr = d_temp_c3_cv2_output;
    m_output_c3_ptr = d_temp_c3_m_output;

    // First branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv1_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;

    // Second branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv2_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;
    
    // Initialize m_output_c3
    if (thread_idx < total_c_hidden_c3_elements) {
        m_output_c3_ptr[thread_idx] = cv1_output_c3_ptr[thread_idx];
    }
    __syncthreads();

    // Bottleneck chain
    for (int i = 0; i < n_bottlenecks; ++i) {
        apply_bottleneck_block(m_output_c3_ptr, m_output_c3_ptr,
                               w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // conv1 params
                               w_ptr + (size_t)c_hidden_c3 * c_hidden_c3 * 1 * 1, b_ptr + c_hidden_c3, g_ptr + c_hidden_c3, bt_ptr + c_hidden_c3, m_ptr + c_hidden_c3, v_ptr + c_hidden_c3, // conv2 params
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3, c_hidden_c3, c_hidden_c3, eps,
                               buffer1, d_apply_conv_block_temp_conv, buffer2, d_apply_conv_block_temp_bn);
        __syncthreads();
        w_ptr += (size_t)c_hidden_c3 * c_hidden_c3 * (1 * 1 + 3 * 3);
        b_ptr += c_hidden_c3 * 2;
        g_ptr += c_hidden_c3 * 2; bt_ptr += c_hidden_c3 * 2; m_ptr += c_hidden_c3 * 2; v_ptr += c_hidden_c3 * 2;
    }
    
    // Concatenate m_output_c3 and cv2_output_c3
    m_output_channels_c3 = c_hidden_c3;
    cv2_output_channels_c3 = c_hidden_c3;
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3_ptr = buffer1; // Use buffer1 for concatenated output

    int total_concatenated_elements_L4 = N_batch * concated_C_c3 * H_current * W_current;
    if (thread_idx < total_concatenated_elements_L4) {
        int w_concat = thread_idx % W_current;
        int h_concat = (thread_idx / W_current) % H_current;
        int c_concat = (thread_idx / (W_current * H_current)) % concated_C_c3;
        int n_concat = thread_idx / (concated_C_c3 * H_current * W_current);

        if (c_concat < m_output_channels_c3) {
            concatenated_output_c3_ptr[thread_idx] =
                m_output_c3_ptr[n_concat * m_output_channels_c3 * H_current * W_current + c_concat * H_current * W_current + h_concat * W_current + w_concat];
        } else {
            concatenated_output_c3_ptr[thread_idx] =
                cv2_output_c3_ptr[n_concat * cv2_output_channels_c3 * H_current * W_current + (c_concat - m_output_channels_c3) * H_current * W_current + h_concat * W_current + w_concat];
        }
    }
    __syncthreads();

    // Final Conv of C3 block (Layer 4)
    current_output = buffer2;
    apply_conv_block(concatenated_output_c3_ptr, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = c2_c3;
    if (thread_idx == 0) {
        printf("Layer 4 (C3) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 5 (Conv) ---
    w_ptr += (size_t)c2_c3 * concated_C_c3 * 1 * 1; // Advance params past L4 C3 final conv
    b_ptr += c2_c3;
    g_ptr += c2_c3; bt_ptr += c2_c3; m_ptr += c2_c3; v_ptr += c2_c3;

    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 512;

    int H_out_L5 = (H_current + 2 * pad_h - dilate * (K_h - 1) - 1) / stride_h + 1;
    int W_out_L5 = (W_current + 2 * pad_w - dilate * (K_w - 1) - 1) / stride_w + 1;

    current_output = buffer1;
    apply_conv_block(current_input, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for Layer 5
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate,
                     H_out_L5, W_out_L5,
                     eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = C_out;
    H_current = H_out_L5;
    W_current = W_out_L5;
    if (thread_idx == 0) {
        printf("Layer 5 (Conv) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 6 (C3) ---
    w_ptr += (size_t)C_out * C_current * 3 * 3; // Advance params past L5 conv
    b_ptr += C_out;
    g_ptr += C_out; bt_ptr += C_out; m_ptr += C_out; v_ptr += C_out;

    c1_c3 = C_current;
    c2_c3 = 512;
    n_bottlenecks = 9;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;
    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3); // 256

    // Reuse C3 internal buffers
    cv1_output_c3_ptr = d_temp_c3_cv1_output;
    cv2_output_c3_ptr = d_temp_c3_cv2_output;
    m_output_c3_ptr = d_temp_c3_m_output;

    // First branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv1_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;

    // Second branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv2_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;
    
    // Initialize m_output_c3
    if (thread_idx < total_c_hidden_c3_elements) {
        m_output_c3_ptr[thread_idx] = cv1_output_c3_ptr[thread_idx];
    }
    __syncthreads();

    // Bottleneck chain
    for (int i = 0; i < n_bottlenecks; ++i) {
        apply_bottleneck_block(m_output_c3_ptr, m_output_c3_ptr,
                               w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // conv1 params
                               w_ptr + (size_t)c_hidden_c3 * c_hidden_c3 * 1 * 1, b_ptr + c_hidden_c3, g_ptr + c_hidden_c3, bt_ptr + c_hidden_c3, m_ptr + c_hidden_c3, v_ptr + c_hidden_c3, // conv2 params
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3, c_hidden_c3, c_hidden_c3, eps,
                               buffer1, d_apply_conv_block_temp_conv, buffer2, d_apply_conv_block_temp_bn);
        __syncthreads();
        w_ptr += (size_t)c_hidden_c3 * c_hidden_c3 * (1 * 1 + 3 * 3);
        b_ptr += c_hidden_c3 * 2;
        g_ptr += c_hidden_c3 * 2; bt_ptr += c_hidden_c3 * 2; m_ptr += c_hidden_c3 * 2; v_ptr += c_hidden_c3 * 2;
    }
    
    // Concatenate m_output_c3 and cv2_output_c3
    m_output_channels_c3 = c_hidden_c3;
    cv2_output_channels_c3 = c_hidden_c3;
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3_ptr = buffer1; // Use buffer1 for concatenated output

    int total_concatenated_elements_L6 = N_batch * concated_C_c3 * H_current * W_current;
    if (thread_idx < total_concatenated_elements_L6) {
        int w_concat = thread_idx % W_current;
        int h_concat = (thread_idx / W_current) % H_current;
        int c_concat = (thread_idx / (W_current * H_current)) % concated_C_c3;
        int n_concat = thread_idx / (concated_C_c3 * H_current * W_current);

        if (c_concat < m_output_channels_c3) {
            concatenated_output_c3_ptr[thread_idx] =
                m_output_c3_ptr[n_concat * m_output_channels_c3 * H_current * W_current + c_concat * H_current * W_current + h_concat * W_current + w_concat];
        } else {
            concatenated_output_c3_ptr[thread_idx] =
                cv2_output_c3_ptr[n_concat * cv2_output_channels_c3 * H_current * W_current + (c_concat - m_output_channels_c3) * H_current * W_current + h_concat * W_current + w_concat];
        }
    }
    __syncthreads();

    // Final Conv of C3 block (Layer 6)
    current_output = buffer2;
    apply_conv_block(concatenated_output_c3_ptr, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = c2_c3;
    if (thread_idx == 0) {
        printf("Layer 6 (C3) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 7 (Conv) ---
    w_ptr += (size_t)c2_c3 * concated_C_c3 * 1 * 1; // Advance params past L6 C3 final conv
    b_ptr += c2_c3;
    g_ptr += c2_c3; bt_ptr += c2_c3; m_ptr += c2_c3; v_ptr += c2_c3;

    K_h = 3; K_w = 3; stride_h = 2; stride_w = 2; pad_h = 1; pad_w = 1; dilate = 1;
    C_out = 1024;

    int H_out_L7 = (H_current + 2 * pad_h - dilate * (K_h - 1) - 1) / stride_h + 1;
    int W_out_L7 = (W_current + 2 * pad_w - dilate * (K_w - 1) - 1) / stride_w + 1;

    current_output = buffer1;
    apply_conv_block(current_input, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // Params for Layer 7
                     N_batch, C_current, H_current, W_current,
                     C_out, K_h, K_w,
                     stride_h, stride_w, pad_h, pad_w,
                     dilate, dilate,
                     H_out_L7, W_out_L7,
                     eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = C_out;
    H_current = H_out_L7;
    W_current = W_out_L7;
    if (thread_idx == 0) {
        printf("Layer 7 (Conv) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 8 (C3) ---
    w_ptr += (size_t)C_out * C_current * 3 * 3; // Advance params past L7 conv
    b_ptr += C_out;
    g_ptr += C_out; bt_ptr += C_out; m_ptr += C_out; v_ptr += C_out;

    c1_c3 = C_current;
    c2_c3 = 1024;
    n_bottlenecks = 3;
    shortcut_c3 = true;
    g_c3 = 1;
    e_c3 = 0.5f;
    c_hidden_c3 = static_cast<int>(c2_c3 * e_c3); // 512

    // Reuse C3 internal buffers
    cv1_output_c3_ptr = d_temp_c3_cv1_output;
    cv2_output_c3_ptr = d_temp_c3_cv2_output;
    m_output_c3_ptr = d_temp_c3_m_output;

    // First branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv1_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;

    // Second branch of C3 (conv 1x1)
    apply_conv_block(current_input, cv2_output_c3_ptr, w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, c1_c3, H_current, W_current, c_hidden_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();
    w_ptr += (size_t)c_hidden_c3 * c1_c3 * 1 * 1; b_ptr += c_hidden_c3; g_ptr += c_hidden_c3; bt_ptr += c_hidden_c3; m_ptr += c_hidden_c3; v_ptr += c_hidden_c3;
    
    // Initialize m_output_c3
    if (thread_idx < total_c_hidden_c3_elements) {
        m_output_c3_ptr[thread_idx] = cv1_output_c3_ptr[thread_idx];
    }
    __syncthreads();

    // Bottleneck chain
    for (int i = 0; i < n_bottlenecks; ++i) {
        apply_bottleneck_block(m_output_c3_ptr, m_output_c3_ptr,
                               w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr, // conv1 params
                               w_ptr + (size_t)c_hidden_c3 * c_hidden_c3 * 1 * 1, b_ptr + c_hidden_c3, g_ptr + c_hidden_c3, bt_ptr + c_hidden_c3, m_ptr + c_hidden_c3, v_ptr + c_hidden_c3, // conv2 params
                               N_batch, c_hidden_c3, H_current, W_current,
                               shortcut_c3, g_c3, c_hidden_c3, c_hidden_c3, eps,
                               buffer1, d_apply_conv_block_temp_conv, buffer2, d_apply_conv_block_temp_bn);
        __syncthreads();
        w_ptr += (size_t)c_hidden_c3 * c_hidden_c3 * (1 * 1 + 3 * 3);
        b_ptr += c_hidden_c3 * 2;
        g_ptr += c_hidden_c3 * 2; bt_ptr += c_hidden_c3 * 2; m_ptr += c_hidden_c3 * 2; v_ptr += c_hidden_c3 * 2;
    }
    
    // Concatenate m_output_c3 and cv2_output_c3
    m_output_channels_c3 = c_hidden_c3;
    cv2_output_channels_c3 = c_hidden_c3;
    concated_C_c3 = m_output_channels_c3 + cv2_output_channels_c3;
    
    concatenated_output_c3_ptr = buffer1; // Use buffer1 for concatenated output

    int total_concatenated_elements_L8 = N_batch * concated_C_c3 * H_current * W_current;
    if (thread_idx < total_concatenated_elements_L8) {
        int w_concat = thread_idx % W_current;
        int h_concat = (thread_idx / W_current) % H_current;
        int c_concat = (thread_idx / (W_current * H_current)) % concated_C_c3;
        int n_concat = thread_idx / (concated_C_c3 * H_current * W_current);

        if (c_concat < m_output_channels_c3) {
            concatenated_output_c3_ptr[thread_idx] =
                m_output_c3_ptr[n_concat * m_output_channels_c3 * H_current * W_current + c_concat * H_current * W_current + h_concat * W_current + w_concat];
        } else {
            concatenated_output_c3_ptr[thread_idx] =
                cv2_output_c3_ptr[n_concat * cv2_output_channels_c3 * H_current * W_current + (c_concat - m_output_channels_c3) * H_current * W_current + h_concat * W_current + w_concat];
        }
    }
    __syncthreads();

    // Final Conv of C3 block (Layer 8)
    current_output = buffer2;
    apply_conv_block(concatenated_output_c3_ptr, current_output,
                     w_ptr, b_ptr, g_ptr, bt_ptr, m_ptr, v_ptr,
                     N_batch, concated_C_c3, H_current, W_current,
                     c2_c3, 1, 1, 1, 1, 0, 0, 1, 1,
                     H_current, W_current, eps, d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn);
    __syncthreads();

    C_current = c2_c3;
    if (thread_idx == 0) {
        printf("Layer 8 (C3) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
    }
    current_input = current_output;


    // --- Layer 9: SPPF [1024, 5] ---
    w_ptr += (size_t)c2_c3 * concated_C_c3 * 1 * 1; // Advance params past L8 C3 final conv
    b_ptr += c2_c3;
    g_ptr += c2_c3; bt_ptr += c2_c3; m_ptr += c2_c3; v_ptr += c2_c3;

    int C_in9 = C_current; // Should be 1024 from Layer 8
    int C_out9 = 1024;
    int k_sppf = 5;

    // Temporary buffers for SPPF block (cv1_output, mp1, mp2, mp3, concat_sppf)
    // SPPF outputs to the final `output` pointer passed to `full_inference`
    float* temp_cv1_out_sppf_ptr = d_temp_c3_cv1_output; // Reuse C3 internal buffers for SPPF internal temps
    float* temp_mp1_out_sppf_ptr = d_temp_c3_cv2_output;
    float* temp_mp2_out_sppf_ptr = d_temp_c3_m_output; // Third temp buffer for MP chain
    float* temp_mp3_out_sppf_ptr = buffer1; // Fourth temp buffer for MP chain
    float* temp_sppf_concat_out_ptr = output; // This will be the final output pointer

    // Parameters for SPPF's internal layers. Assume they follow in parameter buffer.
    const float* sppf_cv1_w = w_ptr; const float* sppf_cv1_b = b_ptr;
    const float* sppf_cv1_g = g_ptr; const float* sppf_cv1_bt = bt_ptr;
    const float* sppf_cv1_m = m_ptr; const float* sppf_cv1_v = v_ptr;
    
    // Advance pointers past SPPF's internal conv1 parameters
    w_ptr += (size_t)(C_in9/2) * C_in9 * 1 * 1;
    b_ptr += C_in9/2;
    g_ptr += C_in9/2; bt_ptr += C_in9/2; m_ptr += C_in9/2; v_ptr += C_in9/2;

    const float* sppf_final_conv_w = w_ptr; const float* sppf_final_conv_b = b_ptr;
    const float* sppf_final_conv_g = g_ptr; const float* sppf_final_conv_bt = bt_ptr;
    const float* sppf_final_conv_m = m_ptr; const float* sppf_final_conv_v = v_ptr;

    apply_sppf_block(current_input, temp_sppf_concat_out_ptr, // output directly to final output
                     sppf_cv1_w, sppf_cv1_b, sppf_cv1_g, sppf_cv1_bt, sppf_cv1_m, sppf_cv1_v, // cv1 params
                     sppf_final_conv_w, sppf_final_conv_b, sppf_final_conv_g, sppf_final_conv_bt, sppf_final_conv_m, sppf_final_conv_v, // Final conv params
                     N_batch, C_in9, H_current, W_current,
                     C_out9, k_sppf, eps,
                     temp_cv1_out_sppf_ptr,
                     temp_mp1_out_sppf_ptr, temp_mp2_out_sppf_ptr, temp_mp3_out_sppf_ptr,
                     temp_sppf_concat_out_ptr, // This is the output_ptr for SPPF, which is the final output
                     d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn); // Passed down temp buffers

    C_current = C_out9;
    // H_current and W_current remain unchanged due to stride 1 and padding k//2 in MaxPool
    if (thread_idx == 0) {
        printf("Layer 9 (SPPF) output: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);
        printf("\nAll backbone layers have been processed from full_inference kernel.\n");
        printf("Final output of backbone: C=%d, H=%d, W=%d\n", C_current, H_current, W_current);

        printf("Sample of final output ****(after Layer 9 SPPF block) ***** from device full_inference:\n");
        // Only print a few elements to avoid excessive device-side printf, which is slow.
        // This loop will run for *every* thread..
        int total_final_output_elements = N_batch * C_current * H_current * W_current;
        for (int i = 0; i < 10; ++i) {
            if (isnan(output[i])) {
                printf("NaN found at index %d\n", i);
            }
            printf("%f ", output[i]);
        }
        printf("\n");
    }

    printf("Head processing started .... \n");








}


// ====================================================================
// Host-side Main Function
// ====================================================================

int main() {
    int N = 1, H = 64, W = 64, C = 3; 
    int max_intermediate_channels = 1024;
    int max_intermediate_H = H; 
    int max_intermediate_W = W;

    size_t buffer_size_elements = (size_t)N * max_intermediate_channels * max_intermediate_H * max_intermediate_W;
    size_t buffer_size_bytes = buffer_size_elements * sizeof(float);
    size_t apply_conv_block_temp_size_elements = buffer_size_elements;
    size_t apply_conv_block_temp_size_bytes = apply_conv_block_temp_size_elements * sizeof(float);

    size_t c3_temp_size_elements = buffer_size_elements;
    size_t c3_temp_size_bytes = c3_temp_size_elements * sizeof(float);


    size_t initial_input_size_elements = (size_t)N * C * H * W;


    size_t final_output_H = calculate_conv_output_dims(H, W, 6, 6, 2, 2, 2, 2, 1, 1).first; // After L0
    final_output_H = calculate_conv_output_dims(final_output_H, final_output_H, 3, 3, 2, 2, 1, 1, 1, 1).first; // After L1
    final_output_H = calculate_conv_output_dims(final_output_H, final_output_H, 1, 1, 1, 1, 0, 0, 1, 1).first; // After L2, L4, L6, L8 C3 blocks (dimensions don't change spatially for 1x1 convs)
    final_output_H = calculate_conv_output_dims(final_output_H, final_output_H, 3, 3, 2, 2, 1, 1, 1, 1).first; // After L3
    final_output_H = calculate_conv_output_dims(final_output_H, final_output_H, 3, 3, 2, 2, 1, 1, 1, 1).first; // After L5
    final_output_H = calculate_conv_output_dims(final_output_H, final_output_H, 3, 3, 2, 2, 1, 1, 1, 1).first; // After L7

    size_t final_output_W = final_output_H; 
    size_t final_output_channels = 1024;

    size_t final_output_size_elements = (size_t)N * final_output_channels * final_output_H * final_output_W;

    float *h_input;
    float *h_output;
    h_input = (float*)malloc(initial_input_size_elements * sizeof(float));
    h_output = (float*)malloc(final_output_size_elements * sizeof(float));

    for (size_t i = 0; i < initial_input_size_elements; ++i) h_input[i] = 1.0f;


    size_t total_weights_elements = 0;
    size_t total_bias_elements = 0;
    size_t total_bn_param_elements = 0; 

    // Layer 0 (Conv: C_in=3, C_out=64, K=6)
    total_weights_elements += (size_t)64 * 3 * 6 * 6;
    total_bias_elements += 64;
    total_bn_param_elements += 64; // For gamma, beta, mean, var (4*C_out)

    // Layer 1 (Conv: C_in=64, C_out=128, K=3)
    total_weights_elements += (size_t)128 * 64 * 3 * 3;
    total_bias_elements += 128;
    total_bn_param_elements += 128;

    // Layer 2 (C3: C_in=128, C_out=128, n=3, e=0.5 -> c_hidden=64)
    // C3 has 2 initial 1x1 convs + n_bottlenecks * (2 convs each) + 1 final 1x1 conv
    // Conv1_C3 (1x1): C_in=128, C_out=64
    total_weights_elements += (size_t)64 * 128 * 1 * 1;
    total_bias_elements += 64;
    total_bn_param_elements += 64;
    // Conv2_C3 (1x1): C_in=128, C_out=64
    total_weights_elements += (size_t)64 * 128 * 1 * 1;
    total_bias_elements += 64;
    total_bn_param_elements += 64;

    // Bottlenecks (n_bottlenecks * 2 convs per bottleneck)
    for (int i = 0; i < 3; ++i) { // n_bottlenecks = 3
        total_weights_elements += (size_t)64 * 64 * 1 * 1;
        // Bottleneck internal Conv1 (1x1): C_in=64, C_out=64
        total_bias_elements += 64;
        total_bn_param_elements += 64;
        // Bottleneck internal Conv2 (3x3): C_in=64, C_out=64
        total_weights_elements += (size_t)64 * 64 * 3 * 3;
        total_bias_elements += 64;
        total_bn_param_elements += 64;
    }
    // Final Conv_C3 (1x1): C_in=(64+64), C_out=128
    total_weights_elements += (size_t)128 * (64 + 64) * 1 * 1;
    total_bias_elements += 128;
    total_bn_param_elements += 128;

    // Layer 3 (Conv: C_in=128, C_out=256, K=3)
    total_weights_elements += (size_t)256 * 128 * 3 * 3;
    total_bias_elements += 256;
    total_bn_param_elements += 256;

    // Layer 4 (C3: C_in=256, C_out=256, n=6, e=0.5 -> c_hidden=128)
    // Conv1_C3 (1x1): C_in=256, C_out=128
    total_weights_elements += (size_t)128 * 256 * 1 * 1;
    total_bias_elements += 128;
    total_bn_param_elements += 128;
    // Conv2_C3 (1x1): C_in=256, C_out=128
    total_weights_elements += (size_t)128 * 256 * 1 * 1;
    total_bias_elements += 128;
    total_bn_param_elements += 128; //bottleneck params

    for (int i = 0; i < 6; ++i) { // n_bottlenecks = 6
        // Bottleneck internal Conv1 (1x1): C_in=128, C_out=128
        total_weights_elements += (size_t)128 * 128 * 1 * 1;
        total_bias_elements += 128;
        total_bn_param_elements += 128;
        // Bottleneck internal Conv2 (3x3): C_in=128, C_out=128
        total_weights_elements += (size_t)128 * 128 * 3 * 3;
        total_bias_elements += 128;
        total_bn_param_elements += 128;
    }
    // Final Conv_C3 (1x1): C_in=(128+128), C_out=256
    total_weights_elements += (size_t)256 * (128 + 128) * 1 * 1;
    total_bias_elements += 256;
    total_bn_param_elements += 256;

    // Layer 5 (Conv: C_in=256, C_out=512, K=3)
    total_weights_elements += (size_t)512 * 256 * 3 * 3;
    total_bias_elements += 512;
    total_bn_param_elements += 512;

    // Layer 6 (C3: C_in=512, C_out=512, n=9, e=0.5 -> c_hidden=256)
    // Conv1_C3 (1x1): C_in=512, C_out=256
    total_weights_elements += (size_t)256 * 512 * 1 * 1;
    total_bias_elements += 256;
    total_bn_param_elements += 256;
    // Conv2_C3 (1x1): C_in=512, C_out=256
    total_weights_elements += (size_t)256 * 512 * 1 * 1;
    total_bias_elements += 256;
    total_bn_param_elements += 256;

    for (int i = 0; i < 9; ++i) { // n_bottlenecks = 9
        // Bottleneck internal Conv1 (1x1): C_in=256, C_out=256
        total_weights_elements += (size_t)256 * 256 * 1 * 1;
        total_bias_elements += 256;
        total_bn_param_elements += 256;
        // Bottleneck internal Conv2 (3x3): C_in=256, C_out=256
        total_weights_elements += (size_t)256 * 256 * 3 * 3;
        total_bias_elements += 256;
        total_bn_param_elements += 256;
    }
    // Final Conv_C3 (1x1): C_in=(256+256), C_out=512
    total_weights_elements += (size_t)512 * (256 + 256) * 1 * 1;
    total_bias_elements += 512;
    total_bn_param_elements += 512;


    // Layer 7 (Conv: C_in=512, C_out=1024, K=3)
    total_weights_elements += (size_t)1024 * 512 * 3 * 3;
    total_bias_elements += 1024;
    total_bn_param_elements += 1024;

    // Layer 8 (C3: C_in=1024, C_out=1024, n=3, e=0.5 -> c_hidden=512)
    // Conv1_C3 (1x1): C_in=1024, C_out=512
    total_weights_elements += (size_t)512 * 1024 * 1 * 1;
    total_bias_elements += 512;
    total_bn_param_elements += 512;
    // Conv2_C3 (1x1): C_in=1024, C_out=512
    total_weights_elements += (size_t)512 * 1024 * 1 * 1;
    total_bias_elements += 512;
    total_bn_param_elements += 512;

    for (int i = 0; i < 3; ++i) { // n_bottlenecks = 3
        // Bottleneck internal Conv1 (1x1): C_in=512, C_out=512
        total_weights_elements += (size_t)512 * 512 * 1 * 1;
        total_bias_elements += 512;
        total_bn_param_elements += 512;
        // Bottleneck internal Conv2 (3x3): C_in=512, C_out=512
        total_weights_elements += (size_t)512 * 512 * 3 * 3;
        total_bias_elements += 512;
        total_bn_param_elements += 512;
    }
    // Final Conv_C3 (1x1): C_in=(512+512), C_out=1024
    total_weights_elements += (size_t)1024 * (512 + 512) * 1 * 1;
    total_bias_elements += 1024;
    total_bn_param_elements += 1024;

    // Layer 9 (SPPF: C_in=1024, C_out=1024, k=5)
    // SPPF internal Conv1 (1x1): C_in=1024, C_out=512
    total_weights_elements += (size_t)512 * 1024 * 1 * 1;
    total_bias_elements += 512;
    total_bn_param_elements += 512;
    // SPPF final Conv (1x1): C_in=(512*4), C_out=1024
    total_weights_elements += (size_t)1024 * (512 * 4) * 1 * 1;
    total_bias_elements += 1024;
    total_bn_param_elements += 1024;


    size_t total_weights_size_bytes = total_weights_elements * sizeof(float);
    size_t total_bias_size_bytes = total_bias_elements * sizeof(float);
    size_t total_bn_param_size_bytes = total_bn_param_elements * sizeof(float);

    float *d_input, *d_output;
    float *d_buffer1, *d_buffer2; 
    float *d_apply_conv_block_temp_conv, *d_apply_conv_block_temp_bn; 
    float *d_temp_c3_cv1_output, *d_temp_c3_cv2_output, *d_temp_c3_m_output; 

    float *d_weights_all, *d_biases_all;
    float *d_bn_gamma_all, *d_bn_beta_all, *d_bn_mean_all, *d_bn_var_all;

    cudaMalloc((void**)&d_input, initial_input_size_elements * sizeof(float));
    cudaMalloc((void**)&d_output, final_output_size_elements * sizeof(float));
    cudaMalloc((void**)&d_buffer1, buffer_size_bytes);
    cudaMalloc((void**)&d_buffer2, buffer_size_bytes);
    cudaMalloc((void**)&d_apply_conv_block_temp_conv, apply_conv_block_temp_size_bytes);
    cudaMalloc((void**)&d_apply_conv_block_temp_bn, apply_conv_block_temp_size_bytes);
    cudaMalloc((void**)&d_temp_c3_cv1_output, c3_temp_size_bytes);
    cudaMalloc((void**)&d_temp_c3_cv2_output, c3_temp_size_bytes);
    cudaMalloc((void**)&d_temp_c3_m_output, c3_temp_size_bytes);

    cudaMalloc((void**)&d_weights_all, total_weights_size_bytes);
    cudaMalloc((void**)&d_biases_all, total_bias_size_bytes);
    cudaMalloc((void**)&d_bn_gamma_all, total_bn_param_size_bytes);
    cudaMalloc((void**)&d_bn_beta_all, total_bn_param_size_bytes);
    cudaMalloc((void**)&d_bn_mean_all, total_bn_param_size_bytes);
    cudaMalloc((void**)&d_bn_var_all, total_bn_param_size_bytes);

    float* h_dummy_weights = (float*)malloc(total_weights_size_bytes);
    float* h_dummy_biases = (float*)malloc(total_bias_size_bytes);
    float* h_dummy_bn_params = (float*)malloc(total_bn_param_size_bytes); // For gamma, beta, mean, var combined

    // random_device rd;
    // mt19937 gen(rd()); 
    // uniform_real_distribution<float> dist(-1.0f, 1.0f); 
    for(size_t i = 0; i < total_weights_elements; ++i){
       float weight = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_dummy_weights[i]=weight;
    }
    for(size_t i = 0; i < total_bias_elements; ++i){
        float weight = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_dummy_biases[i] = weight;
    }
    for(size_t i = 0; i < total_bn_param_elements; ++i){
        float weight = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_dummy_bn_params[i] = weight;
    } 
    cudaMemcpy(d_input, h_input, initial_input_size_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_all, h_dummy_weights, total_weights_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases_all, h_dummy_biases, total_bias_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_gamma_all, h_dummy_bn_params, total_bn_param_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_beta_all, h_dummy_bn_params, total_bn_param_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_mean_all, h_dummy_bn_params, total_bn_param_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_var_all, h_dummy_bn_params, total_bn_param_size_bytes, cudaMemcpyHostToDevice);

    printf("Launching full_inference kernel...\n");

    int threads_per_block = 256; 
    int num_blocks = (buffer_size_elements + threads_per_block - 1) / threads_per_block;

    full_inference<<<num_blocks, threads_per_block>>>(
        d_input, d_output,
        d_weights_all, d_biases_all,
        d_bn_gamma_all, d_bn_beta_all, d_bn_mean_all, d_bn_var_all,
        N, H, W, C,
        d_buffer1, d_buffer2,
        d_apply_conv_block_temp_conv, d_apply_conv_block_temp_bn,
        d_temp_c3_cv1_output, d_temp_c3_cv2_output, d_temp_c3_m_output
    );
    cudaDeviceSynchronize();
    printf("Kernel execution complete. Copying results back...\n");

    cudaMemcpy(h_output, d_output, final_output_size_elements * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nOutput sample (first 10 elements of final output):\n");
    for (int i = 0; i < std::min((int)final_output_size_elements, 10); ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_buffer1);
    cudaFree(d_buffer2);
    cudaFree(d_apply_conv_block_temp_conv);
    cudaFree(d_apply_conv_block_temp_bn);
    cudaFree(d_temp_c3_cv1_output);
    cudaFree(d_temp_c3_cv2_output);
    cudaFree(d_temp_c3_m_output);

    cudaFree(d_weights_all);
    cudaFree(d_biases_all);
    cudaFree(d_bn_gamma_all);
    cudaFree(d_bn_beta_all);
    cudaFree(d_bn_mean_all);
    cudaFree(d_bn_var_all);

    free(h_input);
    free(h_output);
    free(h_dummy_weights);
    free(h_dummy_biases);
    free(h_dummy_bn_params);

    printf("Memory freed. Exiting.\n");
    return 0;
}
