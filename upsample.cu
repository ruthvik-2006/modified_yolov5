#ifndef UPSAMPLE_CU
#define UPSAMPLE_CU

__device__ void upsample_nearest(const float* input, float* output,
                                 int N, int C, int H_in, int W_in,
                                 int H_out, int W_out) {
    int flat_output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long output_plane_size = (long long)H_out * W_out;
    long long output_channel_stride = (long long)C * output_plane_size;

    int n = flat_output_idx / output_channel_stride;
    int remaining = flat_output_idx % output_channel_stride;
    int c = remaining / output_plane_size;
    remaining = remaining % output_plane_size;
    int h_out = remaining / W_out;
    int w_out = remaining % W_out;

    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;

    int h_in = h_out * H_in / H_out;
    int w_in = w_out * W_in / W_out;

    h_in = min(h_in, H_in - 1);
    w_in = min(w_in, W_in - 1);

    long long input_idx = (long long)n * C * H_in * W_in +
                          (long long)c * H_in * W_in +
                          (long long)h_in * W_in + w_in;

    long long output_idx = (long long)n * C * H_out * W_out +
                           (long long)c * H_out * W_out +
                           (long long)h_out * W_out + w_out;

    output[output_idx] = input[input_idx];
}

#endif