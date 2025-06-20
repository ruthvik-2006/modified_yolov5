//requires <float.h> librarry fr accessing the FLOAT_MAX AND MIN VALUES!
#ifndef MAXPOOL2D_CU
#define MAXPOOL2D_CU
#include<float.h>
__device__ void maxpool2d(const float* input,float* output,
int N,int C,int H_in,int W_in,int H_out,int W_out,int kH,int kW,
int pH,int pW,int dH,int dW,int sH,int sW){
    int w_out=blockIdx.x*blockDim.x+threadIdx.x; //col
    int h_out=blockIdx.y*blockDim.y+threadIdx.y;//row

    //for 4d n,c,hout,wout...flatten this 4d to 1d...
    int flat_output_idx=blockIdx.x*blockDim.x+threadIdx.x;

    long long output_plane_size=(long long)H_out*W_out;
    long long output_batch_size=(long long)C*output_plane_size;

    int n=flat_output_idx/output_batch_size;
    int remaining=flat_output_idx%output_batch_size;
    int c=remaining/output_plane_size;
    remaining=remaining%output_plane_size;

    h_out=remaining/W_out; //now in 4d row and col
    w_out=remaining%W_out;

    //0 indexing...
    if(n>=N || c>=C || h_out>=H_out || w_out>=W_out)return;

    float max_val=-FLT_MAX;

    int h_start=h_out*sH-pH; //for unpadded
    int w_start=w_out*sW-pW;

    for(int kh=0;kh<kH;kh++){
        for(int kw=0;kw<kW;kw++){
            int h_in_current=h_start+kh*dH;
            int w_in_current=w_start+kw*dW;
            if(h_in_current>=0 && h_in_current<H_in &&
            w_in_current>=0 && w_in_current<W_in){
                long long input_idx = (long long)n * C * H_in * W_in +
                                      (long long)c * H_in * W_in +
                                      (long long)h_in_current * W_in + w_in_current;

                float current_val = input[input_idx];

                // Update max_val
                if (current_val > max_val) {
                    max_val = current_val;
                }
            }
        }
    }


    //revertt --> // n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out
    long long output_idx = (long long)n * C * H_out * W_out +
                           (long long)c * H_out * W_out +
                           (long long)h_out * W_out + w_out;

    output[output_idx] = max_val;

}

#endif