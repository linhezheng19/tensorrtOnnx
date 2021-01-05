#include "nhwc2nchw.h"

__global__ void transpose_kernel(
        const uint8_t* input,
        float* output,
        const int n,
        const int h,
        const int w,
        const float mean_0,
        const float mean_1,
        const float mean_2,
        const float var_0,
        const float var_1,
        const float var_2,
        const bool bgr_mode) {
    int stride = h * w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int pn = idx / stride;
    if (pn >= n)
        return;

    const uint8_t* ip = input + idx * 3;
    float* op = output + idx + pn * 2 * stride;

    if (bgr_mode) {
        op[0] = (float)(ip[0] - mean_0) / var_0;
        op[stride] = (float)(ip[1] - mean_1) / var_1;
        op[2 * stride] = (float)(ip[2] - mean_2) / var_2;
    } else {
        op[0] = (float)(ip[2] - mean_0) / var_0;
        op[stride] = (float)(ip[1] - mean_1) / var_1;
        op[2 * stride] = (float)(ip[0] - mean_2) / var_2;
    }

}

void NHWC2NCHW(
        const uint8_t* input,
        float* output,
        const int n,
        const int h,
        const int w,
        const float mean_0,
        const float mean_1,
        const float mean_2,
        const float var_0,
        const float var_1,
        const float var_2,
        const bool bgr_mode) {
    transpose_kernel<<<(n * h * w - 1) / BLOCK + 1, BLOCK>>>(input, output, n, h, w,
                                                             mean_0, mean_1, mean_2,
                                                             var_0, var_1, var_2, bgr_mode);
}
