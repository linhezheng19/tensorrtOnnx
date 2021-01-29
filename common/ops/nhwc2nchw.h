/**
 * Convert input image from nhwc mode to nchw mode.
 * 2020/11/20
 */

#ifndef NWHC2NCHW_H
#define NWHC2NCHW_H

#include <cuda.h>
#include <array>

#include "utils.h"

#define BLOCK 512

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
        const ImageFormat format);

extern "C" void NHWC2NCHW(
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
        const ImageFormat format);

#endif  // NWHC2NCHW_H
