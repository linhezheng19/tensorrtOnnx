#ifndef NMS_H
#define NMS_H

#include <cuda.h>

#define BLOCK 512
//#define BLOCK 1024

__device__ __inline__ float logist(float x);

// template <
//     int n,
//     int h,
//     int w,
//     int reid_num,
//     int kernel_h,
//     int kernel_w,
//     float vis_thresh>
// __global__ void nms_kernel(
//     const float* hm,
//     const float* reg,
//     const float* wh,
//     const float* id_feat,
//     float* output,
//     int* count)
__global__ void nms_kernel(
        const float* hm,
        const float* reg,
        const float* wh,
        const float* id_feat,
        float* output,
        int* count,
        const int n,
        const int h,
        const int w,
        const int reid_num,
        const int kernel_h,
        const int kernel_w,
        const float vis_thresh);

extern "C" void det_nms(
        const float* hm,
        const float* reg,
        const float* wh,
        const float* id_feat,
        float* output_nms,
        int* count,
        int& resCount,
        const int n,
        const int h,
        const int w,
        const int reid_num,
        const int kernel_h,
        const int kernel_w,
        const float score_th);

#endif  // NMS_H
