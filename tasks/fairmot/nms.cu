#include "nms.h"

__device__ __inline__ float logist(float x) {
    return 1.f / (1.f + exp(-x));
}

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
        const float score_th) {
    int stride = h * w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n * stride)
        return;

    float objPred = hm[idx];

    if (score_th >= objPred)
        return;

    int pw = idx % w;
    int ph = (idx / w) % h;
    int pn = idx / stride;

    int hstart = ph - (kernel_h - 1) / 2;
    int wstart = pw - (kernel_w - 1) / 2;
    int hend = min(hstart + kernel_h, h);
    int wend = min(wstart + kernel_w, w);
    if (hstart < 0) hstart = 0;
    if (wstart < 0) wstart = 0;

    /* compute local max: */
    const float* ip = hm + pn * stride;
    const float* ip_row;
    for (int y = hstart; y < hend; ++y) {
        ip_row = ip + y * w;
        for (int x = wstart; x < wend; ++x) {
            if (ip_row[x] > objPred)
                return;
        }
    }

    /* set output to local max */
    int resCount = atomicAdd(count, 1);
    float* data = output + resCount * (5 + reid_num);

    const float* rp = reg + idx + pn * stride;
    const float* sp = wh + idx + pn * stride;

    float cx = (pw + rp[0]) * 4.f;
    float cy = (ph + rp[stride]) * 4.f;
    float w_half = sp[0] * 2.f;
    float h_half = sp[stride] * 2.f;

    data[0] = logist(objPred);

    float im_w = w * 4.f, im_h = h * 4.f;
    float shift_w = im_w * pn;

    data[1] = min(max(cx - w_half, 0.f), im_w - 1.f) + shift_w;
    data[2] = min(max(cy - h_half, 0.f), im_h - 1.f);
    data[3] = min(max(cx + w_half, 0.f), im_w - 1.f) + shift_w;
    data[4] = min(max(cy + h_half, 0.f), im_h - 1.f);

    for (int i = 0; i < reid_num; ++i)
        data[i + 5] = id_feat[i * stride];
}

void det_nms(
        const float* hm,
        const float* reg,
        const float* wh,
        const float* id_feat,
        float* nms_output,
        int* count,
        int& resCount,
        const int n,
        const int h,
        const int w,
        const int reid_dim,
        const int kernel_h,
        const int kernel_w,
        const float score_th) {
    cudaMemset(count, 0x00, sizeof(int));

    nms_kernel<<<(n * h * w - 1) / BLOCK + 1, BLOCK>>>(
            hm,
            reg,
            wh,
            id_feat,
            nms_output,
            count,
            n,
            h,
            w,
            reid_dim,
            kernel_h,
            kernel_w,
            score_th);

    cudaMemcpy(&resCount, count, sizeof(int), cudaMemcpyDeviceToHost);
}
