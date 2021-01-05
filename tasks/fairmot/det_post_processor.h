#ifndef DETPOSTPROCESSOR_H
#define DETPOSTPROCESSOR_H

#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <utility>  // for std::pair
#include <cmath>  // for std::log
#include <string.h>  // for memcpy

#define ARRAY1D(T, N) std::array<T, N>
#define ARRAY2D(T, ROW, COL) std::array<std::array<T, COL>, ROW>

class DetPostProcessor
{
private:
    int* resCount;
    float* res;
    int* nms_count;
    float* nms_output;
    float* topk_output;
    const int batch;
    const int height;
    const int width;
    const int reid_dim;
    const int topk;
    const int kernel_h;
    const int kernel_w;
    const float score_th;
    const bool order;
    const bool batched;
public:
    DetPostProcessor() = delete;
    DetPostProcessor(
            int batch,
            int height,
            int width,
            int reid_dim,
            int topk = 32,
            int kernel_h = 3,
            int kernel_w = 3,
            float score_th = 0.6f,
            bool order = true,
            bool batched = true);
    ~DetPostProcessor();
    void process(
            const float* hm,
            const float* reg,
            const float* wh,
            const float* reid);
    std::pair<std::vector<ARRAY1D(float, 5)>,
            std::vector<std::vector<float>>> getDets_batched();
    std::pair<std::vector<std::vector<ARRAY1D(float, 5)>>,
            std::vector<std::vector<std::vector<float>>>> getDets();
private:
    void toCpu();
};

extern "C" void det_nms(
        const float* hm,
        const float* reg,
        const float* wh,
        const float* id_feat,
        float* output_nms,
        int* count,
        int* resCount,
        const int n,
        const int h,
        const int w,
        const int reid_num,
        const int kernel_h,
        const int kernel_w,
        const float score_th,
        const bool batched);

extern "C" void det_topk(
        float* input,
        float* output,
        const int resCount,
        const int topk,
        const int extra_dim,
        const bool order);


#endif //DETPOSTPROCESSOR_H
