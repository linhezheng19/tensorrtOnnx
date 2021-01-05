#include "det_post_processor.h"
#include "assert.h"
#include <iostream>

DetPostProcessor::DetPostProcessor(
        int batch,
        int height,
        int width,
        int reid_dim,
        int topk,
        int kernel_h,
        int kernel_w,
        float score_thresh,
        bool order,
        bool batched) :
        batch(batch),
        height(height),
        width(width),
        reid_dim(reid_dim),
        topk(topk),
        kernel_h(kernel_h),
        kernel_w(kernel_w),
        score_th(std::log(score_thresh / (1.f - score_thresh))),
        order(order),
        batched(batched),
        resCount(new int[batch]),
        res(new float[batch * topk * (5 + reid_dim)]) {
    std::cout
            << "DetPostProcessor: CUDA malloc memory "
            << batch * (sizeof(int) + (topk + height * width) * (5 + reid_dim) * sizeof(float))
               / 1024.f / 1024.f
            << " MB for"
            << " batch: " << batch
            << " width: " << width
            << " height: " << height
            << " topk: " << topk
            << " reid_dim: " << reid_dim
            << std::endl;

    cudaMalloc((void**)&nms_count, batch * sizeof(int));
    cudaMalloc((void**)&nms_output, batch * height * width * (5 + reid_dim) * sizeof(float));
    cudaMalloc((void**)&topk_output, batch * topk * (5 + reid_dim) * sizeof(float));
}
DetPostProcessor::~DetPostProcessor() {
    delete []resCount;
    delete []res;

    std::cout
            << "DetPostProcessor: CUDA free memory "
            << batch * (sizeof(int) + (topk + height * width) * (5 + reid_dim) * sizeof(float))
               / 1024.f / 1024.f
            << "MB\n";
    cudaFree(nms_count);
    cudaFree(nms_output);
    cudaFree(topk_output);
}

void DetPostProcessor::process(
        const float* hm,
        const float* reg,
        const float* wh,
        const float* reid) {
    det_nms(
            hm,
            reg,
            wh,
            reid,
            nms_output,
            nms_count,
            resCount,
            batch,
            height,
            width,
            reid_dim,
            kernel_h,
            kernel_w,
            score_th,
            batched);

    if (batched) {
        if (resCount[0] > batch * topk) {
            det_topk(nms_output,
                     topk_output,
                     resCount[0],
                     batch * topk,
                     reid_dim + 4,
                     order);
        }
    } else {
        for (int i = 0; i < batch; ++i) {
            if (resCount[i] > topk) {
                det_topk(nms_output + i * height * width * (5 + reid_dim),
                         topk_output + i * height * width * (5 + reid_dim),
                         resCount[i],
                         topk,
                         reid_dim + 4,
                         order);
            }
        }
    }
}

void DetPostProcessor::toCpu() {
    int data_dim = 5 + reid_dim;

    if (batched) {
        if (resCount[0] <= batch * topk) {
            cudaMemcpy(
                    res,
                    nms_output,
                    resCount[0] * data_dim * sizeof(float),
                    cudaMemcpyDeviceToHost);
        } else {
            resCount[0] = batch * topk;
            cudaMemcpy(
                    res,
                    topk_output,
                    resCount[0] * data_dim * sizeof(float),
                    cudaMemcpyDeviceToHost);
        }
    } else {
        int offset = height * width * data_dim;
        for (int i = 0; i < batch; ++i) {
            if (resCount[i] <= topk) {
                cudaMemcpy(
                        res + i * offset,
                        nms_output + i * offset,
                        resCount[i] * data_dim * sizeof(float),
                        cudaMemcpyDeviceToHost);
            } else {
                resCount[i] = topk;
                cudaMemcpy(
                        res + i * offset,
                        topk_output + i * offset,
                        resCount[i] * data_dim * sizeof(float),
                        cudaMemcpyDeviceToHost);
            }
        }
    }
}


std::pair<std::vector<ARRAY1D(float, 5)>, std::vector<std::vector<float>>>
DetPostProcessor::getDets_batched() {
    assert(batched);
    toCpu();
    int data_dim = 5 + reid_dim;
    std::vector<ARRAY1D(float, 5)> dets;
    std::vector<std::vector<float>> id_features;
    float* det;
    for (int i = (resCount[0] - 1) * data_dim; i >= 0; i -= data_dim) {
        det = res + i;
        dets.push_back( { det[1], det[2], det[3], det[4], det[0] } );
        std::vector<float> id_feat(reid_dim);
        memcpy(&id_feat[0], &det[5], reid_dim * sizeof(float));
        id_features.emplace_back(id_feat);
    }
    return std::make_pair(dets, id_features);
}


std::pair<std::vector<std::vector<ARRAY1D(float, 5)>>, std::vector<std::vector<std::vector<float>>>>
DetPostProcessor::getDets() {
    assert(!batched);
    toCpu();
    int data_dim = 5 + reid_dim;
    int offset = height * width * data_dim;
    std::vector<std::vector<ARRAY1D(float, 5)>> dets(batch);
    std::vector<std::vector<std::vector<float>>> id_features(batch);
    float* det;
    for (int n = 0; n < batch; ++n) {
        dets[n].reserve(topk);
        id_features[n].reserve(topk);
        for (int i = (resCount[n] - 1) * data_dim; i >= 0; i -= data_dim) {
            det = res + n * offset + i;
            dets[n].push_back( { det[1], det[2], det[3], det[4], det[0] } );
            std::vector<float> id_feat(reid_dim);
            memcpy(&id_feat[0], &det[5], reid_dim * sizeof(float));
            id_features[n].emplace_back(id_feat);
        }
    }
    return std::make_pair(dets, id_features);
}
