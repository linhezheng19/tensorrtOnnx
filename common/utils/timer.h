/**
 * Create by linhezheng.
 * Timer.
 * 2020/12/04
 */

#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <vector>

#include <cuda.h>
#include "utils.h"

using namespace std;

class Timer {

public:
    Timer(const cudaStream_t& stream, bool show_time) : mStream(stream), mShowTime(show_time) {
        CUDA_CHECK(cudaEventCreate(&mEvents));
        CUDA_CHECK(cudaEventCreate(&mEvente));
        
    };
    ~Timer() {
        CUDA_CHECK(cudaEventDestroy(mEvents));
        CUDA_CHECK(cudaEventDestroy(mEvente));
    };

    void inferStart() {
        if (!mShowTime) return;
        CUDA_CHECK(cudaEventRecord(mEvents, mStream));
    };

    void inferEnd() {
        if (!mShowTime) return;
        CUDA_CHECK(cudaEventRecord(mEvente, mStream));
        CUDA_CHECK(cudaEventSynchronize(mEvente));
        float once_cost;
        CUDA_CHECK(cudaEventElapsedTime(&once_cost, mEvents, mEvente));
        mInferTime += once_cost;
        mInferCount++;
    };

    void dataStart() {
        if (!mShowTime) return;
        CUDA_CHECK(cudaEventRecord(mEvents, mStream));
    };

    void dataEnd() {
        if (!mShowTime) return;
        CUDA_CHECK(cudaEventRecord(mEvente, mStream));
        CUDA_CHECK(cudaEventSynchronize(mEvente));
        float once_cost;
        CUDA_CHECK(cudaEventElapsedTime(&once_cost, mEvents, mEvente));
        mDataTime += once_cost;
        mDataCount++;
    };

    void postStart() {
        if (!mShowTime) return;
        CUDA_CHECK(cudaEventRecord(mEvents, mStream));
    };

    void postEnd() {
        if (!mShowTime) return;
        CUDA_CHECK(cudaEventRecord(mEvente, mStream));
        CUDA_CHECK(cudaEventSynchronize(mEvente));
        float once_cost;
        CUDA_CHECK(cudaEventElapsedTime(&once_cost, mEvents, mEvente));
        mPostTime += once_cost;
        mPostCount++;
    };

    float getDataTime(bool avg=false) {
        if (!mShowTime) return 0.f;
        float t = mDataTime / static_cast<float>(mDataCount);
        if(avg) return t;
        if(mDataCount > 100){
            mDataTime  = 0.f;
            mDataCount = 0;
        }
        return t;
    };

    float getInferTime(bool avg=false) {
        if (!mShowTime) return 0.f;
        float t = mInferTime / static_cast<float>(mInferCount);
        if(avg) return t;
        if(mInferCount > 100){
            mInferTime  = 0.f;
            mInferCount = 0;
        }
        return t;
    };

    float getPostTime(bool avg=false) {
        if (!mShowTime) return 0.f;
        float t = mPostTime / static_cast<float>(mPostCount);
        if(avg) return t;
        if(mPostCount > 100){
            mPostTime  = 0.f;
            mPostCount = 0;
        }
        return t;
    };

    bool showTime() {
        return mShowTime;
    };

private:
    bool mShowTime = false;
    cudaEvent_t mEvents, mEvente;
    cudaStream_t mStream;
    float mInferTime  = 0.f, mDataTime   = 0.f, mPostTime  = 0.f;
    int   mWarmupIter = 0,   mInferCount = 0,   mDataCount = 0, mPostCount = 0;
};

#endif  // TIMER_H
