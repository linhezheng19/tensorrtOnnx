/**
 * Created by linhezheng.
 * Timer based on api in "watch.h".
 * 2020/11/01
 */

#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <vector>

#include "watch.h"
#include "utils.h"

using namespace std;

class Timer {

public:
    Timer() {
        sdkCreateTimer(&watch);
        sdkResetTimer(&watch);
    };
    ~Timer() {
        if (watch){
            sdkDeleteTimer(&watch);
            delete watch;
            watch = NULL;
        }
    };

    void start() {
        CUDA_CHECK(cudaDeviceSynchronize());
        sdkCreateTimer(&watch);
        sdkResetTimer(&watch);
        sdkStartTimer(&watch);
    };
    void click() {
        CUDA_CHECK(cudaDeviceSynchronize());
        sdkStopTimer(&watch);
        float t = sdkGetAverageTimerValue(&watch);
        total_time += t;
        count++;
    };

    float getTime(bool avg=true) {
        if (!avg) {
            return sdkGetTimerValue(&watch);
        }

        float t = total_time / static_cast<float>(count);

        if (count > 100) {
            count = 0;
            total_time = 0.f;
        }

        return t;
    };


private:
    StopWatchInterface* watch = NULL;
    vector<float> times;
    int   count      = 0;
    float total_time = 0.f;
};
#endif
