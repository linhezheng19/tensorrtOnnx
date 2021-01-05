#ifndef SEMSEG_OUTPUTS_H
#define SEMSEG_OUTPUTS_H

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

#include "NvInfer.h"

using namespace std;

void postProcess(const vector<float*> outputs, vector<cv::Mat>& preds, const vector<nvinfer1::Dims>& dims);

#endif  // SEMSEG_OUTPUTS_H
