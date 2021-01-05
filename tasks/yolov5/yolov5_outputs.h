#ifndef YOLOV5_OUTPUTS_H
#define YOLOV5_OUTPUTS_H

#include <algorithm>
#include <array>
#include <dirent.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <NvInfer.h>

#include "struct.h"
#include "yolov5.h"

using namespace std;

BatchBox postProcess(vector<float*> inputs, vector<size_t> sizes, vector<nvinfer1::Dims> dims, YOLOParams yolo_params, const vector<cv::Mat>& imgs);

#endif  // YOLOV5_OUTPUTS_H
