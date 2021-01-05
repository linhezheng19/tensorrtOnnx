#ifndef F_TRACK_OUTPUTS_H
#define F_TRACK_OUTPUTS_H

#include <algorithm>
#include <array>
#include <dirent.h>
#include <iostream>
#include <math.h>
#include <string>
#include <thread>
#include <vector>

#include <NvInfer.h>

#include "struct.h"

std::pair<std::vector<std::vector<std::array<float, 5>>>, std::vector<std::vector<std::vector<float>>>> f_track_postProcess(std::vector <float*> inputs, std::vector<size_t> sizes, std::vector<nvinfer1::Dims> dims, int mModel_H, int mModel_W, int NumClass, float postThres, float area_thresh, float  ratio, float nmsThres);

#endif  // F_TRACK_OUTPUTS_H
