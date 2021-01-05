#ifndef FCOSOUTPUTS_H
#define FCOSOUTPUTS_H

#include <algorithm>
#include <array>
#include <dirent.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <NvInfer.h>

#include "structs.h"

BatchBox postProcess(std::vector <float*> inputs, std::vector<size_t> sizes, std::vector<nvinfer1::Dims> dims, int mModel_H, int mModel_W, int NumClass, float postThres, float nmsThres);

#endif  // FCOSOUTPUTS_H
