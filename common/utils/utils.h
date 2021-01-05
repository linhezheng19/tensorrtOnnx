/**
 * Created by linhezheng
 * Basic tensor rt engine API utils.
 * 2020/09/01
 */

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <algorithm>
#include <numeric>
#include <NvInfer.h>

#include "cuda_runtime.h"

#define UNUSED(unusedVariable) (void)(unusedVariable)
// suppress compiler warning: unused parameter

enum class RunMode : int {
    kFP32,
    kFP16,
    kINT8
};

inline int64_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        default: throw std::runtime_error("Invalid DataType.");
    }
}

inline void setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f) {
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++) {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet()) {
                input->setDynamicRange(-inScales, inScales);
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet()) {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING) {
                    output->setDynamicRange(-inScales, inScales);
                } else {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}


#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(0);                                                                         \
        }                                                                                      \
    }
#endif

inline bool stringEndsWith(const std::string str1, const std::string str2) {
    bool status = str1.find(str2) == 0 ? true : false;
    return status;
}

inline bool stringStartsWith(const std::string str1, const std::string str2) {
    bool status = str1.rfind(str2) == (str1.length() - str2.length()) ? true : false;
    return status;
}

inline void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline void safeCudaFree(void* deviceMem) {
    CUDA_CHECK(cudaFree(deviceMem));
}

inline void error(const std::string& message, const int line, const std::string& function, const std::string& file) {
    std::cout << message << " at " << line << " in " << function << " in " << file << std::endl;
}
#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) COMPILE_TEMPLATE_BASIC_TYPES(className, class)
#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) COMPILE_TEMPLATE_BASIC_TYPES(className, struct)
#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
    template classType  className<char>; \
    template classType  className<signed char>; \
    template classType  className<short>; \
    template classType  className<int>; \
    template classType  className<long>; \
    template classType  className<long long>; \
    template classType  className<unsigned char>; \
    template classType  className<unsigned short>; \
    template classType  className<unsigned int>; \
    template classType  className<unsigned long>; \
    template classType  className<unsigned long long>; \
    template classType  className<float>; \
    template classType  className<double>; \
    template classType  className<long double>

// const auto CUDA_NUM_THREADS = 512u;
// inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired,
//     const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
// {
// return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
// }

#endif  // UTILS_H
