/**
 * Basic API.
 * 2020/09/01
 */

#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "NvInfer.h"
#include "logger.h"
#include "utils.h"

class RTEngineLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};


class RTEngine {
public:
    /**
     * Default constructor.
     */
    RTEngine() = default;
    ~RTEngine();

    /**
     * Create engine from onnx model.
     * onnxModel: path to onnx model
     * engineFile: path to saved engien file will be load or save, if it's empty them will not
     *              save engine file
     * maxBatchSize: max batch size for inference.
     * runMode: mode while running, fp32/fp16/int8
     * return:
     */
    void CreateEngine(const std::string& onnxModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& customOutput,
                      int maxBatchSize,
                      RunMode runMode,
                      long workspace_size);

    /**
     * Do inference on engine context, make sure you already copy your data to device memory,
     * using CopyFromHostToDevice etc.
     */
    void Forward();

    /**
     * Async inference on engine context.
     * stream: cuda stream for async inference and data copy
     */
    void ForwardAsync(const cudaStream_t& stream);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex);

    void CopyFromHostToDevice(const std::vector<float>& input, int bindIndex,const cudaStream_t& stream);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,const cudaStream_t& stream);

    void SetDevice(int device);

    int GetDevice();

    /**
     * Get max batch size of build engine.
     * return: max batch size of build engine.
     */
    int GetMaxBatchSize() const;

    /**
     * Get binding data pointer in device. For example if you want to do some post processing
     * on inference output but want to process them in gpu directly for efficiency, you can
     * use this function to avoid extra data io.
     * return: pointer point to device memory.
     */
    void* GetBindingPtr(int bindIndex) const;

    /**
     * Get binding data size in byte, so maybe you need to divide it by sizeof(T) where T is data type
     * like float.
     * return: size in byte.
     */
    size_t GetBindingSize(int bindIndex) const;

    /**
     * Get binding dimemsions.
     * return: binding dimemsions, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_dims.html
     */
    nvinfer1::Dims GetBindingDims(int bindIndex) const;

    /**
     * Get binding data type.
     * return: binding data type, see https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/namespacenvinfer1.html#afec8200293dc7ed40aca48a763592217
     */
    nvinfer1::DataType GetBindingDataType(int bindIndex) const;

    std::vector<std::string> mBindingName;

private:
    bool DeserializeEngine(const std::string& engineFile);

    bool BuildEngine(const std::string& onnxModel,
                     const std::string& engineFile,
                     const std::vector<std::string>& customOutput,
                     int maxBatchSize,
                     RunMode runMode,
                     long workspace_size);

    /**
     * Init resource such as device memory
     */
    void InitEngine();

    /**
     * Save engine to engine file
     */
    void SaveEngine(const std::string& fileName);

private:
    RTEngineLogger mLogger;
    logger::Logger mInfoLogger;

    // tensorrt run mode 0:fp32 1:fp16 2:int8
    int mRunMode;

    nvinfer1::ICudaEngine* mEngine = nullptr;

    nvinfer1::IExecutionContext* mContext = nullptr;

    nvinfer1::IPluginFactory* mPluginFactory = nullptr;

    nvinfer1::IRuntime* mRuntime = nullptr;

    std::vector<void*> mBinding;

    std::vector<size_t> mBindingSize;

    std::vector<nvinfer1::Dims> mBindingDims;

    std::vector<nvinfer1::DataType> mBindingDataType;

    int mInputSize = 0;
    int mBatchSize;
    bool mFromOnnx = true;
};

#endif  // ENGINE_H
