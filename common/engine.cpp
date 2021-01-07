/**
 * Basic TensorRT engine API based on ONNX parser.
 * 2020/09/01
 */
#include "engine.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"
#include "utils.h"

using namespace nvinfer1;

RTEngine::~RTEngine() {
    if (mPluginFactory != nullptr) {
        delete mPluginFactory;
        mPluginFactory = nullptr;
    }
    if (mContext != nullptr) {
        mContext->destroy();
        mContext = nullptr;
    }
    if (mEngine != nullptr) {
        mEngine->destroy();
        mEngine = nullptr;
    }
    for (size_t i = 0;i < mBinding.size(); ++i) {
        safeCudaFree(mBinding[i]);
    }
}

void RTEngine::CreateEngine(const std::string& onnxModel,
                            const std::string& engineFile,
                            const std::vector<std::string>& customOutput,
                            int maxBatchSize,
                            RunMode runMode,
                            long workspace_size) {
    if (!DeserializeEngine(engineFile)) {
        if (!BuildEngine(onnxModel,engineFile,customOutput,maxBatchSize, runMode, workspace_size)) {
            mInfoLogger.logger("ERROR: could not deserialize or build engine");
            return;
        }
    }
    mInfoLogger.logger("Create execute context and malloc device memory...");
    InitEngine();
}

void RTEngine::Forward() {
    mContext->execute(mBatchSize, &mBinding[0]);
}

void RTEngine::ForwardAsync(const cudaStream_t& stream) {
    mContext->enqueue(mBatchSize, &mBinding[0], stream, nullptr);
}

void RTEngine::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
}

void RTEngine::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex, const cudaStream_t& stream) {
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, stream));
}

void RTEngine::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
}

void RTEngine::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex, const cudaStream_t& stream) {
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, stream));
}

void RTEngine::SetDevice(int device) {
    mInfoLogger.logger("Make sure save engine file match choosed device.", logger::LEVEL::WARNING);
    CUDA_CHECK(cudaSetDevice(device));
}

int RTEngine::GetDevice() {
    int* device = nullptr;  // NOTE: memory leaks here
    CUDA_CHECK(cudaGetDevice(device));
    if (device != nullptr) {
        return device[0];
    }
    else {
        mInfoLogger.logger("Get Device Error", logger::LEVEL::ERROR);
        return -1;
    }
}

int RTEngine::GetMaxBatchSize() const {
    return mBatchSize;
}

void* RTEngine::GetBindingPtr(int bindIndex) const {
    return mBinding[bindIndex];
}

size_t RTEngine::GetBindingSize(int bindIndex) const {
    return mBindingSize[bindIndex];
}

nvinfer1::Dims RTEngine::GetBindingDims(int bindIndex) const {
    return mBindingDims[bindIndex];
}

nvinfer1::DataType RTEngine::GetBindingDataType(int bindIndex) const {
    return mBindingDataType[bindIndex];
}

void RTEngine::SaveEngine(const std::string& fileName) {
    if (fileName == "") {
        mInfoLogger.logger("Empty engine file name, skip save");
        return;
    }
    if (mEngine != nullptr) {
        mInfoLogger.logger("Save engine to: ", fileName);
        nvinfer1::IHostMemory* data = mEngine->serialize();
        std::ofstream file;
        file.open(fileName,std::ios::binary | std::ios::out);
        if(!file.is_open()) {
            mInfoLogger.logger("Read create engine file failed: ",fileName);
            return;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        data->destroy();
    } else {
        mInfoLogger.logger("Engine is empty, save engine failed");
    }
}

bool RTEngine::DeserializeEngine(const std::string& engineFile) {
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    if (in.is_open()) {
        mInfoLogger.logger("Deserialize engine from:", engineFile);
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);
        initLibNvInferPlugins(&mLogger, "");
        mRuntime = nvinfer1::createInferRuntime(mLogger);
        mEngine = mRuntime->deserializeCudaEngine((void*)engineBuf.get(), bufCount, nullptr);
        assert(mEngine != nullptr);
        mBatchSize = mEngine->getMaxBatchSize();
        mInfoLogger.logger("Max batch size of deserialized engine:",mEngine->getMaxBatchSize());
        mRuntime->destroy();
        return true;
    }
    return false;
}

bool RTEngine::BuildEngine(const std::string& onnxModel,
                           const std::string& engineFile,
                           const std::vector<std::string>& customOutput,
                           int maxBatchSize,
                           RunMode runMode,
                           long workspace_size) {
    mInfoLogger.logger("The ONNX Parser shipped with TensorRT 5.1.x+ supports ONNX IR (Intermediate Representation) version 0.0.3, opset version 9");
    mBatchSize = maxBatchSize;
    mInfoLogger.logger("Build onnx engine from: ", onnxModel);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr && "Builder is NULL!");
    auto explicitBtach = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBtach);
    assert(network != nullptr && "Network is NULL");
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, mLogger);
    if (!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        mInfoLogger.logger("Could not parse onnx engine", logger::LEVEL::ERROR);
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    builder->setMaxBatchSize(mBatchSize);
    config->setMaxWorkspaceSize(workspace_size << 20);

    switch(runMode) {
        case(RunMode::kFP16): {
            if (!builder->platformHasFastFp16()) {
                mInfoLogger.logger("Device do not support FP16 mode, switch to fp32 mode.", logger::LEVEL::WARNING);
                break;
            }
            else {
                mInfoLogger.logger("Set engine to fp16 mode ");
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
                break;
            }
        }
        case (RunMode::kINT8): {
            if (!builder->platformHasFastInt8()) {
                mInfoLogger.logger("Device do not support INT8 mode, switch to fp32 mode.", logger::LEVEL::WARNING);
                break;
            } else {
                mInfoLogger.logger("Set engine to int8 mode ");
                config->setFlag(nvinfer1::BuilderFlag::kINT8);
                setAllTensorScales(network, 127.0f, 127.0f);
    //            builder->setStrictTypeConstraints(true);
                break;
            }
        }
    }

    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);
    mInfoLogger.logger("Serialize engine to: ", engineFile);
    SaveEngine(engineFile);

    builder->destroy();
    network->destroy();
    parser->destroy();
    return true;
}

void RTEngine::InitEngine() {
    mInfoLogger.logger("Init engine...");
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    mInfoLogger.logger("Malloc device memory...");
    int nbBindings = mEngine->getNbBindings();
    mInfoLogger.logger("nbBingdings: ", nbBindings);
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for (int i=0; i< nbBindings; i++) {
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);
        int64_t totalSize;
        if (mFromOnnx) {
            totalSize = volume(dims) * getElementSize(dtype);
        } else {
            totalSize = volume(dims) * mBatchSize * getElementSize(dtype);
        }
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;
        if (mEngine->bindingIsInput(i)) {
            mInfoLogger.logger("Input: ");
        } else {
            mInfoLogger.logger("Output: ");
        }
        std::cout << "Binding bindIndex: " << i << ", Name: " << name << ", Size in bytes: " << totalSize << std::endl;
        std::cout << "Binding dims with " << dims.nbDims << " dimemsions" << std::endl;
        for (int j=0;j<dims.nbDims;j++) {
            std::cout << dims.d[j] << " x ";
        }
        std::cout << "\b\b  "<< std::endl;
        mBinding[i] = safeCudaMalloc(totalSize);
        if (mEngine->bindingIsInput(i)) {
            mInputSize++;
        }
    }
}
