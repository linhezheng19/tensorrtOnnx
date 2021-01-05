#include <array>

#include "fcos.h"
#include "fcos_outputs.h"

FCOS::FCOS(const YAML::Node& cfg) : DetectionTask(cfg) {
    mNumClasses = cfg["params"]["num_classes"].as<int>();
}

bool FCOS::prepareInputs(const vector<Mat>& imgs) {
    return DetectionTask::prepareInputs(imgs);
}

BatchBox FCOS::processOutputs() {
    vector<float*> inputs;
    vector<size_t> sizes;
    vector<nvinfer1::Dims> dims;
    vector<int> idx_list = cfg["params"]["output_index"].as<vector<int>>();
    for (int i = 0; i < 9; ++i) {  // get size  TODO
        //fcos  onnx outputs from 1-9
        inputs.push_back((float*)mNet->GetBindingPtr(idx_list[i]));
        sizes.push_back((size_t)mNet->GetBindingSize(idx_list[i]));
        dims.push_back(mNet->GetBindingDims(idx_list[i]));
    }
    float det_thresh = cfg["params"]["det_thresh"].as<float>();
    float nms_thresh = cfg["params"]["nms_thresh"].as<float>();
    BatchBox results = postProcess(inputs, sizes, dims, mModel_H, mModel_W,  mNumClasses, det_thresh, nms_thresh);
    return results;
}

BatchBox FCOS::run(const vector<Mat>& imgs) {
    mTimer->dataStart();
    if (!prepareInputs(imgs)) {
        mLogger.logger("Prepare Input Data Failed!", logger::LEVEL::ERROR);
    }
    mTimer->dataEnd();

    mTimer->inferStart();
    mNet->ForwardAsync(mStream);
    mTimer->inferEnd();

    mTimer->postStart();
    auto results = processOutputs();
    mTimer->postEnd();

    if (mTimer->showTime()) {
        mLogger.logger("FCOS Data time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("FCOS Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("FCOS Post time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

	return results;
}
