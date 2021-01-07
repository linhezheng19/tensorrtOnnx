#include "semseg.h"
#include "semseg_outputs.h"

SEMSEG::SEMSEG(const YAML::Node& cfg) : SegmentationTask(cfg) {
    mNumClasses = cfg["params"]["num_classes"].as<int>();
}

bool SEMSEG::prepareInputs(const vector<Mat>& imgs) {
    return SegmentationTask::prepareInputs(imgs);
}

vector<Mat> SEMSEG::processOutputs() {
    vector<Mat> semseg_results(mBatchSize);
    // process outputs in semseg_outputs.cu, for result tensor
    // vector<int> idx_list = cfg["params"]["output_index"].as<vector<int>>();
    // vector<float*> outputs;
    // vector<nvinfer1::Dims> dims;
    // for (auto idx : idx_list) {
    //     nvinfer1::Dims dim = mNet->GetBindingDims(idx);
    //     auto output = static_cast<float*>(mNet->GetBindingPtr(idx));
    //     dims.emplace_back(dim);
    //     outputs.emplace_back(output);
    // }
    // postProcess(outputs, semseg_results, dims);
    // copy to mat directly, for tensor is after argmax function.
    int output_idx = 1;
    int stride = mModel_W * mModel_H;
    vector<float> semseg_outputs(mBatchSize * mNumClasses * stride);
    mNet->CopyFromDeviceToHost(semseg_outputs, output_idx, mStream);
    for (int b = 0; b < mBatchSize; ++b) {
        auto output = vector<float>(semseg_outputs.begin() + stride * b, semseg_outputs.begin() + stride * (b + 1));
        Mat temp = Mat(output);
        Mat semseg = temp.reshape(1, mModel_H).clone();
        semseg_results.emplace_back(semseg);
    }
    return semseg_results;
}

vector<Mat> SEMSEG::run(const vector<Mat>& imgs) {
    mTimer->dataStart();
    if (!prepareInputs(imgs)) {
        mLogger.logger("Prepare Input Data Failed!", logger::LEVEL::ERROR);
        assert(false);
    }
    mTimer->dataEnd();

    mTimer->inferStart();
    mNet->ForwardAsync(mStream);
    mTimer->inferEnd();

    mTimer->postStart();
    auto results = processOutputs();
    mTimer->postEnd();

    if (mTimer->showTime()) {
        mLogger.logger("Semseg Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Semseg Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Semseg Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}