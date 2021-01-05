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
    vector<uint8_t> semseg_outputs;
    vector<int> idx_list = cfg["params"]["output_index"].as<vector<int>>();
    // process outputs in semseg_outputs.cu
    vector<float*> outputs;
    vector<nvinfer1::Dims> dims;
    for (auto idx : idx_list) {
        nvinfer1::Dims dim = mNet->GetBindingDims(idx);
        auto output = static_cast<float*>(mNet->GetBindingPtr(idx));
        dims.emplace_back(dim);
        outputs.emplace_back(output);
    }
    postProcess(outputs, semseg_results, dims);
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