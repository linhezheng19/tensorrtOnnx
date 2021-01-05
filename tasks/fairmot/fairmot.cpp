#include <string>

#include "fairmot.h"

FairMOT::FairMOT(const YAML::Node& cfg) : TrackTask(cfg) {
    mDetPostProcessor = new DetPostProcessor(mBatchSize, mModel_H / 4, mModel_W / 4, 512, 32, 3, 3, 0.6, true);
}

FairMOT::~FairMOT() {
    delete mDetPostProcessor;
    mDetPostProcessor = nullptr;
}

bool FairMOT::prepareInputs(const vector<Mat>& imgs) {
    return TrackTask::prepareInputs(imgs);
}

TrackRes FairMOT::processOutputs() {
    DetPostProcessor& det_post_processer = *mDetPostProcessor;
    vector<int> idx_list = cfg["params"]["output_index"].as<vector<int>>();
    float* feat_gpu = (float*)mNet->GetBindingPtr(idx_list[0]);
    float* wh_gpu = (float*)mNet->GetBindingPtr(idx_list[1]);;
    float* reg_gpu = (float*)mNet->GetBindingPtr(idx_list[2]);;
    float* reid_gpu = (float*)mNet->GetBindingPtr(idx_list[3]);;

    det_post_processer.process(feat_gpu, reg_gpu, wh_gpu, reid_gpu);
    auto res = det_post_processer.getDets();

    return res;
}

TrackRes FairMOT::run(const vector<Mat>& imgs) {
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
        mLogger.logger("FairMOT Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("FairMOT Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("FairMOT Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}
