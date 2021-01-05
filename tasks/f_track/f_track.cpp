#include <array>
#include <string>

#include "f_track.h"
#include "f_track_outputs.h"

FTrack::FTrack(const YAML::Node& cfg) : TrackTask(cfg) {
    mNumClasses = cfg["params"]["num_classes"].as<int>();
}

bool FTrack::prepareInputs(const vector<Mat>& imgs) {
    return TrackTask::prepareInputs(imgs);
}

TrackRes FTrack::processOutputs() {
    vector <float*> inputs;
    vector<size_t >sizes;
    vector<nvinfer1::Dims> dims;
    vector<int> idx_list = cfg["params"]["output_index"].as<vector<int>>();
    for (int i = 0; i < 12; ++i)
    {
        //fcos  onnx outputs from 1-15
        inputs.push_back((float*)mNet->GetBindingPtr(idx_list[i]));
        sizes.push_back((size_t)mNet->GetBindingSize(idx_list[i]));
        dims.push_back(mNet->GetBindingDims(idx_list[i]));
    }
    // results type -> std::pair<std::vector<std::vector<std::array<float, 5>>>, std::vector<std::vector<std::vector<float>>>>
    float det_thresh   = cfg["params"]["det_thresh"] ? cfg["params"]["det_thresh"].as<float>() : 0.;
    float area_thresh  = cfg["params"]["area_thresh"] ? cfg["params"]["area_thresh"].as<float>() : 0.;
    float ratio_thresh = cfg["params"]["ratio_thresh"] ? cfg["params"]["ratio_thresh"].as<float>() : 0.;
    float nms_thresh   = cfg["params"]["nms_thresh"].as<float>();
    auto results = f_track_postProcess(inputs, sizes, dims, mModel_H, mModel_W,  mNumClasses, det_thresh, area_thresh, ratio_thresh, nms_thresh);
    std::vector<std::vector<std::array<float, 5>>> boxes = results.first;
    // cout << "box1: "<<boxes[0][0][0] << " " << boxes[0][0][1] << " " << boxes[0][0][2] << " "<< boxes[0][0][3]<< " "<< boxes[0][0][4]<< endl;

    return results;
}

TrackRes FTrack::run(const vector<Mat>& imgs){
    mTimer->dataStart();
    if (!prepareInputs(imgs)){
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
        mLogger.logger("FTrack Data time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("FTrack Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("FTrack Post time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}
