#include "cls.h"

#include <array>

CLS::CLS(const YAML::Node& cfg) : ClassificationTask(cfg) {
    mNumClasses = cfg["params"]["num_classes"].as<int>();
}

bool CLS::prepareInputs(const vector<Mat>& imgs) {
    return ClassificationTask::prepareInputs(imgs);
}

bool CLS::prepareInputs(uint8_t* imgs) {
    NHWC2NCHW(
            mInputDataNHWC,
            (float*)mNet->GetBindingPtr(0),
            mBatchSize,
            mModel_H,
            mModel_W,
            103.52, 116.28, 123.675,
            57.375,57.12,58.395,
            mImageFormat);
//    float* cls_f = (float*)malloc(10*sizeof(float));
//    cudaMemcpy(cls_f, (float*)mNet->GetBindingPtr(0), 10*sizeof(float), cudaMemcpyDeviceToHost);
//    cout << "* GetBindingPtr0" << * cls_f << * (cls_f + 1) << * (cls_f + 2) <<endl;
     return true;
}

vector<int> CLS::processOutputs() {
    vector<float> cls_res(mNumClasses * mBatchSize);
    int cls_bind_idx = 1;
    vector<int> labels;
    mNet->CopyFromDeviceToHost(cls_res, cls_bind_idx, mStream);
    for (int b = 0; b < mBatchSize; ++b) {
        float max_score = 0.f;
        int label = -1;
        for (int i = b * mNumClasses; i < (b * mNumClasses + mNumClasses); i++) {
            max_score = cls_res[i] > max_score ? cls_res[i] : max_score;
            label = cls_res[i] > max_score ? (i % mNumClasses) : label;
        }
        labels.push_back(label);
    }
    cout << "cls result is : " << labels.size()<< endl;
//    cout << "net output: " << cls_res[0] << " " << cls_res[1] << " " << cls_res[2] << endl;
    return labels;
}

vector<int> CLS::run(const vector<Mat>& imgs) {
    return ClassificationTask::run(imgs);
}

vector<int> CLS::run(uint8_t* p_input) {
    mTimer->dataStart();
    if (!prepareInputs(p_input)) {
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
        mLogger.logger("CLS Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("CLS Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("CLS Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}
