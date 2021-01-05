/**
 * Created by linhezheng.
 * Basic Task API.
 * 2020/09/01
 */

#include "tasks.h"

/* -==================Base Task Class================*/
Task::Task(const YAML::Node& cfg) : cfg(cfg) {
    // init member variables
    mNX_ON         = cfg["engine"]["nx"].as<bool>();
    mGPU_ID        = cfg["engine"]["gpu_id"].as<int>();
    mBatchSize     = cfg["engine"]["bchw"].as<vector<int>>()[0];
    mModel_W       = cfg["engine"]["bchw"].as<vector<int>>()[3];
    mModel_H       = cfg["engine"]["bchw"].as<vector<int>>()[2];
    mWorkspaceSize = cfg["engine"]["workspace"].as<int>();
    mOnnxFile      = cfg["engine"]["onnx_file"].as<string>();
    mEngineFile    = cfg["engine"]["engine_file"].as<string>();
    CUDA_CHECK(cudaStreamCreate(&mStream));

    // create timer
    mTimer = new Timer(mStream, cfg["misc"]["show_time"].as<bool>());

    // init Net
    int mode = cfg["engine"]["mode"].as<int>();
    switch(mode) {
        case (32): mRunMode = RunMode::kFP32; break;
        case (16): mRunMode = RunMode::kFP16; break;
        case (8) : mRunMode = RunMode::kINT8; break;
        default  : mRunMode = RunMode::kFP32; break;
    }

    if (!mNX_ON) cudaSetDevice(mGPU_ID);
    mNet = new RTEngine();

    if (!initEngine()) {
        mLogger.logger("Initialize RT Engine Failed!", logger::LEVEL::ERROR);
    }

    CUDA_CHECK(cudaMalloc((void**)&mInputDataNHWC, mBatchSize * 3 * mModel_W * mModel_H * sizeof(uint8_t)));
}

Task::~Task() {
    if (mNet) {
        delete mNet;
        mNet = nullptr;
    }
    if (mTimer) {
        delete mTimer;
        mTimer = nullptr;
    }
    CUDA_CHECK(cudaFree(mInputDataNHWC));
    CUDA_CHECK(cudaStreamDestroy(mStream));
}

bool Task::initEngine() {
    if (mOnnxFile.empty()) {
        mLogger.logger("ONNX file not specified! Set it in specific yaml file.", logger::LEVEL::ERROR);
    }
    if (mEngineFile.empty()) {
        mLogger.logger("Engine file not specified! Set it in specific yaml file.", logger::LEVEL::ERROR);
    }
    mNet->SetDevice(mGPU_ID);
    mNet->CreateEngine(mOnnxFile, mEngineFile, mOutputNames, mBatchSize, mRunMode, mWorkspaceSize);

    return true;
}

bool Task::prepareInputs(const vector<Mat>& imgs) {
    int img_stride = 3 * mModel_W * mModel_H;
    for (int i = 0; i < mBatchSize; ++i) {
        CUDA_CHECK(cudaMemcpy(mInputDataNHWC + i * img_stride, imgs[i].data, img_stride * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    // debug code
//    cout << "imgs[i].data " << (float)imgs[0].data[0] << " "<< (float)imgs[0].data[1] << " "<< (float)imgs[0].data[2]<<endl;
    vector<float> means = cfg["params"]["means"].as<vector<float>>();
    vector<float> stds  = cfg["params"]["stds"].as<vector<float>>();
    string color_mode   = cfg["params"]["color_mode"].as<string>();
    bool bgr_mode = true;
    if (color_mode == "rgb")
        bgr_mode = false;
    NHWC2NCHW(
            mInputDataNHWC,
            (float*)mNet->GetBindingPtr(0),
            mBatchSize,
            mModel_H,
            mModel_W,
            means[0], means[1], means[2],
            stds[0], stds[1], stds[2],
            bgr_mode);
    // debug code
//    uint8_t* cls_f = (uint8_t*)malloc(320*320*3*sizeof(uint8_t));
//    cudaMemcpy(cls_f, mInputDataNHWC, 320*320*3*sizeof(uint8_t), cudaMemcpyDeviceToHost);
//    cout << "* GetBindingPtr" << (unsigned)cls_f[144450] << " " <<(unsigned) (cls_f[144451]) << " "<<(unsigned) (cls_f[144452]) <<endl;
    return true;
}

/* -==================Classification Task Class================*/
ClassificationTask::ClassificationTask(const YAML::Node& cfg) : Task(cfg) {}

bool ClassificationTask::prepareInputs(const vector<Mat>& imgs) {
    return Task::prepareInputs(imgs);
}

vector<int> ClassificationTask::run(const vector<Mat>& imgs) {
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
        mLogger.logger("Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}

/* -==================Detection Task Class================*/
DetectionTask::DetectionTask(const YAML::Node& cfg) : Task(cfg) {}

bool DetectionTask::prepareInputs(const vector<Mat>& imgs) {
    return Task::prepareInputs(imgs);
}

BatchBox DetectionTask::run(const vector<Mat>& imgs) {
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
        mLogger.logger("Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}

/* -==================Track Task Class================*/
TrackTask::TrackTask(const YAML::Node& cfg) : Task(cfg) {}

bool TrackTask::prepareInputs(const vector<Mat>& imgs) {
    return Task::prepareInputs(imgs);
}

TrackRes TrackTask::run(const vector<Mat>& imgs){
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
        mLogger.logger("Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}

/* -==================Segmentation Task Class================*/
SegmentationTask::SegmentationTask(const YAML::Node& cfg) : Task(cfg) {}

bool SegmentationTask::prepareInputs(const vector<Mat>& imgs) {
    return Task::prepareInputs(imgs);
}

vector<Mat> SegmentationTask::run(const vector<Mat>& imgs) {
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
        mLogger.logger("Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}

/* -==================Keypoint Task Class================*/
KeypointTask::KeypointTask(const YAML::Node& cfg) : Task(cfg) {}

bool KeypointTask::prepareInputs(const vector<Mat>& imgs) {
    return Task::prepareInputs(imgs);
}

vector<int> KeypointTask::run(const vector<Mat>& imgs) {
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
        mLogger.logger("Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}
