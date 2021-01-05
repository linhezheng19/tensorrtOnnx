/**
 * Created by linhezheng.
 * Basic Task API. Task is basic class of all tasks, and other classes of task
 * is basic class for relevant methods. The main purpose is to achieve some 
 * common operations.
 * 2020/11/20
 */

#ifndef TASKS_H
#define TASKS_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "engine.h"
#include "structs.h"
#include "logger.h"
#include "timer.h"
#include "utils.h"
#include "nhwc2nchw.h"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace cv;

/* -==================Base Task Class================*/
class Task
{
public:
    Task(const YAML::Node& cfg);
    virtual ~Task();

protected:
    /**
    ! Base task provided two basic method.
    ! initEngine: call api in tensorrt.h to build a engine.
    ! prepareInputs: take vector of cv::Mat as task's input, do pre-process in it.
    */
    virtual bool initEngine();
    virtual bool prepareInputs(const vector<Mat>& imgs);

protected:
    RTEngine*      mNet;
    cudaStream_t   mStream;
    RunMode        mRunMode;
    Timer*         mTimer;
    logger::Logger mLogger;

    YAML::Node cfg;

    int mBatchSize;
    int mModel_W;
    int mModel_H;
    int mGPU_ID;
    bool mNX_ON;
    long mWorkspaceSize;
    string mOnnxFile;
    string mEngineFile;
    uint8_t* mInputDataNHWC;
    vector<string> mOutputNames {};
};

/* -==================Classification Task Class================*/
class ClassificationTask : public Task
{
public:
    /**
    ! Instance inteface.
    ! run: run all pipeline of task: prepare inputs, inference, process outputs, recommend override it.
    */
    virtual vector<int> run(const vector<Mat>& imgs);

protected:
    /**
    ! Base classification task provided some basic method.
    ! prepareInputs: inherit from Task.
    ! processOutputs: get outputs from engine and process as you want, shold override it.
    */
    ClassificationTask(const YAML::Node& cfg);
    virtual ~ClassificationTask() = default;
    virtual bool prepareInputs(const vector<Mat>& imgs) override;
    virtual vector<int> processOutputs() {};
};
/* -==================Detection Task Class================*/
class DetectionTask : public Task
{
public:
    /**
    ! Instance inteface.
    ! run: run all pipeline of task: prepare inputs, inference, process outputs, recommend override it.
    */
    virtual BatchBox run(const vector<Mat>& imgs);

protected:
    /**
    ! Base detection task provided some basic method.
    ! prepareInputs: inherit from Task.
    ! processOutputs: get outputs from engine and process as you want, shold override it.
    */
    DetectionTask(const YAML::Node& cfg);
    virtual ~DetectionTask() = default;
    virtual bool prepareInputs(const vector<Mat>& imgs) override;
    virtual BatchBox processOutputs() {};
};

/* -==================Track Task Class================*/
class TrackTask : public Task
{
public:
    /**
    ! Instance inteface.
    ! run: run all pipeline of task: prepare inputs, inference, process outputs, recommend override it.
    */
    virtual TrackRes run(const vector<Mat>& imgs);

protected:
    /**
    ! Base track task provided some basic method.
    ! prepareInputs: inherit from Task.
    ! processOutputs: get outputs from engine and process as you want, shold override it.
    */
    TrackTask(const YAML::Node& cfg);
    virtual ~TrackTask() = default;
    virtual bool prepareInputs(const vector<Mat>& imgs) override;
    virtual TrackRes processOutputs() {};
};

/* -==================Segmentation Task Class================*/
class SegmentationTask : public Task
{
public:
    /**
    ! Instance inteface.
    ! run: run all pipeline of task: prepare inputs, inference, process outputs, recommend override it.
    */
    virtual vector<Mat> run(const vector<Mat>& imgs);

protected:
    /**
    ! Base segmentation task provided some basic method.
    ! prepareInputs: inherit from Task.
    ! processOutputs: get outputs from engine and process as you want, shold override it.
    */
    SegmentationTask(const YAML::Node& cfg);
    virtual ~SegmentationTask() = default;
    virtual bool prepareInputs(const vector<Mat>& imgs) override;
    virtual vector<Mat> processOutputs() {};
};

/* -==================Keypoint Task Class================*/
class KeypointTask : public Task
{
public:
    /**
    ! Instance inteface.
    ! run: run all pipeline of task: prepare inputs, inference, process outputs, recommend override it.
    */
    virtual vector<int> run(const vector<Mat>& imgs);

protected:
    /**
    ! Base keypoint task provided some basic method.
    ! prepareInputs: inherit from Task.
    ! processOutputs: get outputs from engine and process as you want, shold override it.
    */
    KeypointTask(const YAML::Node& cfg);
    virtual ~KeypointTask() = default;
    virtual bool prepareInputs(const vector<Mat>& imgs) override;
    virtual vector<int> processOutputs() {};
};

#endif  // TASKS_H
