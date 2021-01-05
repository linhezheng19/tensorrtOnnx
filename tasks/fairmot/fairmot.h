#ifndef FAIRMOT_H
#define FAIRMOT_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #include "engine.h"
#include "logger.h"
#include "utils.h"
#include "timer.h"
#include "yaml-cpp/yaml.h"
#include "nhwc2nchw.h"
#include "det_post_processor.h"
#include "tasks.h"

using namespace std;
using namespace cv;

class FairMOT : public TrackTask {
public:
    FairMOT(const YAML::Node& cfg);
    ~FairMOT();

    TrackRes run(const vector<Mat>& imgs);

private:
    bool prepareInputs(const vector<Mat>& imgs) override;
    TrackRes processOutputs() override;

private:
    DetPostProcessor *mDetPostProcessor;
};

#endif  /// FAIRMOT_H
