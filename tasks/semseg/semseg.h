#ifndef SEMSEG_H
#define SEMSEG_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "logger.h"
#include "nhwc2nchw.h"
#include "struct.h"
#include "tasks.h"
#include "timer.h"
#include "utils.h"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace cv;

class SEMSEG : public SegmentationTask {
public:
    SEMSEG(const YAML::Node& cfg);
    ~SEMSEG() = default;

    vector<Mat> run(const vector<Mat>& imgs) override;

private:
    bool prepareInputs(const vector<Mat>& imgs) override;
    vector<Mat> processOutputs() override;

private:
    int mNumClasses;
};

#endif  // SEMSEG_H
