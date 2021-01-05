#ifndef FCOS_H
#define FCOS_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #include "engine.h"
#include "logger.h"
#include "nhwc2nchw.h"
#include "struct.h"
#include "timer.h"
#include "utils.h"
#include "yaml-cpp/yaml.h"
#include "tasks.h"
#include "cls.h"

using namespace std;
using namespace cv;

class FCOS : public DetectionTask {
public:
    FCOS(const YAML::Node& cfg);
    ~FCOS() = default;

    BatchBox run(const vector<Mat>& imgs) override;

private:
    bool prepareInputs(const vector<Mat>& imgs) override;
    BatchBox processOutputs() override;

private:
    int mNumClasses;
};

#endif  // FCOS_H
