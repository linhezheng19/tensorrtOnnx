#ifndef F_TRACK_H
#define F_TRACK_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #include "engine.h"
#include "logger.h"
#include "nhwc2nchw.h"
#include "struct.h"
#include "timer.h"
#include "utils.h"
#include "tasks.h"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace cv;

class FTrack : public TrackTask {
public:
    FTrack(const YAML::Node& cfg);
    ~FTrack() = default;

    TrackRes run(const vector<Mat>& imgs) override;

private:
    bool prepareInputs(const vector<Mat>& imgs) override;
    TrackRes processOutputs() override;

private:
    int mNumClasses;
};

#endif  // F_TRACK_H
