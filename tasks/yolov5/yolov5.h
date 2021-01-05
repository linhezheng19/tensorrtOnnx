#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yaml-cpp/yaml.h"

// #include "engine.h"
#include "logger.h"
#include "nhwc2nchw.h"
#include "struct.h"
#include "timer.h"
#include "utils.h"
#include "tasks.h"

using namespace std;
using namespace cv;

struct Anchor{
    int width;
    int height;
};

struct YOLOParams{
    int width;
    int height;
    int num_classes;
    float nms_thresh;
    float post_thresh;
    bool padding;
	std::string color_mode;
    std::vector<std::vector<Anchor>> anchors;
};

class YOLOV5 : DetectionTask {
public:
    YOLOV5(const YAML::Node& cfg);
    BatchBox run(const vector<Mat>& imgs) override;

private:
    void initParams();
    bool prepareInputs(const vector<Mat>& imgs) override;
    BatchBox processOutputs() override;

private:
    YOLOParams mYoloParams;
    vector<Mat> mInputImgs;
};

#endif  // YOLOV5_H
