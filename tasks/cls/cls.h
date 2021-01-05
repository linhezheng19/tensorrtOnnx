#ifndef CLS_H
#define CLS_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tasks.h"
#include "logger.h"
#include "nhwc2nchw.h"
#include "structs.h"
#include "timer.h"
#include "utils.h"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace cv;

class CLS : public ClassificationTask {
public:
    CLS(const YAML::Node& cfg);

    vector<int> run(const vector<Mat>& imgs) override;
    vector<int> run(uint8_t* p_input);
    uint8_t* getInputPtr() {
        return mInputDataNHWC;
    }
    int getInputSize() {
        return 3 * mModel_W * mModel_H;
    }

private:
    bool prepareInputs(const vector<Mat>& imgs) override;
    bool prepareInputs(uint8_t* imgs);
    vector<int> processOutputs() override;

private:
    int mNumClasses;
};

#endif  // CLS_H
