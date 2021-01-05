#include "semseg_outputs.h"

void postProcess(const vector<float*> outputs, vector<cv::Mat>& preds, const vector<nvinfer1::Dims>& dims) {
    // 调用cuda的转换，包括logits到class
    // class 到 bgr255
    // 复制到cpu然后存到Mat中
}
