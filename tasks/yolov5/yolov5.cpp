#include <array>
#include <algorithm>

#include "yolov5.h"
#include "yolov5_outputs.h"

YOLOV5::YOLOV5(const YAML::Node& cfg) : DetectionTask(cfg) {
    initParams();
}

void YOLOV5::initParams() {
    vector<vector<Anchor>> anchors;
    vector<vector<int>> anchors_xy = cfg["params"]["anchors"].as<vector<vector<int>>>();
    for (auto anchor_xy : anchors_xy) {
        vector<Anchor> anchor;
        for (int i = 0; i < anchor_xy.size(); i += 2) {
            Anchor single_anchor;
            single_anchor.width = anchor_xy[i];
            single_anchor.height = anchor_xy[i+1];
            anchor.emplace_back(single_anchor);
        }
        anchors.emplace_back(anchor);
    }
    mYoloParams.width       = mModel_W;
    mYoloParams.height      = mModel_H;
    mYoloParams.anchors     = anchors;
    mYoloParams.num_classes = cfg["params"]["num_classes"].as<int>();
    mYoloParams.nms_thresh  = cfg["params"]["nms_thresh"].as<float>();
    mYoloParams.post_thresh = cfg["params"]["post_thresh"].as<float>();
    mYoloParams.padding     = cfg["params"]["padding"].as<bool>();
    mYoloParams.color_mode  = cfg["params"]["color_mode"].as<string>();
}

bool YOLOV5::prepareInputs(const vector<Mat>& imgs) {
    vector<Mat> processed_ims;
    for (auto im : imgs) {
        cv::Mat processed_im(mYoloParams.height, mYoloParams.width, CV_8UC3);
        int dh = 0;
        int dw = 0;
        if (mYoloParams.padding) {
            int ih = im.rows;
            int iw = im.cols;
            float scale = std::min(static_cast<float>(mYoloParams.width) / static_cast<float>(iw), static_cast<float>(mYoloParams.height) / static_cast<float>(ih));
            int nh = static_cast<int>(scale * static_cast<float>(ih));
            int nw = static_cast<int>(scale * static_cast<float>(iw));
            dh = (mYoloParams.height - nh) / 2;
            dw = (mYoloParams.width - nw) / 2;
            cv::resize(im, processed_im, cv::Size(nw, nh));
            cv::copyMakeBorder(processed_im, processed_im, dh, mYoloParams.height-nh-dh, dw, mYoloParams.width-nw-dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        } else {
            cv::resize(im, processed_im, cv::Size(mYoloParams.width, mYoloParams.height));
        }
        processed_ims.emplace_back(processed_im);
    }

    int img_stride = 3 * mModel_W * mModel_H;
    for (int i = 0; i < mBatchSize; ++i) {
        cudaMemcpy(
                mInputDataNHWC + i * img_stride,
                processed_ims[i].data,
                img_stride * sizeof(uint8_t),
                cudaMemcpyHostToDevice);
    }
//    cout << "imgs[i].data " << (float)processed_ims[0].data[0] << " "<< (float)processed_ims[0].data[1] << " "<< (float)processed_ims[0].data[2]<<endl;
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
//    float* test_im = (float*)malloc(640*640*3*sizeof(float));
//    cudaMemcpy(test_im, (float*)mNet->GetBindingPtr(0), 640*640*3*sizeof(float), cudaMemcpyDeviceToHost);
//    cout << "imgsL : " << test_im[0] << " " << test_im[1] << " " << test_im[2] << endl;
    return true;
}

BatchBox YOLOV5::processOutputs() {
    vector <float*> inputs;
    vector<size_t >sizes;
    vector<nvinfer1::Dims> dims;
    vector<int> idx_list = cfg["params"]["output_index"].as<vector<int>>();
    for (int i = 0; i < 3; ++i) {
        inputs.push_back((float*)mNet->GetBindingPtr(idx_list[i]));
        sizes.push_back((size_t)mNet->GetBindingSize(idx_list[i]));
        dims.push_back(mNet->GetBindingDims(idx_list[i]));
    }
    BatchBox results = postProcess(inputs, sizes, dims, mYoloParams, mInputImgs);
    return results;
}

BatchBox YOLOV5::run(const vector<Mat>& imgs) {
    mInputImgs = imgs;  // for decode in post process
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
        mLogger.logger("YOLO Data  time: ", mTimer->getDataTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("YOLO Infer time: ", mTimer->getInferTime(), "ms", logger::LEVEL::INFO);
        mLogger.logger("YOLO Post  time: ", mTimer->getPostTime(), "ms", logger::LEVEL::INFO);
    }

    return results;
}
