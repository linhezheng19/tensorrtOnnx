/**
 * Created by linhezheng
 * Tools for debug or something test.
 * 2020/11/16
*/

#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "struct.h"

using namespace std;

inline void vis_detection(BatchBox boxes, vector<cv::Mat>& ims) {
    for (int i = 0; i < ims.size(); i++) {
        vector<array<float, 5>> box = boxes[i];
        for (auto b : box) {
            cv::Point p1 = cv::Point(static_cast<int>(b[0]), static_cast<int>(b[1]));
            cv::Point p2 = cv::Point(static_cast<int>(b[2]), static_cast<int>(b[3]));
            cv::rectangle(ims[i], p1, p2, cv::Scalar(255, 204,0), 2);
            cv::Point p3 = cv::Point(static_cast<int>(b[0]), static_cast<int>(b[1] - 2));
            cv::putText(ims[i], to_string(static_cast<float>(b[4])), p3, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
        }
    }
}

#endif  // TOOLS_H
