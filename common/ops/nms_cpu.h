/**
 * NMS operation on cpu.
 * TODO: NMS on GPU.
 * 2020/11/20
 */

#ifndef NMS_CPU_H
#define NMS_CPU_H

#include <iostream>
#include <vector>
#include <algorithm>

#include "struct.h"

void nms_cpu(std::vector<Bbox> &bboxes, float threshold);

#endif  // NMS_CPU_H
