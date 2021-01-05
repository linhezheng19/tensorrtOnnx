#include "nms_cpu.h"

void nms_cpu(std::vector<Bbox> &bboxes, float threshold) {
	if (bboxes.empty()) {
		return ;
	}
	// 1.之前需要按照score排序
	std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});
	// 2.先求出所有bbox自己的大小
	std::vector<float> area(bboxes.size());
	for (int i=0; i<bboxes.size(); ++i) {
		area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
	}
	// 3.循环
	for (int i=0; i<bboxes.size(); ++i) {
		for (int j=i+1; j<bboxes.size(); ) {
			float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
			float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
			float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
			float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
			float width = std::max(right - left + 1, 0.f);
			float height = std::max(bottom - top + 1, 0.f);
			float u_area = height * width;
			float iou = (u_area) / (area[i] + area[j] - u_area);
			if (iou>=threshold) {
				bboxes.erase(bboxes.begin()+j);
				area.erase(area.begin()+j);
			} else {
				++j;
			}
		}
	}
}