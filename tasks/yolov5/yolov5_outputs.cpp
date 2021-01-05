#include "yolov5_outputs.h"

#include <cassert>

#include "misc.h"
#include "utils.h"
#include "nms_cpu.h"

// __device__ __inline__ float logist(float x) {
// 	return 1.f / (1.f + exp(-x));
// }

// =============Post Process=============>
BatchBox postProcess(vector<float*> inputs,vector<size_t> sizes, vector<nvinfer1::Dims> dims, YOLOParams yolo_params, const vector<cv::Mat>& imgs){
    assert(inputs.size() == sizes.size());
    assert(inputs.size() == dims.size());
	std::vector<Bbox> bboxes_nms;  // boxes after nms

#define CPU
#ifdef CPU
    int batch_size = dims[0].d[0];
	vector<vector<array<float, 5>>> batch_boxes;  // outputs
	for (int b = 0; b < batch_size; ++b) {
        int dh = 0;
		int dw = 0;
		float scale = 1;
        if (yolo_params.padding) {
            int ih = imgs[b].rows;
            int iw = imgs[b].cols;
            scale = std::min(static_cast<float>(yolo_params.width) / static_cast<float>(iw), static_cast<float>(yolo_params.height) / static_cast<float>(ih));
            int nh = static_cast<int>(scale * static_cast<float>(ih));
            int nw = static_cast<int>(scale * static_cast<float>(iw));
            dh = (yolo_params.height - nh) / 2;
            dw = (yolo_params.width - nw) / 2;
        }

		std::vector<Bbox> bboxes;
		Bbox bbox;
		for (int i = 0; i < inputs.size(); ++i) {
            int H             = dims[i].d[2];
            int W             = dims[i].d[3];
            int image_length  = W * H;
            int stride        = pow(2, i + 3);  // [8，16，32]
            int num_anchors   = dims[i].d[1];
            int num_outputs   = dims[i].d[4];
            int output_offset = num_anchors * image_length * num_outputs;
            
            size_t         output_size = output_offset * sizeof(float);
            float*         outputs     = (float*)malloc(output_size);
            vector<Anchor> anchors     = yolo_params.anchors[i];

            CUDA_CHECK(cudaMemcpy(outputs, static_cast<const float*>(inputs[i]) + output_offset * b, output_size, cudaMemcpyDeviceToHost));
//            cout << "* outputs " << * outputs << " "<< * (outputs + 1) << " " << * (outputs + 2) << endl;
            // decode yolov5 outputs
            for (int anchor_ind = 0; anchor_ind < num_anchors; ++anchor_ind) {
                float* output = outputs + anchor_ind * image_length * num_outputs;
                for (int pos = 0; pos < image_length * num_outputs; pos += num_outputs) {
                    const float* cls_ptr = output + pos + 5;
                    int   cid   = argmax(cls_ptr, cls_ptr + yolo_params.num_classes);
                    float score = sigmoid(output[pos + 4]) * sigmoid(cls_ptr[cid]);
                    if (score >= yolo_params.post_thresh) {
                        int im_pos = pos / num_outputs;
                        int grid_x = im_pos % W;
                        int grid_y = im_pos / W;
                        float cx = (sigmoid(output[pos]) * 2.f - 0.5f + static_cast<float>(grid_x)) * static_cast<float>(stride);
                        float cy = (sigmoid(output[pos + 1]) * 2.f - 0.5f + static_cast<float>(grid_y)) * static_cast<float>(stride);
                        float w  = pow(sigmoid(output[pos + 2]) * 2.f, 2) * static_cast<float>(anchors[anchor_ind].width);
                        float h  = pow(sigmoid(output[pos + 3]) * 2.f, 2) * static_cast<float>(anchors[anchor_ind].height);
                        bbox.xmin  = clip(static_cast<int>((cx - (w + 0.5) / 2 - dw) / scale), 0, yolo_params.width);
                        bbox.ymin  = clip(static_cast<int>((cy - (h + 0.5) / 2 - dh) / scale), 0, yolo_params.height);
                        bbox.xmax  = clip(static_cast<int>((cx + (w + 0.5) / 2 - dw) / scale), 0, yolo_params.width);
                        bbox.ymax  = clip(static_cast<int>((cy + (h + 0.5) / 2 - dh) / scale), 0, yolo_params.height);
                        bbox.score = score;
                        bbox.cid   = cid;
                        bboxes.emplace_back(bbox);
                    }
                }
            }
			free(outputs);
		}
		std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score > b2.score;});
		nms_cpu(bboxes, yolo_params.nms_thresh);
//      	for(auto box : bboxes)
//        {
//            cout << "  xmin, ymin, xmax, ymax : " << box.xmin
//                 << ", " << box.ymin
//                 << ", " << box.xmax
//                 << ", " << box.ymax
//                 << "  score : " << box.score
//                 << "  cid : " << box.cid
//                 <<endl;
//        }
		vector<array<float, 5>> one_img_box;
		for (int i = 0; i < bboxes.size(); ++i) {
			float x1 = static_cast<float>(bboxes[i].xmin);
			float y1 = static_cast<float>(bboxes[i].ymin);
			float x2 = static_cast<float>(bboxes[i].xmax);
			float y2 = static_cast<float>(bboxes[i].ymax);
			float c  = static_cast<float>(bboxes[i].score);
			array<float, 5> one_box = {x1, y1, x2, y2, c};
//			cout << "box : " << x1<<" " << y1<<" " << x2<<" " << y2<<" " << c<<endl;
			one_img_box.emplace_back(one_box);
		}
		batch_boxes.emplace_back(one_img_box);
	}
	return batch_boxes;
#else
    // TODO: GPU post process
#endif
}
