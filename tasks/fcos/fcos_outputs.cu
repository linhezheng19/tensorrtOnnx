#include "fcos_outputs.h"

#include <cassert>

#include "misc.h"
#include "nms_cpu.h"
#include "utils.h"

using namespace std;

__device__ __inline__ float logist(float x) {
	return 1.f / (1.f + exp(-x));
}

// =============Post Process=============>

BatchBox postProcess(vector <float*> inputs,vector<size_t >sizes, vector<nvinfer1::Dims> dims, int mModel_H, int mModel_W, int NumClass, float postThres, float nmsThres) {
    assert(inputs.size() == sizes.size());
    assert(inputs.size() == dims.size());
	std::vector<Bbox> bboxes_nms;  // outputs
    // 将所有features转换为bboxes [xmin, ymin, xmax, ymax, score, cid]

#define CPU
#ifdef CPU
    int batch_size = dims[0].d[0];
	vector<vector<array<float, 5>>> batch_boxes;
//	batch_boxes.resize(batch_size);

	for (int b = 0; b < batch_size; ++b) {
		std::vector<Bbox> bboxes;
		Bbox bbox;
		for(int i = 0; i < inputs.size(); i += 3) {
			int stride = pow(2,(i / 3) + 3) ;  // [8，16，32]
			int H = dims[i].d[2];
			int W = dims[i].d[3];
			int length = H * W;

			int cls_offset = length * dims[i].d[1];
			int cen_offset = length * dims[i + 1].d[1];
			int reg_offset = length * dims[i + 2].d[1];

			size_t cls_size = cls_offset * sizeof(float);
			size_t cen_size = cen_offset * sizeof(float);
			size_t reg_size = reg_offset * sizeof(float);

			float* cls_f = (float*)malloc(cls_size);
			float* cen_f = (float*)malloc(cen_size);
			float* reg_f = (float*)malloc(reg_size);

			CUDA_CHECK(cudaMemcpy(cls_f, (const float*)inputs[i] + cls_offset * b, cls_size, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(cen_f, (const float*)inputs[i + 1] + cen_offset * b, cen_size, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(reg_f, (const float*)inputs[i + 2] + reg_offset * b , reg_size, cudaMemcpyDeviceToHost));
//            cout << "* cls_f" << * cls_f << * (cls_f + 1) << * (cls_f + 2) <<endl;
//		    cout << "* reg_f" << * reg_f << * (reg_f + 1) << * (reg_f + 2) <<endl;
//		    cout << "* cen_f" << * cen_f << * (cen_f + 1) << * (cen_f + 2) <<endl;
			// CHW
			int index = 0;
			for (int pos = 0; pos < length; ++pos) {
				int cid = 0;
				float score = 0;
				for (int c = 0; c < NumClass; ++c) {
					float cls_score = sigmoid(cls_f[pos + length * c]) > 0.05? sigmoid(cls_f[pos + length * c]) : 0;
					float tmp = sqrt(cls_score * sigmoid(cen_f[pos]));
					if (tmp > score) {
						cid = c;
						score = tmp;
					}
				}

				if (score >= postThres) {
					int w = pos % W;
					int h = pos / W;
					bbox.xmin = clip(int(((w + 1) * stride) - reg_f[pos]), 0, mModel_W);
					bbox.ymin = clip(int(((h + 1) * stride) - reg_f[pos+length]), 0, mModel_H);
					bbox.xmax = clip(int(((w + 1) * stride) + reg_f[pos+length*2]), 0, mModel_W);
					bbox.ymax = clip(int(((h + 1) * stride) + reg_f[pos+length*3]), 0, mModel_H);
					bbox.score = score;
					bbox.cid = cid;
					bboxes.emplace_back(bbox);
				}
				++index;
			}
			// 取前topK个
			//free memery
			free(cls_f);
			free(reg_f);
			free(cen_f);

		}

		std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score > b2.score;});
		nms_cpu(bboxes, nmsThres);
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
			float c = static_cast<float>(bboxes[i].score);
			array<float, 5> one_box = {x1, y1, x2, y2, c};
//			 cout << "box : " << x1<<" " << y1<<" " << x2<<" " << y2<<" " << c<<endl;
			one_img_box.push_back(one_box);
		}
		batch_boxes.push_back(one_img_box);
//
	}
	return batch_boxes;
#else
    // GPU
	float * candidate_boxes =NULL;
	size_t size = 100 * 6 * sizeof(float);
	CUDA_CHECK(cudaMalloc((void**)&candidate_boxes, size));
    float * ptr_candidate_boxes = candidate_boxes;
	size_t candidate_boxes_size = 0;
	for (int i = 0; i < inputs.size(); i += 3) {
		int stride = pow(2,(i / 3) + 4) ;  // [16,32,64,128,256]
		int H = dims[i].d[2];
		int W = dims[i].d[3];
		int length = H * W;
		float* cls_f = (float*)inputs[i];
		float* cls_f_sigmod = (float*)malloc(sizes[i]);
		sigmoid_gpu2cpu(cls_f, cls_f_sigmod, sizes[i], 0);  //todo  multi gpu
		float* reg_f = (float*)inputs[i + 2];

		float* cen_f = (float*)inputs[i + 1];
		float* cen_f_sigmod = (float*)malloc(sizes[i + 1]);
		sigmoid_gpu2cpu(cen_f, cen_f_sigmod, sizes[i + 1], 0);  //todo  multi gpu
		for(int pos = 0; pos < length; ++pos) {
			int cid = 0;
			float score = 0;
			for (int c = 0; c < NumClass; ++c) {
				float tmp = logist(cls_f[pos + length * c]) * logist(cen_f[pos]);
				if (tmp > score) {
					cid = c;
					score = tmp;
				}
			}
			if (score >= postThres) {
	//				cout << "score" << score << endl;
				int w = pos % W;
				int h = pos / W;
				*ptr_candidate_boxes       = clip(int((w * stride + stride / 2) - sigmoid(reg_f[pos])), 0, W - 1);
				*(ptr_candidate_boxes + 1) = clip(int((h * stride + stride / 2) - reg_f[pos + length]), 0, H - 1);
				*(ptr_candidate_boxes + 1) = clip(int((w * stride + stride / 2) + reg_f[pos + length * 2]), 0, W - 1);
				*(ptr_candidate_boxes + 1) = clip(int((h * stride + stride / 2) + reg_f[pos + length * 3]), 0, H - 1);
				*(ptr_candidate_boxes + 1) = score;
				*(ptr_candidate_boxes + 1) = cid;
				ptr_candidate_boxes += 6;
				candidate_boxes_size += 6;
			}
		}
	}
	    // TODO 按类别做nms
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score > b2.score;});
    std::vector<int> nms_idx = nms(bboxes, nmsThres);
    std::vector<Bbox> bboxes_nms(nms_idx.size());
    for (int i = 0; i < nms_idx.size(); ++i){
        bboxes_nms[i] = bboxes[nms_idx[i]];
    }
	CUDA_CHECK(cudaFree(candidate_boxes));
	// return bboxes_nms;
//	return make_pair(batch_boxes, batch_reid_feats);
#endif
}
