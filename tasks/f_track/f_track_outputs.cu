#include "f_track_outputs.h"

#include <cassert>

#include "misc.h"
#include "nms_cpu.h"
#include "utils.h"

using namespace std;

__device__ __inline__ float logist(float x) {
    return 1.f / (1.f + exp(-x));
}

// =============Post Process=============>
void filter(std::vector<Bbox> &bboxes, float area_thresh, float ratio) {
    if (bboxes.empty()) {
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    auto iter = bboxes.begin();
    while (iter != bboxes.end()) {
        float area = ( iter->xmax - iter->xmin + 1) * (iter->ymax - iter->ymin + 1);
        if (area < area_thresh) {
            iter = bboxes.erase(iter);
        } else {
            ++iter;
        }
    }

}
__global__ void gather_feat_kernel(
        float* output,
        const float** reid_fs,
        const int* strides,
        const int* dims, // [w1, h1, w2, h2, ...]
        const int reid_dim,
        const int* ctrs, // [x1, y1, x2, y2, ...]
        const int* feat_ids,
        const int num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // if (i >= num * reid_dim) {
    // 	return;
    // }

    // int b_idx = i / reid_dim;
    // int c_idx = i % reid_dim;

    // when <<<num, reid_dim>>>
    int b_idx = blockIdx.x;
    int c_idx = threadIdx.x;

    int f_idx = feat_ids[b_idx];
    // int pos = ctrs[b_idx * 2 + 1] * dims[f_idx * 2] + ctrs[b_idx * 2] + c_idx * dims[f_idx * 2] * dims[f_idx * 2 + 1];
    int pos =  dims[f_idx * 2] * (ctrs[b_idx * 2 + 1] + dims[f_idx * 2 + 1] * c_idx) + ctrs[b_idx * 2];

    output[i] = *(reid_fs[f_idx] + pos);
}


void gather_feat(
        float* output,
        const float** reid_fs,
        const int* strides,
        const int* dims,
        const int reid_dim,
        const int* ctrs,
        const int* feat_ids,
        const int num) {
    gather_feat_kernel<<<num, reid_dim>>>(output, reid_fs, strides, dims,
                                          reid_dim, ctrs, feat_ids, num);
}

vector<vector<float>> getReidFeature_GPU(vector<Bbox> boxes, vector<float*> reid_fs, vector<nvinfer1::Dims> dims, vector<int> strides) {
    int num_boxes = boxes.size();
    int num_features = strides.size();
    int* feat_dims = new int[num_features * 2];
    int* ctrs = new int[num_boxes * 2];
    int* feat_ids = new int[num_boxes];

    int reid_dim = 0;
    for (int i = 0; i < num_features; ++i) {
        nvinfer1::Dims dim = dims[i];

        feat_dims[i * 2]     = dim.d[3];   // w
        feat_dims[i * 2 + 1] = dim.d[2];   // h
        reid_dim = std::max(dim.d[1], reid_dim);
    }

    for (int i = 0; i < num_boxes; ++i) {
        auto& box = boxes[i];
        int f_id = box.fea_index;
        feat_ids[i] = f_id;
        ctrs[i * 2]     = box.w;
        ctrs[i * 2 + 1] = box.h;
    }

    float ** reid_fs_cuda = NULL;
    int * strides_cuda = NULL;
    int * feat_dims_cuda = NULL;
    int * ctrs_cuda = NULL;
    int * feat_ids_cuda = NULL;

    CUDA_CHECK(cudaMalloc((void**)&reid_fs_cuda, num_features * sizeof(float*)));
    CUDA_CHECK(cudaMalloc((void**)&strides_cuda, num_features * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&feat_dims_cuda, 2 * num_features * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ctrs_cuda, 2 * num_boxes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&feat_ids_cuda, num_boxes * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(reid_fs_cuda, &reid_fs[0], num_features * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(strides_cuda, &strides[0], num_features * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(feat_dims_cuda, feat_dims, 2 * num_features * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctrs_cuda, ctrs, 2 * num_boxes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(feat_ids_cuda, feat_ids, num_boxes * sizeof(int), cudaMemcpyHostToDevice));

    float * output = NULL;
    CUDA_CHECK(cudaMalloc((void**)&output, num_boxes * reid_dim * sizeof(float)));

    gather_feat(
            output,
            (const float **)reid_fs_cuda,
            strides_cuda,
            feat_dims_cuda,
            reid_dim,
            ctrs_cuda,
            feat_ids_cuda,
            num_boxes);

    float * output_cpu = (float *)malloc(num_boxes * reid_dim * sizeof(float));;
    CUDA_CHECK(cudaMemcpy(output_cpu, output, num_boxes * reid_dim * sizeof(float), cudaMemcpyDeviceToHost));

    vector<vector<float>> out_reid_fs;
    for (int i = 0; i < num_boxes; ++i) {
        float* begin = output_cpu + i * reid_dim;
        vector<float> out_reid_f(begin, begin + reid_dim);
        out_reid_fs.emplace_back(out_reid_f);
    }

    cudaFree(reid_fs_cuda);
    cudaFree(strides_cuda);
    cudaFree(feat_dims_cuda);
    cudaFree(ctrs_cuda);
    cudaFree(feat_ids_cuda);
    cudaFree(output);
    free(output_cpu);

    return out_reid_fs;
}


vector<vector<float>> getReidFeature(vector<Bbox> boxes, vector<float*> reid_fs, vector<nvinfer1::Dims> dims, vector<int> strides) {
    vector<vector<float>> out_reid_fs;
    for (auto box: boxes){
        vector<float> out_reid_f;
        int id = box.fea_index;
        nvinfer1::Dims dim = dims[id];
        float* reid_f = reid_fs[id];
        int h = dim.d[2];
        int w = dim.d[3];
        int c = dim.d[1];
        int length = h * w;
        int cx = box.w;
        int cy = box.h;
//		int cx = static_cast<int>((box.xmin + box.xmax) / 2.0 / strides[id] + 0.5);  // 四舍五入: +0.5
//		int cy = static_cast<int>((box.ymin + box.ymax) / 2.0 / strides[id] + 0.5);
        // float* out_reid_f = new float[c*sizeof(float)];
        for (int i = 0; i < c; i++) {
            int pos = (cy)*w + cx;
            out_reid_f.push_back(*(reid_f + pos + i * length));
        }
        out_reid_fs.push_back(out_reid_f);
    }
    return out_reid_fs;

    // cout << "cx: " << cx << " cy: " << cy ;
    // float* out_reid = new float[reid_f_length * 4];

    // cout << endl << endl;
    // f_reids.push_back(f_reid);
}

std::pair<std::vector<std::vector<std::array<float, 5>>>, std::vector<std::vector<std::vector<float>>>>
f_track_postProcess(vector <float*> inputs,
        vector<size_t >sizes,
        vector<nvinfer1::Dims> dims,
        int mModel_H,
        int mModel_W,
        int NumClass,
        float postThres,
        float area_thresh,
        float  ratio,
        float nmsThres) {
    assert(inputs.size() == sizes.size());
    assert(inputs.size() == dims.size());
    std::vector<Bbox> bboxes_nms;  // outputs
    // 将所有features转换为bboxes [xmin, ymin, xmax, ymax, score, cid]

#define CPU
#ifdef CPU
    vector<vector<array<float, 5>>> batch_boxes;
    vector<vector<vector<float>>> batch_reid_feats;
    int batch_size = dims[0].d[0];
    for (int b = 0; b < batch_size; b++) {
        std::vector<Bbox> bboxes;
        Bbox bbox;

        int offset0 = dims[1].d[1] * dims[1].d[2] * dims[1].d[3];
        int offset1 = dims[5].d[1] * dims[5].d[2] * dims[5].d[3];
        int offset2 = dims[9].d[1] * dims[9].d[2] * dims[9].d[3];

        // size_t f_size0 = offset0 * sizeof(float);
        // size_t f_size1 = offset1 * sizeof(float);
        // size_t f_size2 = offset2 * sizeof(float);

        // float* reid_f0 = (float*)malloc(f_size0);
        // float* reid_f1 = (float*)malloc(f_size1);
        // float* reid_f2 = (float*)malloc(f_size2);
        // CUDA_CHECK(cudaMemcpy(reid_f0, (const float*)inputs[1] + offset0*b, f_size0, cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(reid_f1, (const float*)inputs[5] + offset1*b, f_size1, cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(reid_f2, (const float*)inputs[9] + offset2*b, f_size2, cudaMemcpyDeviceToHost));
        // vector<float*> features= {reid_f0, reid_f1, reid_f2};
        vector<nvinfer1::Dims> fea_dims={dims[1], dims[5], dims[9]};
        vector<int> strides = {8, 16, 32};

        for (int i = 0; i < inputs.size(); i += 4 ) {
            int stride = pow(2,(i / 4) + 3) ;  // [8，16，32]
            int H = dims[i].d[2];
            int W = dims[i].d[3];
            int length = H * W;
            //		cout << "length " << length << endl;
            //        std::cout << "i : "<< i<<endl
            ////          << "  sizes: " << sizes[i]
            ////          << "  sizes: " << sizes[i + 1]
            ////          << "  sizes: " << sizes[i + 2]
            ////          << "  sizes: " << sizes[i + 3]
            //          << "  stride " << stride
            //          << std::endl;

            int cls_offset = length * dims[i].d[1];
            int cen_offset = length * dims[i + 2].d[1];
            int reg_offset = length * dims[i + 3].d[1];

            size_t cls_size = cls_offset * sizeof(float);
            size_t cen_size = cen_offset * sizeof(float);
            size_t reg_size = reg_offset * sizeof(float);

            float* cls_f = (float*)malloc(cls_size);
            float* cen_f = (float*)malloc(cen_size);
            float* reg_f = (float*)malloc(reg_size);

            CUDA_CHECK(cudaMemcpy(cls_f, (const float*)inputs[i] + cls_offset * b, cls_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cen_f, (const float*)inputs[i + 2] + cen_offset * b, cen_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(reg_f, (const float*)inputs[i + 3] + reg_offset * b , reg_size, cudaMemcpyDeviceToHost));

            // CHW
            int index = 0;
            for (int pos = 0; pos < length; ++pos) {
                int cid = 0;
                float score = 0;
                for(int c = 0; c < NumClass; ++c){
                    float cls_score = sigmoid(cls_f[pos + length * c]) > 0.05? sigmoid(cls_f[pos + length * c]) : 0;
                    float tmp = sqrt(cls_score * sigmoid(cen_f[pos]));
                    if(tmp > score){
                        cid = c;
                        score = tmp;
                    }
                }

            if (score>=postThres) {
                int w = pos % W;
                int h = pos / W;
                bbox.xmin = clip(int(((w + 1) * stride) - reg_f[pos]), 0, mModel_W);
                bbox.ymin = clip(int(((h + 1) * stride) - reg_f[pos+length]), 0, mModel_H);
                bbox.xmax = clip(int(((w + 1) * stride) + reg_f[pos+length*2]), 0, mModel_W);
                bbox.ymax = clip(int(((h + 1) * stride) + reg_f[pos+length*3]), 0, mModel_H);
                bbox.score = score;
                bbox.cid = cid;
                bbox.w = w;
                bbox.h = h;
                bbox.fea_index = i / 4;
                bboxes.emplace_back(bbox);
            }
            index++;
            }
            // 取前topK个
            //free memery
            free(cls_f);
            free(reg_f);
            free(cen_f);
        }
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score > b2.score;});
    nms_cpu(bboxes, nmsThres);
    // filter(bboxes, area_thresh, ratio);

    vector<float*> features_gpu = {
        inputs[1] + offset0 * b,
        inputs[5] + offset1 * b,
        inputs[9] + offset2 * b,
    };

    vector<vector<float>> reid_results = getReidFeature_GPU(bboxes, features_gpu, fea_dims, strides);
    // vector<vector<float>> reid_results = getReidFeature(bboxes, features, fea_dims, strides);
    if (bboxes.size() != reid_results.size()) 
        cout << "Box size != ReID Feature size.";
    vector<array<float, 5>> one_img_box;
    for (int i = 0; i < bboxes.size(); i++) {
        float x1 = static_cast<float>(bboxes[i].xmin);
        float y1 = static_cast<float>(bboxes[i].ymin);
        float x2 = static_cast<float>(bboxes[i].xmax);
        float y2 = static_cast<float>(bboxes[i].ymax);
        float c = static_cast<float>(bboxes[i].score);
        array<float, 5> one_box = {x1, y1, x2, y2, c};
        // cout << "box : " << x1<<" " << y1<<" " << x2<<" " << y2<<" " << c<<endl;
        one_img_box.push_back(one_box);
    }
    batch_boxes.push_back(one_img_box);
    batch_reid_feats.push_back(reid_results);
    // free(reid_f0);
    // free(reid_f1);
    // free(reid_f2);
    }
    return std::make_pair(batch_boxes, batch_reid_feats);
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
        for (int pos = 0; pos < length; ++pos) {
            int cid = 0;
            float score = 0;
            for (int c = 0; c < NumClass; ++c) {
                float tmp = logist(cls_f[pos + length * c]) * logist(cen_f[pos]);
                if (tmp > score) {
                    cid = c;
                    score = tmp;
                }
            }
//
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
    for (int i = 0; i < nms_idx.size(); ++i) {
        bboxes_nms[i] = bboxes[nms_idx[i]];
    }
    CUDA_CHECK(cudaFree(candidate_boxes));
    // return bboxes_nms;
    return make_pair(batch_boxes, batch_reid_feats);
#endif
}