#include <iostream>
#include <string>

#include "cls.h"
#include "semseg.h"
#include "fcos.h"
#include "yolov5.h"
#include "f_track.h"
#include "fairmot.h"
#include "tasks.h"
#include "tools.h"
#include "utils.h"
#include "yaml-cpp/yaml.h"
#include <thread>
#include <chrono>

using namespace std;
using namespace cv;


int main(){
    //cfg
    YAML::Node main_cfg = YAML::LoadFile("../cfgs/main.yaml");
    YAML::Node task = main_cfg["tasks"];
    YAML::Node f_track_cfg, fcos_cfg, fairmot_cfg, cls_cfg, yolo_cfg, semseg_cfg;

    FTrack* f_track = nullptr;
    FCOS*    fcos    = nullptr;
    FairMOT* fairmot = nullptr;
    CLS*     cls     = nullptr;
    YOLOV5*  yolo    = nullptr;
    SEMSEG*  semseg  = nullptr;

/* -==================Classification task================*/
    if (task["cls"] && task["cls"].as<bool>()){
        string cfg_file = main_cfg["cls"]["cfg_file"].as<string>();
        cls_cfg =  YAML::LoadFile(cfg_file);
        cls = new CLS(cls_cfg);
    }
/* -==================Segmentation task================*/
    if (task["semseg"] && task["semseg"].as<bool>()){
        string semseg_file = main_cfg["semseg"]["cfg_file"].as<string>();
        semseg_cfg =  YAML::LoadFile(semseg_file);
        semseg = new SEMSEG(semseg_cfg);
    }
/* -==================Track task=======================*/
    if (task["fairmot"] && task["fairmot"].as<bool>()){
        string cfg_file = main_cfg["fairmot"]["cfg_file"].as<string>();
        fairmot_cfg =  YAML::LoadFile(cfg_file);
        fairmot = new FairMOT(fairmot_cfg);
    }
    if (task["f_track"] && task["f_track"].as<bool>()){
        string cfg_file = main_cfg["f_track"]["cfg_file"].as<string>();
        f_track_cfg =  YAML::LoadFile(cfg_file);
        f_track = new FTrack(f_track_cfg);
    }
/* -==================Detection task=================*/
    if (task["fcos"] && task["fcos"].as<bool>()){
        string cfg_file = main_cfg["fcos"]["cfg_file"].as<string>();
        fcos_cfg =  YAML::LoadFile(cfg_file);
        fcos = new FCOS(fcos_cfg);
    }
    if (task["yolo"] && task["yolo"].as<bool>()){
        string cfg_file = main_cfg["yolo"]["cfg_file"].as<string>();
        yolo_cfg =  YAML::LoadFile(cfg_file);
        yolo = new YOLOV5(yolo_cfg);
    }


/* -==================Run tasks=================*/
    int count = main_cfg["misc"]["runtimes"].as<int>();
    // multithreading
    // NOTE: not complement yet.
    if(main_cfg["misc"]["multithreading"].as<bool>())
    {
        for (size_t i = 0; i < count; i++)
        {
            auto thread_func_0 = [&](){
                cv::Mat frame = imread(fairmot_cfg["inputs"]["img_path"].as<string>());
                int im_w = fairmot_cfg["inputs"]["width"].as<int>();
                int im_h = fairmot_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = fairmot_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto fairmot_results = fairmot->run(imgs);
            };
            auto thread_func_1 = [&](){
                cv::Mat frame = imread(fcos_cfg["inputs"]["img_path"].as<string>());
                int im_w = fcos_cfg["inputs"]["width"].as<int>();
                int im_h = fcos_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = fcos_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto fcos_results = fcos->run(imgs);
            };
            auto thread_func_2 = [&](){
                cv::Mat frame = imread(f_track_cfg["inputs"]["img_path"].as<string>());
                int im_w = f_track_cfg["inputs"]["width"].as<int>();
                int im_h = f_track_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = f_track_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto f_track_results = f_track->run(imgs);
            };
            thread thread_ctx_0(thread_func_0);
            thread thread_ctx_1(thread_func_1);
            thread thread_ctx_2(thread_func_2);
            thread_ctx_0.join();
            thread_ctx_1.join();
            thread_ctx_2.join();

        }
    }

    //singlethreading
    else
    {
/* -==================classification task================*/
        if (cls){
            for (size_t i = 0; i < count; i++)
            {
                cv::Mat frame = imread(cls_cfg["inputs"]["img_path"].as<string>());
                int im_w = cls_cfg["inputs"]["width"].as<int>();
                int im_h = cls_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = cls_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto cls_results = cls->run(imgs);
            }
            delete cls;
            cls = nullptr;
        }
/* -==================segmentation task================*/
        if (semseg){
            for (size_t i = 0; i < count; i++)
            {
                cv::Mat frame = imread(semseg_cfg["inputs"]["img_path"].as<string>());
                int im_w = semseg_cfg["inputs"]["width"].as<int>();
                int im_h = semseg_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = semseg_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto semseg_results = semseg->run(imgs);
            }
            delete semseg;
            semseg = nullptr;
        }
/* -==================track task================*/
        // fairmot
        if (fairmot){
            for (size_t i = 0; i < count; i++)
            {
                cv::Mat frame = imread(fairmot_cfg["inputs"]["img_path"].as<string>());
                int im_w = fairmot_cfg["inputs"]["width"].as<int>();
                int im_h = fairmot_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = fairmot_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto fairmot_results = fairmot->run(imgs);
            }
            delete fairmot;
            fairmot = nullptr;
        }    
        // f-track
        if (f_track){
            for (size_t i = 0; i < count; i++)
            {
                cv::Mat frame = imread(f_track_cfg["inputs"]["img_path"].as<string>());
                int im_w = f_track_cfg["inputs"]["width"].as<int>();
                int im_h = f_track_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = f_track_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                // auto start = chrono::system_clock::now();
                auto f_track_results = f_track->run(imgs);
                // auto end = chrono::system_clock::now();
                // auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                // cout << "Infer Timer : " << duration.count() << "ms" << endl;
            }
            delete f_track;
            f_track = nullptr;
        }
/* -==================detection task================*/
        // fcos
        if (fcos){
            for (size_t i = 0; i < count; i++)
            {
                cv::Mat frame = imread(fcos_cfg["inputs"]["img_path"].as<string>());
                int im_w = fcos_cfg["inputs"]["width"].as<int>();
                int im_h = fcos_cfg["inputs"]["height"].as<int>();
                cv::resize(frame, frame, cv::Size(im_w, im_h));
                int batch_size = fcos_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                // auto start = chrono::system_clock::now();
                auto fcos_results = fcos->run(imgs);
                // auto end = chrono::system_clock::now();
                // auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                // cout << "Infer Timer : " << duration.count() << "ms" << endl;
                // vis_detection(fcos_results, imgs);
                // cv::imwrite("../data/fcos_results.jpg", imgs[0]);
            }
            delete fcos;
            fcos = nullptr;
        }
        // yolo
        if (yolo){
            for (size_t i = 0; i < count; i++)
            {
                cv::Mat frame = imread(yolo_cfg["inputs"]["img_path"].as<string>());
                int im_w = yolo_cfg["inputs"]["width"].as<int>();
                int im_h = yolo_cfg["inputs"]["height"].as<int>();
                int batch_size = yolo_cfg["engine"]["bchw"].as<vector<int>>()[0];
                vector<cv::Mat> imgs;
                for(int i = 0; i < batch_size; i++){
                    imgs.emplace_back(frame);
                }
                auto yolo_results = yolo->run(imgs);
                vis_detection(yolo_results, imgs);
                cv::imwrite("../data/yolo_results.jpg", imgs[0]);
            }
            delete yolo;
            yolo = nullptr;
        }
    }
    cout << "DONE!\n";
    return 0;
}
