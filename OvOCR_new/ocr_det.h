#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <openvino/openvino.hpp>

#include "utility.h"
#include "preprocess_op.h"

namespace PaddleOCR {

class Detector
{
public:
    Detector(std::string model_path);
    void Run(cv::Mat& src_img, std::vector<OCRPredictResult>& ocr_results);

private:
    ov::InferRequest infer_request;
    std::string model_path;
    cv::Mat src_img;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
        
    float ratio_h{};
    float ratio_w{};
    cv::Mat resize_img;
    std::string limit_type_ = "max";
    int limit_side_len_ = 960;

    // pre-process
    ResizeImgType0 resize_op_;
    /*Normalize normalize_op_;
    Permute permute_op_;*/
};
}
