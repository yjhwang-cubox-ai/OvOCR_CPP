#include "paddleocr.h"

namespace PaddleOCR {
    //추후 플래그로 변경
    std::string model_dir = "C:/Users/youngjun/work/OvOCR_new/OvOCR_new/model/korean_PP-OCRv4_rec_infer/inference.pdmodel";

PPOCR::PPOCR() {
    this->detector_ = std::make_unique<Detector>(model_dir);
};

std::vector<OCRPredictResult> PPOCR::ocr(cv::Mat img) {
    std::vector<OCRPredictResult> ocr_result;
    this->detector_->Run(img, ocr_result);

    return ocr_result;

};

} // namespace PaddleOCR
