#pragma once

#include <memory>
#include "ocr_det.h"
#include "utility.h"

namespace PaddleOCR{

class PPOCR {
public:
	PPOCR();
	std::vector<OCRPredictResult> ocr(cv::Mat img);

private:
	std::unique_ptr<Detector> detector_;
	//std::unique_ptr<Recognizer> recognizer_;
};

}