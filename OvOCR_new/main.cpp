#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include "paddleocr.h"

#include <openvino/openvino.hpp>

using namespace PaddleOCR;

int main() {

	cv::Mat src_img = cv::imread("../sample_idcard.png");

	PPOCR ppocr;
	std::vector<OCRPredictResult> ocr_result = ppocr.ocr(src_img);
	



	return 0;
}