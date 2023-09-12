#pragma once

#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace PaddleOCR {

class ResizeImgType0 {
public:
	void Run(const cv::Mat& img, cv::Mat& resize_img,
		std::string limit_type, int limit_side_len, float& ratio_h,
		float& ratio_w);
};

}