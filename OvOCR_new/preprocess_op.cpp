#include "preprocess_op.h"


namespace PaddleOCR {
void ResizeImgType0::Run(const cv::Mat& img, cv::Mat& resize_img, std::string limit_type, int limit_side_len, float& ratio_h, float& ratio_w) {
	int w = img.cols;
	int h = img.rows;
	float ratio = 1.f;

	if (limit_type == "min")
	{
		if (std::min(h, w) < limit_side_len) {
			if (h < w) ratio = float(limit_side_len) / float(h);
			else ratio = float(limit_side_len) / float(w);
		}
	}
	else if (limit_type == "max") 
	{
		if (std::max(h, w) > limit_side_len) {
			if (h > w) ratio = float(limit_side_len) / float(h);
			else ratio = float(limit_side_len) / float(w);
		}
	}

	int resize_h = int(float(h) * ratio);
	int resize_w = int(float(w) * ratio);

	resize_h = std::max(int(round(resize_h / 32) * 32), 32);
	resize_w = std::max(int(round(resize_w / 32) * 32), 32);

	cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

	ratio_h = float(resize_h) / float(h);
	ratio_w = float(resize_w) / float(w);
}
}
