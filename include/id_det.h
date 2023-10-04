#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "utility.h"
#include "preprocess_op.h"
#include "postprocess_op.h"
#include <openvino/openvino.hpp>

namespace PaddleOCR {
class IDCardDetector {
public:
	struct Box {
		float start_x, start_y, end_x, end_y = 0.f;
	};

	struct BboxInfo {
		float confidence = 0.f;
		float classID = 0.f;
		std::vector<float> mask;
		Box boundingbox;
	};

public:
	explicit IDCardDetector(std::string model_path);
	void Run(cv::Mat& src_img, cv::Mat& dst, int& class_id);
	std::vector<IDCardDetector::BboxInfo> NMS(std::vector<IDCardDetector::BboxInfo>& vec);
	Box xywh2xyxy(float x, float y, float w, float h);
	float CalculateIoU(const IDCardDetector::Box& obj0, const IDCardDetector::Box& obj1);
	bool IsOverlapped(const IDCardDetector::Box& obj0, const IDCardDetector::Box& obj1);
	void sigmoid(cv::Mat& mat);
	cv::Mat crop_mask(const cv::Mat& masks, const IDCardDetector::Box& box);
	void drawPrediction(cv::Mat& img, const std::vector<IDCardDetector::BboxInfo>& bbox, int thickness, bool hideConf);
	void drawSegmentationMask(cv::Mat& img, std::vector<cv::Mat>& masks, double alpha, double gamma);
	int clip(float value, const int lower, const int upper);
	std::vector<cv::Point> sort_corner_order(std::vector<cv::Point>& quadrangle);

	std::vector<cv::Mat> getProto(const float* data, const ov::Shape& shape);
	std::vector<IDCardDetector::BboxInfo> getPred(const float* data, const ov::Shape& shape);
	std::vector<cv::Mat> process_mask(std::vector<IDCardDetector::BboxInfo>& pred, const std::vector<cv::Mat>& proto, const ov::Shape& shape);
	std::vector<cv::Point> getKeypoint(std::vector<cv::Mat> masks_);
	cv::Mat align_idcard(cv::Mat src_img_, std::vector<cv::Point> keypoints_);

private:
	ov::InferRequest infer_request;
	std::string model_path;
	cv::Mat src_img;
	std::shared_ptr<ov::Model> model;
	ov::CompiledModel compiled_model;

	cv::Mat resize_img;
	float ratio;
	int resize_side_len = 640;
	//int proto_len = 160;

	float conf_thres = 0.85;
	float iou_thres = 0.5;

	ResizeIdcard resize_op_;
	Permute permute_op_;
};
}