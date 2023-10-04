#include "id_det.h"
#include "opencv2/opencv.hpp"
#include <cassert>

#define DEBUG_

namespace PaddleOCR {
	IDCardDetector::IDCardDetector(std::string model_path) {
		ov::Core core;
		this->model_path = model_path;
		this->model = core.read_model(this->model_path);
		this->compiled_model = core.compile_model(this->model, "CPU");
		this->infer_request = this->compiled_model.create_infer_request();
	};

	void IDCardDetector::Run(cv::Mat& src_img, cv::Mat& dst, int& class_id) {
		this->src_img = src_img;
		this->resize_op_.Run(this->src_img, this->resize_img, this->resize_side_len, this->ratio);

		double e = 1.0 / 255.0;
		(this->resize_img).convertTo(this->resize_img, CV_32FC3, e);
			
		std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
		ov::Shape intput_shape = { 1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols) };

		int h = resize_img.rows;
		int w = resize_img.cols;
		int c = resize_img.channels();

		this->permute_op_.Run(&resize_img, input.data());

		auto input_port = this->compiled_model.input();

		// -------- set input --------
		ov::Tensor input_tensor(input_port.get_element_type(), intput_shape, input.data());
		this->infer_request.set_input_tensor(input_tensor);

		// -------- start inference --------
		this->infer_request.infer();

		auto output_pred = this->infer_request.get_tensor("output0");
		auto output_proto = this->infer_request.get_tensor("output1");
		const float* pred_data = output_pred.data<const float>();
		const float* proto_data = output_proto.data<const float>();

		ov::Shape pred_shape = output_pred.get_shape();     // 1,25200,38
		ov::Shape proto_shape = output_proto.get_shape();  // 1,32,160,160
		
		// -------- start PostProcess --------
		std::vector<cv::Mat> proto = getProto(proto_data, proto_shape);
		std::vector<IDCardDetector::BboxInfo> pred = getPred(pred_data, pred_shape);
		std::vector<cv::Mat> masks = process_mask(pred, proto, proto_shape);


		// detection & segmentation 결과 출력
		cv::Mat img_result = src_img.clone();

		drawPrediction(img_result, pred, 2, false);
		cv::imwrite("img_detect_result.jpg", img_result);
		drawSegmentationMask(img_result, masks, 0.4, 50);
		cv::imwrite("img_seg_result.jpg", img_result);

		// align
		std::vector<cv::Point> keypoints = getKeypoint(masks);

		cv::Mat aligned_img = cv::Mat::zeros(cv::Size(125,80), CV_8UC3);
		if (keypoints.size()) {
			aligned_img = align_idcard(src_img, keypoints);
			cv::imwrite("aligned_img.jpg", aligned_img);

			cv::circle(img_result, keypoints[0], 20, (255, 0, 0), -1, cv::LINE_AA);
			cv::circle(img_result, keypoints[1], 20, (255, 255, 0), -1, cv::LINE_AA);
			cv::circle(img_result, keypoints[2], 20, (0, 0, 255), -1, cv::LINE_AA);
			cv::circle(img_result, keypoints[3], 20, (0, 255, 255), -1, cv::LINE_AA);

			cv::imwrite("img_key_result.jpg", img_result);

		}

		// 결과
		dst = aligned_img.clone();
	};

	/// 함수들 ///

	std::vector<cv::Mat> IDCardDetector::getProto(const float* data, const ov::Shape& shape) {
		//prototypes 벡터로 정리//
		const size_t n = shape[1];
		const size_t h = shape[2];
		const size_t w = shape[3];
		const int size = h * w;

		// prototypes //

		std::vector<float> proto;
		std::vector<cv::Mat> proto_mat;

		for (int i = 0; i < n; i++) {

			for (int j = size * i; j < size * (1 + i); j++) {
				proto.push_back(data[j]);
			}

			cv::Mat mat(h, w, CV_32FC1, proto.data());
			proto_mat.push_back(mat.clone());

			proto.clear();
		}

		return proto_mat;
	}

	std::vector<IDCardDetector::BboxInfo> IDCardDetector::getPred(const float* data, const ov::Shape& shape) {
		// prediction head //
		const size_t candidate_size = shape[1];
		const size_t info_size = shape[2];

		std::vector<float> pred;
		std::vector<std::vector<float>> pred_list;
		std::vector<IDCardDetector::BboxInfo> Infos;

		for (int i = 0; i < candidate_size; i++) {
			for (int j = info_size * i; j < info_size * (i + 1); j++) {
				pred.push_back(data[j]);
			}
			pred_list.push_back(pred);
			pred.clear();
		}

		for (int i = 0; i < pred_list.size(); i++) {
			IDCardDetector::BboxInfo info;

			//confidence 값이 threshold 보다 낮으면 삭제
			if (pred_list[i][4] < conf_thres)
				continue;

			info.confidence = pred_list[i][4];
			info.classID = pred_list[i][5];
			info.boundingbox = xywh2xyxy(pred_list[i][0], pred_list[i][1], pred_list[i][2], pred_list[i][3]);

			for (int j = 6; j < pred_list[i].size(); j++) {
				info.mask.push_back(pred_list[i][j]);
			}
			Infos.push_back(info);
		}

		std::vector<IDCardDetector::BboxInfo> nms_result;
		nms_result = NMS(Infos);

		for (int i = 0; i < nms_result.size(); i++) {
			int mask_size = nms_result[i].mask.size();

			for (int j = 0; j < mask_size; j++) {
				nms_result[i].mask[j] *= nms_result[i].confidence;
			}
		}

		return nms_result;
	}

	std::vector<cv::Mat> IDCardDetector::process_mask(std::vector<IDCardDetector::BboxInfo>& pred, const std::vector<cv::Mat>& proto, const ov::Shape& shape) {
		//process_mask (각 proto 에 32개의 마스크 값을 각각 곱함, 행렬곱 (candidate,32)*(32,160,160))
		const float proto_h = shape[2];
		const float proto_w = shape[3];
		
		
		std::vector<cv::Mat> masks;

		for (int i = 0; i < pred.size(); i++) {
			cv::Mat mask = cv::Mat::zeros(proto_h, proto_w, CV_32FC1);

			for (int j = 0; j < proto.size(); j++) {
				cv::Mat temp = proto[j] * pred[i].mask[j];
				cv::add(mask, temp, mask);
			}
			
			sigmoid(mask);

			IDCardDetector::Box downsampled_bbox;
			downsampled_bbox.start_x = pred[i].boundingbox.start_x * (proto_w / float(resize_side_len));
			downsampled_bbox.start_y = pred[i].boundingbox.start_y * (proto_h / float(resize_side_len));
			downsampled_bbox.end_x = pred[i].boundingbox.end_x * (proto_w / float(resize_side_len));
			downsampled_bbox.end_y = pred[i].boundingbox.end_y * (proto_h / float(resize_side_len));

			cv::Mat crop_mask_ = crop_mask(mask, downsampled_bbox);

			cv::resize(crop_mask_, crop_mask_, cv::Size(resize_side_len, resize_side_len), 0, 0, cv::INTER_LINEAR);

			// 이진화
			cv::threshold(crop_mask_, crop_mask_, 0.5, 1, cv::THRESH_BINARY);

			//원본사이즈로 resize
			float ratio = this->ratio;
			cv::resize(crop_mask_, crop_mask_, cv::Size(), 1 / ratio, 1 / ratio);
			// border 추가 했던 부분 자르기
			cv::Rect raw(0, 0, src_img.cols, src_img.rows);
			cv::Mat crop_mask_raw = crop_mask_(raw);
			crop_mask_raw.convertTo(crop_mask_raw, CV_8UC1);

			// resize bbox
			pred[i].boundingbox.start_x = static_cast<int>(pred[i].boundingbox.start_x / ratio);
			pred[i].boundingbox.start_y = static_cast<int>(pred[i].boundingbox.start_y / ratio);
			pred[i].boundingbox.end_x = static_cast<int>(pred[i].boundingbox.end_x / ratio);
			pred[i].boundingbox.end_y = static_cast<int>(pred[i].boundingbox.end_y / ratio);

			pred[i].boundingbox.start_x = clip(pred[i].boundingbox.start_x, 0, src_img.cols);
			pred[i].boundingbox.start_y = clip(pred[i].boundingbox.start_y, 0, src_img.rows);
			pred[i].boundingbox.end_x = clip(pred[i].boundingbox.end_x, 0, src_img.cols);
			pred[i].boundingbox.end_y = clip(pred[i].boundingbox.end_y, 0, src_img.rows);


			masks.push_back(crop_mask_raw.clone());
		}

		return masks;
	}

	std::vector<cv::Point> IDCardDetector::getKeypoint(std::vector<cv::Mat> masks_) {
		const int morph_ksize = 21;
		const float contour_thres = 0.02;
		const float poly_thres = 0.03;

		
		cv::Mat keypoints_image;
		cv::morphologyEx(masks_[0], keypoints_image, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_ksize, morph_ksize)));

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(keypoints_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);

		std::vector<std::vector<cv::Point>> filteredContours;

		for (const std::vector<cv::Point>& contour : contours) {
			if (cv::contourArea(contour) > contour_thres) {
				filteredContours.push_back(contour);
			}
		}

		std::vector<std::vector<cv::Point>> approximatedQuadrangles;
		for (const std::vector<cv::Point>& contour : filteredContours) {
			std::vector<cv::Point> approx;
			cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * poly_thres, true);
			if (approx.size() == 4) { // Check if it's a quadrangle (4 vertices)
				approximatedQuadrangles.push_back(approx);
			}
		}
		
		std::vector<cv::Point> keypoints;
		if (approximatedQuadrangles.size() == 1) {
			keypoints = sort_corner_order(approximatedQuadrangles[0]);
		}

		return keypoints;
		
	}

	cv::Mat IDCardDetector::align_idcard(cv::Mat src_img_, std::vector<cv::Point> keypoints_) {
		int idcard_ratio_w = 125;
		int idcard_ratio_h = 88;

		// Calculate contour area
		cv::Mat keypointsMat(keypoints_);
		double contourArea = cv::contourArea(keypointsMat);

		// Calculate the square root
		double sqrtContourArea = std::sqrt(contourArea);

		// Replace idcard_ratio with the actual value

		// Calculate dsize_factor
		int dsize_factor = cvRound(sqrtContourArea / idcard_ratio_w);

		int idcard_ratio_w_ = idcard_ratio_w * dsize_factor;
		int idcard_ratio_h_ = idcard_ratio_h * dsize_factor;

		cv::Point2f srcPt[4], dstPt[4];

		srcPt[0] = keypoints_[0];
		srcPt[1] = keypoints_[1];
		srcPt[2] = keypoints_[2];
		srcPt[3] = keypoints_[3];

		dstPt[0] = cv::Point2f(0, 0);
		dstPt[1] = cv::Point2f(0, idcard_ratio_h_);
		dstPt[2] = cv::Point2f(idcard_ratio_w_, idcard_ratio_h_);
		dstPt[3] = cv::Point2f(idcard_ratio_w_, 0);

		cv::Mat M = cv::getPerspectiveTransform(srcPt, dstPt);
		cv::warpPerspective(src_img_, src_img_, M, cv::Size(idcard_ratio_w_, idcard_ratio_h_));

		return src_img_;
	}

	std::vector<IDCardDetector::BboxInfo> IDCardDetector::NMS(std::vector<IDCardDetector::BboxInfo>& vec) {
		std::vector<IDCardDetector::BboxInfo> result;

		std::sort(vec.begin(), vec.end(), [](IDCardDetector::BboxInfo& lhs, IDCardDetector::BboxInfo& rhs) {
			if (lhs.confidence > rhs.confidence)
				return true;
			return false;
			});

		//iou_thres = 0.5
		for (size_t i = 0; i < vec.size(); ++i) {
			auto& item = vec[i];
			result.push_back(item);
			for (size_t j = i + 1; j < vec.size(); ++j) {
				float iou = CalculateIoU(item.boundingbox, vec[j].boundingbox);
				if (iou > iou_thres) {
					vec.erase(vec.begin() + j);
					--j;
				}
			}
		}

		return result;

	}

	IDCardDetector::Box IDCardDetector::xywh2xyxy(float x, float y, float w, float h) {
		
		IDCardDetector::Box box;

		box.start_x = x - w/2;
		box.start_y = y - h/2;
		box.end_x = x + w/2;
		box.end_y = y + h/2;

		return box;
	};

	float IDCardDetector::CalculateIoU(const IDCardDetector::Box& obj0, const IDCardDetector::Box& obj1) {
		int obj0_area = (obj0.end_x - obj0.start_x) * (obj0.end_y - obj0.start_y);
		int obj1_area = (obj1.end_x - obj1.start_x) * (obj1.end_y - obj1.start_y);
		int interx0 = (std::max)(obj0.start_x, obj1.start_x);
		int intery0 = (std::max)(obj0.start_y, obj1.start_y);
		int interx1 = (std::min)(obj0.end_x, obj1.end_x);
		int intery1 = (std::min)(obj0.end_y, obj1.end_y);
		if (interx1 < interx0 || intery1 < intery0)
			return 0.f;

		// FIXME - overlapped bbox
		/* IOU가 0.4 이하이지만 겹쳐져 있는 경우.

		 +-------------+
		 |             |
		 |             |
		 | +---+       |
		 | |   |       |
		 | +---+       |
		 +-------------+

		 */
		if (IsOverlapped(obj0, obj1))
			return 1.f;

		int areaInter = (interx1 - interx0) * (intery1 - intery0);
		int areaSum = obj0_area + obj1_area - areaInter;

		return static_cast<float>(areaInter) / areaSum;
	}

	bool IDCardDetector::IsOverlapped(const IDCardDetector::Box& obj0, const IDCardDetector::Box& obj1)
	{
		return (
			obj0.start_x <= obj1.start_x &&
			obj0.start_y <= obj1.start_y &&
			obj0.end_x >= obj1.end_x &&
			obj0.end_y >= obj1.end_y);
	}

	void IDCardDetector::sigmoid(cv::Mat& mat) {
		for (int i = 0; i < mat.rows; ++i) {
			for (int j = 0; j < mat.cols; ++j) {
				mat.at<float>(i, j) = 1.0 / (1.0 + exp(-mat.at<float>(i, j)));
			}
		}
	}

	cv::Mat IDCardDetector::crop_mask(const cv::Mat& masks, const IDCardDetector::Box& box) {
		
		int h = masks.size().height;
		int w = masks.size().width;

		float x1 = box.start_x;
		float y1 = box.start_y;
		float x2 = box.end_x;
		float y2 = box.end_y;

		cv::Mat result = cv::Mat::zeros(masks.size(), CV_32F);

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (x >= x1 && x < x2 && y >= y1 && y < y2) {
					result.at<float>(y, x) = masks.at<float>(y, x);
				}
			}
		}

		return result;
	}

	void IDCardDetector::drawPrediction(cv::Mat& img, const std::vector<IDCardDetector::BboxInfo>& bbox, int thickness, bool hideConf) {
		assert(img.channels() == 3 && img.depth() == CV_8U && img.type() == CV_8UC3);

		cv::Scalar bboxColor(0, 255, 0);
		cv::Scalar confColor(0, 255, 0);

		for (size_t i = 0; i < bbox.size(); ++i) {
			// Draw bbox
			cv::rectangle(img, cv::Point(bbox[i].boundingbox.start_x, bbox[i].boundingbox.start_y), cv::Point(bbox[i].boundingbox.end_x, bbox[i].boundingbox.end_y), bboxColor, thickness, cv::LINE_AA);

			// Text confidence
			if (!hideConf) {
				cv::putText(img, cv::format("%0.2f", bbox[i].confidence), cv::Point(bbox[i].boundingbox.start_x, bbox[i].boundingbox.start_y - 2), cv::FONT_HERSHEY_SIMPLEX, 3, confColor, thickness, cv::LINE_AA);
			}
		}
	}

	void IDCardDetector::drawSegmentationMask(cv::Mat& img, std::vector<cv::Mat>& masks, double alpha = 0.4, double gamma = 50) {
		
		// mask 전부 합치기
		cv::Mat mask = cv::Mat::zeros(masks[0].size(), CV_8UC1);

		for (int i = 0; i < masks.size(); i++) {
			cv::add(mask, masks[i], mask);
		}

		cv::Mat coloredMask(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		
		std::vector<cv::Mat> channels;
		cv::split(coloredMask, channels);

		cv::Mat blueChannel = channels[0];
		cv::Mat greenChannel = channels[1];
		cv::Mat redChannel = channels[2];

		int colors[3] = { 56, 56, 255 };

		for (int j = 0; j < mask.rows; ++j) {
			for (int k = 0; k < mask.cols; ++k) {
				if (mask.at<uchar>(j, k) == 1) {
					blueChannel.at<uchar>(j, k) = colors[2];
					greenChannel.at<uchar>(j, k) = colors[1];
					redChannel.at<uchar>(j, k) = colors[0];

				}
			}
		}

		cv::Mat mergedImage;
		cv::merge(channels, mergedImage);

		cv::Mat result;

		cv::addWeighted(img, alpha, mergedImage, 1 - alpha, gamma, img);
	}

	int IDCardDetector::clip(float value, const int lower, const int upper) {
		return (value < lower) ? lower : (value > upper) ? upper : value;
	}

	std::vector<cv::Point> IDCardDetector::sort_corner_order(std::vector<cv::Point>& quadrangle) {
		cv::Mat quadranglePoints(4, 2, CV_32S);

		for (int i = 0; i < 4; i++) {
			quadranglePoints.at<int>(i, 0) = quadrangle[i].x;
			quadranglePoints.at<int>(i, 1) = quadrangle[i].y;
		}

		cv::Moments moments = cv::moments(quadranglePoints);
		int mcx = cvRound(moments.m10 / moments.m00);  // mass center x
		int mcy = cvRound(moments.m01 / moments.m00);  // mass center y

		std::vector<cv::Point> sortedQuadrangleVector(4);

		for (int i = 0; i < 4; i++) {
			const cv::Point& point = quadrangle[i];
			if (point.x < mcx && point.y < mcy) {
				sortedQuadrangleVector[0] = point;
			}
			else if (point.x < mcx && point.y > mcy) {
				sortedQuadrangleVector[1] = point;
			}
			else if (point.x > mcx && point.y > mcy) {
				sortedQuadrangleVector[2] = point;
			}
			else if (point.x > mcx && point.y < mcy) {
				sortedQuadrangleVector[3] = point;
			}
		}

		return sortedQuadrangleVector;
	};
}