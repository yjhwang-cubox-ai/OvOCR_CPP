// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "args.h"
#include "paddleocr.h"
#include "opencv2/opencv.hpp"

namespace PaddleOCR {

	PPOCR::PPOCR() {
		this->id_detector_ = new IDCardDetector("../model/yolov5n-seg-idcard_class.onnx");
		this->ocr_detector_ = new Detector(FLAGS_det_model_dir);
		this->ocr_recognizer_ = new Recognizer(FLAGS_rec_model_dir, FLAGS_label_dir);
	};

	std::vector<OCRPredictResult> PPOCR::ocr(cv::Mat& img, cv::Mat& dst, int& class_id) {
		std::vector<OCRPredictResult> ocr_result;
		cv::Mat IDcard;

		this->id_detector_->Run(img, IDcard, class_id);

		// detect the sentence in input image
		this->ocr_detector_->Run(IDcard, ocr_result);
		// crop image
		std::vector<cv::Mat> img_list;
		for (int j = 0; j < ocr_result.size(); j++) {
			cv::Mat crop_img;
			crop_img = Utility::GetRotateCropImage(IDcard, ocr_result[j].box);
			img_list.push_back(crop_img);
		}

		// recognize the words in sentence and print them
		this->ocr_recognizer_->Run(img_list, ocr_result);

		dst = IDcard.clone();

		return ocr_result;
	}

	PPOCR::~PPOCR() {
		if (this->id_detector_ != nullptr) {
			delete this->id_detector_;
		}
		if (this->ocr_detector_ != nullptr) {
			delete this->ocr_detector_;
		}
		if (this->ocr_recognizer_ != nullptr) {
			delete this->ocr_recognizer_;
		}
	}
} // namespace PaddleOCR
