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

#pragma once

//#include "ocr_cls.h"
#include "ocr_det.h"
#include "ocr_rec.h"
#include "id_det.h"

namespace PaddleOCR {

class PPOCR {
public:
	explicit PPOCR();
	~PPOCR();

	std::vector<OCRPredictResult> ocr(cv::Mat& img, cv::Mat& dst, int& class_id);

protected:
	IDCardDetector* id_detector_ = nullptr;
	Detector* ocr_detector_ = nullptr;
	//Classifier *classifier_ = nullptr;
	Recognizer* ocr_recognizer_ = nullptr;
};

} // namespace PaddleOCR
