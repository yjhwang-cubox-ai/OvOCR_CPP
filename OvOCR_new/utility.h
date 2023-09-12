#pragma once

#include <iostream>
#include <vector>


namespace PaddleOCR {

struct OCRPredictResult {
	std::vector<std::vector<int>> box;
	std::string text;
	float score = -1.0;
	float cls_score;
	int cls_label = -1;
};
}