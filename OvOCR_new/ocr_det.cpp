#include "ocr_det.h"

namespace PaddleOCR {

	Detector::Detector(std::string model_path)
	{
		ov::Core core;
		this->model_path = model_path;
		this->model = core.read_model(this->model_path);
		std::cout << "load complete!" << std::endl;
		this->model->reshape({ 1, 3, ov::Dimension(32, this->limit_side_len_), ov::Dimension(1, this->limit_side_len_) });
		this->compiled_model = core.compile_model(this->model, "CPU");
		this->infer_request = this->compiled_model.create_infer_request();
	}

	void Detector::Run(cv::Mat& src_img, std::vector<OCRPredictResult>& ocr_results)
	{
		this->src_img = src_img;
		// todo
		// resize
		//normalize
		this->resize_op_.Run(this->src_img, this->resize_img, this->limit_type_, this->limit_side_len_, this->ratio_h, this->ratio_w);



	};

}
