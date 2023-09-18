#include "id_det.h"
#include "opencv2/opencv.hpp"

namespace PaddleOCR {
	IdDetector::IdDetector(std::string model_path) {
		ov::Core core;
		this->model_path = model_path;
		this->model = core.read_model(this->model_path);
		this->compiled_model = core.compile_model(this->model, "CPU");
		this->infer_request = this->compiled_model.create_infer_request();
	};
	void IdDetector::Run(cv::Mat& scr_img) {
		this->src_img = scr_img;
		this->resize_op_.Run(this->src_img, this->resize_img, this->limit_side_len_, this->ratio);

		double e = 1.0 / 255.0;
		(this->resize_img).convertTo(this->resize_img, CV_32FC3, e);

		while(true){
			cv::imshow("test", resize_img);
			if (cv::waitKey() == 27)
				cv::destroyAllWindows();
				break;
		}
			
		std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
		ov::Shape intput_shape = { 1, 3, static_cast<size_t>(resize_img.rows), static_cast<size_t>(resize_img.cols) };

		this->permute_op_.Run(&resize_img, input.data());

		auto input_port = this->compiled_model.input();

		// -------- set input --------
		ov::Tensor input_tensor(input_port.get_element_type(), intput_shape, input.data());
		this->infer_request.set_input_tensor(input_tensor); \

			// -------- start inference --------
			this->infer_request.infer();

		auto output = this->infer_request.get_output_tensor(0);
		auto output2 = this->infer_request.get_output_tensor(1);
		const float* out_data = output.data<const float>();
		const float* out_data2 = output2.data<const float>();

		ov::Shape output_shape = output.get_shape();     // 1,25200,38
		ov::Shape output_shape2 = output2.get_shape();  // 1,32,160,160


		//prototypes 벡터로 정리//
		const size_t n1 = output_shape2[1];
		const size_t n2 = output_shape2[2];
		const size_t n3 = output_shape2[3];
		const int n = n2 * n3;

		std::vector<float> proto;
		std::vector<std::vector<float>> proto_list; //prototypes 32개 저장
		std::vector<cv::Mat> proto_mat;

		int proto_idx = 0;
		for (int i = 0; i < n1; i++) {

			for (int j = n * proto_idx; j < n * (1 + proto_idx); j++) {
				proto.push_back(out_data2[j]);
			}

			proto_list.push_back(proto);

			proto.clear();

			proto_idx++;
		}

		for (int i = 0; i < proto_list.size(); i++) {
			cv::Mat mat(160, 160, CV_32FC1, proto_list[i].data());
			proto_mat.push_back(mat);
		}


		/*for (int i = 0; i < n1; i++) {
			cv::imshow("test", proto_mat[i]);
			cv::waitKey(0);
		}*/


		std::cout << proto_list.size() << std::endl;

		// prediction head //
		const size_t h1 = output_shape[1];
		const size_t h2 = output_shape[2];

		std::vector<float> pred;
		std::vector<std::vector<float>> pred_list;
		std::vector<IdDetector::BboxInfo> bboxInfos;

		int head_idx = 0;
		for (int i = 0; i < h1; i++) {
			for (int j = h2 * head_idx; j < h2 * (1 + head_idx); j++) {
				pred.push_back(out_data[j]);
			}
			pred_list.push_back(pred);
			pred.clear();

			head_idx++;
		}		

		for (int i = 0; i < pred_list.size(); i++) {
			IdDetector::BboxInfo bboxInfo;
			
			//confidence 값이 threshold 보다 낮으면 삭제
			if (pred_list[i][4] < conf_thres)
				continue;

			bboxInfo.confidence = pred_list[i][4];
			bboxInfo.classID = pred_list[i][5];
			bboxInfo.boundingbox = xywh2xyxy(pred_list[i][0], pred_list[i][1], pred_list[i][2], pred_list[i][3]);
			

			for (int j = 6; j < pred_list[i].size(); j++) {
				bboxInfo.maskcoef.push_back(pred_list[i][j]);
			}
			bboxInfos.push_back(bboxInfo);
		}

		std::vector<IdDetector::BboxInfo> result;
		result = NMS(bboxInfos);

		cv::rectangle(resize_img, cv::Rect(cv::Point(result[0].boundingbox.start_x, result[0].boundingbox.start_y), cv::Point(result[0].boundingbox.end_x, result[0].boundingbox.end_y)), cv::Scalar(0, 0, 255), 1, 8, 0);
		//cv::rectangle(resize_img, cv::Rect(cv::Point(99, 87), cv::Point(329,232)), cv::Scalar(0, 0, 255), 1, 8, 0);
		while (true) {
			cv::imshow("test", resize_img);
			if (cv::waitKey() == 27)
				cv::destroyAllWindows();
				break;
		}

		std::cout << "test" << std::endl;

	};

	/// 함수 ///
	std::vector<IdDetector::BboxInfo> IdDetector::NMS(std::vector<IdDetector::BboxInfo>& vec) {
		std::vector<IdDetector::BboxInfo> result;

		std::sort(vec.begin(), vec.end(), [](IdDetector::BboxInfo& lhs, IdDetector::BboxInfo& rhs) {
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

	IdDetector::Box IdDetector::xywh2xyxy(float x, float y, float w, float h) {
		
		IdDetector::Box box;

		box.start_x = x - w/2;
		box.start_y = y - h/2;
		box.end_x = x + w/2;
		box.end_y = y + h/2;

		return box;
	};

	float IdDetector::CalculateIoU(const IdDetector::Box& obj0, const IdDetector::Box& obj1) {
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

	bool IdDetector::IsOverlapped(const IdDetector::Box& obj0, const IdDetector::Box& obj1)
	{
		return (
			obj0.start_x <= obj1.start_x &&
			obj0.start_y <= obj1.start_y &&
			obj0.end_x >= obj1.end_x &&
			obj0.end_y >= obj1.end_y);
	}
}



