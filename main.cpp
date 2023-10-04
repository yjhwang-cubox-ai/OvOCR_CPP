#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "args.h"
#include "paddleocr.h"
#include "paddlestructure.h"
#include <gflags/gflags.h>
#include <filesystem>

using namespace PaddleOCR;

#define MODE 0

void check_params()
{
  if (FLAGS_type == "ocr")
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty())
    {
      std::cout << "Need a path to detection and recogition model"
                   "[Usage] --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --rec_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << std::endl;
      exit(1);
    }
  }
  else if (FLAGS_type == "structure")
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty() || FLAGS_lay_model_dir.empty() || FLAGS_tab_model_dir.empty())
    {
      std::cout << "Need a path to detection, recogition, layout and table model"
                   "[Usage] --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --rec_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --lay_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --tab_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << std::endl;
      exit(1);
    }
  }
}

int main(int argc, char *argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  check_params();
  
#if MODE
  // read image
  cv::Mat src_img = imread("../data/test.jpg");
  cv::Mat aligned_img;
  int class_id;

  if (FLAGS_type == "ocr")
  {
      PPOCR ppocr;
      std::vector<OCRPredictResult> ocr_result = ppocr.ocr(src_img, aligned_img, class_id);

      Utility::print_result(ocr_result);
      Utility::VisualizeBboxes(aligned_img, ocr_result,
          "./ocr_result.jpg", class_id);
  }

#else
  // 디렉토리 내 모든 이미지 파일에 대해 추론
  std::string directoryPath = FLAGS_input;
  std::vector<std::string> imageFilePaths;

  for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
      if (entry.is_regular_file()) {
          std::string extension = entry.path().extension().string();
          if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp" || extension == ".gif") {
              imageFilePaths.push_back(entry.path().string());
          }
      }
  }

  PPOCR ppocr;

  int idx = 0;
  for (auto& img : imageFilePaths) {
      ++idx;

      cv::Mat src_img = imread(img);
      cv::Mat aligned_img;
      std::string dst_name = "result_" + std::to_string(idx) + ".jpg";
      int class_id;


      std::vector<OCRPredictResult> ocr_result = ppocr.ocr(src_img, aligned_img, class_id);

      Utility::print_result(ocr_result);
      Utility::VisualizeBboxes(aligned_img, ocr_result,
          dst_name, class_id);
  }


#endif // USING_PATH



}
