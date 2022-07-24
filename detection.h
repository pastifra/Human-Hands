#ifndef DEC
#define DEC

#include <vector>
#include <opencv2/core/types.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

void infer(cv::dnn::Net yolo, std::vector<cv::Rect>& outBoxes, std::vector<std::string> outNames, cv::Mat& image, std::vector<float>& final_scores);
void draw(std::vector<cv::Rect>& outBoxes, cv::Mat& img, std::vector<float>& scores);
void draw_gt(std::vector<cv::Rect>& gtBoxes, cv::Mat& image);
double calc_iou(cv::Rect bBox, cv::Rect gt);
double single_img_results(std::vector<cv::Rect> prop_bboxes, std::vector<cv::Rect> true_bboxes);

#endif
