#ifndef SEGM
#define SEGM

#include <opencv2/core/types.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/imgproc.hpp>

void segmentation_rect(cv::Mat& img, cv::Rect& r, cv::Mat& mask1);
void segmentImg(cv::Mat& originalImg, std::vector<cv::Rect>& outBoxes, cv::Mat& mask);
double pixelAccuracy(cv::Mat& mask, cv::Mat& mask_gt);

#endif //SEGM
