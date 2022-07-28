#ifndef SEGM
#define SEGM

#include <opencv2/core/types.hpp>

void segmentation_rect(cv::Mat& img, cv::Rect& r, cv::Mat& mask1);
void segmentImg(cv::Mat& originalImg, std::vector<cv::Rect>& outBoxes, cv::Mat& mask);
void closing(cv::Mat& mask1, cv::Rect& r);
double pixelAccuracy(cv::Mat& mask, cv::Mat& mask_gt);

#endif //SEGM
