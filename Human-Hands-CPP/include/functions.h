#ifndef FUNCTIONS
#define FUNCTIONS

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

void readNumbers(std::vector<int>& numbers, std::string filePath);

void improveBySkin(cv::Mat& img);
void segmentation_rect(cv::Mat& img, cv::Rect& r, cv::Mat& mask1);
void improveMask(cv::Rect& r, cv::Mat& mask1);
void segmentation_mask(cv::Mat& img, cv::Mat& mask1);
void calculate_avg(std::vector<double>& avg_intensity,cv::Mat& img,cv::Rect& r,cv::Mat& mask1);
void improveMask_avg(cv::Mat& img, cv::Rect& r, std::vector<double>& avg_intensity, cv::Mat& mask1);

double pixelAccuracy(cv::Mat& mask, cv::Mat& mask_got);

#endif //FUNCTIONS
