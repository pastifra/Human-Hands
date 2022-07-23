#ifndef DEC
#define DEC

std::vector<cv::Rect> inferAndDisplay(cv::dnn::Net yolo,std::vector<std::string> outNames, cv::Mat image);

#endif