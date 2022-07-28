#ifndef IOFUNCTIONS
#define IOFUNCTIONS

#include <string>

std::string getPath(std::string indication);
void readNumbers(std::string filePath, std::vector<cv::Rect>& gtBoxes);
void displayImg(std::string caption, cv::Mat& image);
void printBbox(cv::Rect& r);
void printBboxes(std::vector<cv::Rect>& bBoxes, std::string description);

#endif //FUNCTIONSIO
