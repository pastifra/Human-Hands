#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include "classes.h"

using namespace std;
using namespace cv;
using namespace dnn;

int main(int argc, char** argv)
{
    Net yolo;
    string cfg_path = "/home/local/pastfra10151/Desktop/HumanHands/Model/yolov4-hands.cfg";
    string weights_path = "/home/local/pastfra10151/Desktop/HumanHands/Model/yolov4-hands.weights";
    
    //NET CONFIGURATION
    yolo = readNetFromDarknet(cfg_path, weights_path);
    vector<string> output_names = yolo.getUnconnectedOutLayersNames(); //size = 3
    
    //IMAGE READING
    Mat image = imread("/home/local/pastfra10151/Desktop/HumanHands/Data/25.jpg");
    
    vector<Rect> outBoxes = inferAndDisplay(yolo, output_names, image);
    
    return 0;
}