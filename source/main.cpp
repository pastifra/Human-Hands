#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <filesystem>
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
    
    
    String dataPath;
    cout << "Please enter the path of the directory containing the test images ";
    cin >> dataPath;
    
    vector<string> everyImgPath;
    glob(dataPath + "/*.jpg", fn, false); //Get a list of all the images paths
    
    for (string& entry : everyImgPath) //For each image in the test directory
    {
        Mat image = imread(entry);
        
        if (image.empty())
        {
            printf("Error opening image");
            return 0;
        }
        vector<Rect> outBoxes = inferAndDisplay(yolo, output_names, image);        
    }

    return 0;
}