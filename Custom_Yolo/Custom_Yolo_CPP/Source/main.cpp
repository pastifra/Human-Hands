#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <filesystem>
#include <iostream>
#include "classes.h"

using namespace std;
using namespace cv;
using namespace dnn;


int main(int argc, char** argv)
{
    Net yolo = readNetFromTensorflow("/home/local/pastfra10151/Desktop/Visionari/Model/frozen_graph.pb");
    
    String dataPath;
    cout << "Please enter the path of the directory containing the test images ";
    cin >> dataPath;
    
    vector<string> everyImgPath;
    glob(dataPath + "/*.jpg", everyImgPath, false); //Get a list of all the images paths
    
    for (string& entry : everyImgPath) //For each image in the test directory
    {
        Mat show = imread(entry);
        Mat frame;
        
        //PRE-PROCESS input image
        cvtColor(show,frame,COLOR_BGR2RGB);
        resize(frame, frame, Size(448,448), INTER_LINEAR);
        resize(show, show, Size(448,448), INTER_LINEAR);
        frame.convertTo(frame, CV_32F);
        frame = frame/255.0;
        Mat in_net = blobFromImage(frame);
        
        //INFER in the network
        yolo.setInput(in_net);
        Mat out = yolo.forward(); //1x5x7x7
        vector<Mat> outs; //1*(7x7x5)
        imagesFromBlob(out,outs);
        
        //DECODE the output and show the image
        decodeNetOut(outs[0], show);
    }

    return 0;
}
