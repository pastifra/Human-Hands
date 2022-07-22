#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn;

int main( int argc, char** argv )
{
    Net yolo;
    string cfg_path = "/home/local/pastfra10151/Desktop/HumanHands/Model/yolov4-hands.cfg";
    string weights_path = "/home/local/pastfra10151/Desktop/HumanHands/Model/yolov4-hands.weights";
    
    //NET CONFIGURATION
    yolo = readNetFromDarknet(cfg_path, weights_path);
    auto output_names = yolo.getUnconnectedOutLayersNames();
    
    //IMAGE PREPROCESSING
    Mat frame = imread("/home/local/pastfra10151/Desktop/HumanHands/Data/10.jpg");
    Mat blob;
    
    /* Creation of a blob of dimension 1x3x416x416 with values scaled in range [0,1] and swapping 
       the channels from OpenCV BGR to RGB as the network train format*/
    blobFromImage(frame, blob, 1.0/255.0, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
    
    //FORWARD PASS
    vector<Mat> detections;
    yolo.setInput(blob);
    yolo.forward(detections, output_names);
    
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (auto& output : detections)
    {   
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width/2, y - height/2, width, height);

            auto confidence = *output.ptr<float>(i, 5);
            if (confidence >= 0)
            {
                boxes.push_back(rect);
                scores.push_back(confidence);
            }
        }
    }
    
    cv::dnn::NMSBoxes(boxes, scores, 0.0, 0.4, indices);
        
    for (size_t i = 0; i < indices.size(); ++i)
    {
        auto idx = indices[i];
        const auto& rect = boxes[idx];
        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), (0,0,255), 3);

        std::ostringstream label_ss;
        label_ss << "Hand : " << std::fixed << std::setprecision(2) << scores[idx];
        auto label = label_ss.str();
                
        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), (0,0,255), cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    

    cv::namedWindow("output");
    cv::imshow("output", frame);
    cv::waitKey(0);
    return 0;
}