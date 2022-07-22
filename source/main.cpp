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
    blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
    
    //FORWARD PASS
    vector<Mat> detections;
    yolo.setInput(blob);
    yolo.forward(detections, output_names);
    
    std::vector<int> indices[1];
    std::vector<cv::Rect> boxes[1];
    std::vector<float> scores[1];
    
    for (auto& output : detections)
    {   
        cout<<output.size;
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width/2, y - height/2, width, height);

            for (int c = 0; c < 1; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= 0)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }
    
    for (int c = 0; c < 1; c++)
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, 0.4, indices[c]);
        
    for (int c= 0; c < 1; c++)
    {
        for (size_t i = 0; i < indices[c].size(); ++i)
        {

            auto idx = indices[c][i];
            const auto& rect = boxes[c][idx];
            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), (0,0,255), 3);

            std::ostringstream label_ss;
            label_ss << "Hand : " << std::fixed << std::setprecision(2) << scores[c][idx];
            auto label = label_ss.str();
                
            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), (0,0,255), cv::FILLED);
            cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }
    }
    

    cv::namedWindow("output");
    cv::imshow("output", frame);
    cv::waitKey(0);
    return 0;
}