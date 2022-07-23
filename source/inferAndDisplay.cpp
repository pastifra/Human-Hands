#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "classes.h"

using namespace std;
using namespace cv;
using namespace dnn;

std::vector<cv::Rect> inferAndDisplay(cv::dnn::Net yolo, std::vector<string> outNames, cv::Mat image)
{
    /* Creation of a blob of dimension 1x3x416x416 with values scaled in range [0,1] and swapping 
     * the channels from OpenCV BGR to RGB as the network train format*/
    Mat blob;
    blobFromImage(image, blob, 1.0/255.0, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
    
    /* FORWARD PASS of the network where the image is inferred, three different outputs are
     * obtained at three different "head" layers listed in outNames
     * Hence detections has size = 3 */
    vector<Mat> detections;
    yolo.setInput(blob);
    yolo.forward(detections, outNames);
    
    
    vector<int> indices; // For indices of boxes after non maxima suppresion
    vector<Rect> boxes;  // For all the predicted boxes
    vector<float> scores;// For all the corresponding boxes probability
    
    
    /* DECODE NETWORK OUTPUT 
     * Yolo preforms detections at three separate resolutions:
     * 13x13, 26x26 and 52x52.
     * The detections are hence three feature maps at three different resolutions
     * that are mapped into predicted bounding boxes by special layers called "head" */
    
    for (Mat& output : detections) // For each unconnected output layer
    {   
        // Each box has 6 entries [ox = center,oy = center,width,height, P = confidence , C = class] 
        // First output layer has 8112 bboxes => detections size = 8112x6
        // Second output layer has 2028 bboxes => detections size = 2028x6
        // Third output layer has 507 bboxes => detections size = 507x6
        
        const int num_boxes = output.rows;
        
        for (int i = 0; i < num_boxes; i++)
        {
            /* Get center,width and height of the bounding box
             * and scaled them up to the original image size */
             
            int x = output.at<float>(i, 0) * image.cols;
            int y = output.at<float>(i, 1) * image.rows;
            int width = output.at<float>(i, 2) * image.cols;
            int height = output.at<float>(i, 3) * image.rows;
            
            Rect rect(x - width/2, y - height/2, width, height); //Actual bounding box!
            
            float confidence = output.at<float>(i, 5);
            
            if (confidence >= 0.5)
            {
                boxes.push_back(rect);
                scores.push_back(confidence);
            }
        }
    }
    
    /* Perform non maxima suppression of overlapping boxes and save the 
     * best one indices from the vector boxes in the vector indices */
    NMSBoxes(boxes, scores, 0.0, 0.4, indices);
    
    vector<Rect> outBoxes;
        
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        const Rect& rect = boxes[idx];
        outBoxes.push_back(rect);
        
        cv::rectangle(image, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), (0,0,255), 3);

        ostringstream label_stream;
        label_stream << "Hand : " << std::fixed << std::setprecision(2) << scores[idx];
        string label = label_stream.str();
                
        int baseline;
        Size label_size = getTextSize(label.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        rectangle(image, cv::Point(rect.x, rect.y - label_size.height - baseline - 10), cv::Point(rect.x + label_size.width, rect.y), (0,0,255), FILLED);
        putText(image, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
    }
    
    namedWindow("output");
    imshow("output", image);
    waitKey(0);
    
    return outBoxes;
}