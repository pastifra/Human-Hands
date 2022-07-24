#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <string>

#include "detection.h"
#include "segmentation.h"
#include "IOfunctions.h"

using namespace std;
using namespace cv;
using namespace dnn;

int main(int argc, char** argv)
{
    Net yolo;
    String cfg_path = getPath("Please enter the path of the .cfg file ");
    
    String weights_path = getPath("Please enter the path of the .weights file ");
    
    //NET CONFIGURATION
    yolo = readNetFromDarknet(cfg_path, weights_path);
    vector<string> output_names = yolo.getUnconnectedOutLayersNames(); //size = 3
    
    String dataPath = getPath("Please enter the path of the directory containing the test images ");
    vector<string> everyImgPath;
    glob(dataPath + "/*.jpg", everyImgPath, false); //Get a list of all the images paths
    
    String folder_bboxes = getPath("Please enter the path of the directory containing the ground truth bounding boxes ");
    vector<string> filenames_bboxes;
    glob(folder_bboxes + "/*.txt", filenames_bboxes, false);
    
    String folder_gt = getPath("Please enter the path of the directory containing the ground truth masks ");
    vector<string> filenames_gt;
    glob(folder_gt, filenames_gt);
    
    int counter = 0; //used to know which image we're analyzing in the for cycle
    double sum_IoUs = 0.0; //used to calculate the global average IoU over all the images
    double sum_accuracies = 0.0; //used to calculate the global average pixel accuracy over all the images
    
    for (string& entry : everyImgPath) //For each image in the test directory
    {
        Mat image = imread(entry);
        Mat originalImg = image.clone();
        
        if (image.empty())
        {
            printf("Error opening image");
            return 0;
        }
        
        vector<Rect> gtBoxes;
        readNumbers(filenames_bboxes[counter], gtBoxes); //get the ground truth bounding boxes of the current image
        
        vector<Rect> outBoxes;
        vector<float> scores;//bounding boxes probability
        infer(yolo, outBoxes, output_names, image, scores); //get the bounding boxes computed by the net
        
        draw(outBoxes, image, scores);
        draw_gt(gtBoxes, image);
        displayImg("output bounding boxes", image);
        
        cout << "Image: " << counter + 1 << "\n" << endl;
        for(int i = 0; i < outBoxes.size(); i++){
            printBbox(outBoxes.at(i), "Computed bounding box:"); //print the computed bounding box
            printBbox(gtBoxes.at(i), "Ground truth bounding box:"); //print the ground truth bounding box
        }
        
        double avg_IoU = single_img_results(outBoxes, gtBoxes); //get the average intersection over union of the imahe
        cout << "Avg IoU: "<< avg_IoU << "\n" << endl;
        
        sum_IoUs += avg_IoU; //update the parameter
        
        Mat mask_gt = imread(filenames_gt[counter]); //ground truth mask
        Mat mask = Mat::zeros(image.rows,image.cols, CV_8UC1);//all black, it will contain the outputmask
        
        segmentImg(originalImg, outBoxes, mask);
        
        displayImg("segmented image", originalImg);
        
        double accuracy = pixelAccuracy(mask, mask_gt);
        cout<<"Pixel accuracy:"<< accuracy << "\n" << endl;
        
        sum_accuracies += accuracy; //update parameters
        counter++;
    }
    
    double glob_avg_IoU = sum_IoUs/counter; //average IoU over all the images tested
    cout << "Global average intersection over union: " << glob_avg_IoU << endl;
    
    double glob_avg_accuracy = sum_accuracies/counter;
    cout << "Global average accuracy: " << glob_avg_accuracy << endl;
    
    return 0;
}
