#include <opencv2/imgproc.hpp>
#include "../include/segmentation.h"

using namespace std;
using namespace cv;

double pixelAccuracy(Mat& mask, Mat& mask_gt){
    double TP = 0;
    double TN = 0;
    double FP = 0;
    double FN = 0;
    for(int i = 0; i < mask.rows; i++){
        for(int j = 0; j < mask.cols; j++){
           if(mask.at<Vec3b>(j,i)[0] == 255 && mask.at<Vec3b>(j,i)[1] == 255 && mask.at<Vec3b>(j,i)[3] == 255){
                if(mask_gt.at<Vec3b>(j,i)[0] == 255 && mask_gt.at<Vec3b>(j,i)[1] == 255 && mask_gt.at<Vec3b>(j,i)[3] == 255){//correctly classified as hands
                    TP++;
                }
                else{//incorrectly classified as hands
                    FP++;
                }
            }else if(mask_gt.at<Vec3b>(j,i)[0] == 255 && mask_gt.at<Vec3b>(j,i)[1] == 255 && mask_gt.at<Vec3b>(j,i)[3] == 255){//incorrectly classified as not hands
                    FN++;  
                }
                else{//correctly classified as not hands
                    TN++;
                }
         }
    }
        
    //calculate accuracy
    return(TP + TN)/(TP + TN + FP +FN);
    }
