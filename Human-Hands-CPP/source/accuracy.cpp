#include <opencv2/imgproc.hpp>
#include "../include/segmentation.h"

using namespace std;
using namespace cv;

double pixelAccuracy(Mat& mask, Mat& mask_gt){
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    for(int i = 0; i < mask.rows && mask_gt.rows; i++){
        for(int j = 0; j < mask.cols && mask_gt.cols; j++){
           if(mask.at<Vec3b>(i,j)[0] == 255 && mask.at<Vec3b>(i,j)[1] == 255 && mask.at<Vec3b>(i,j)[3] == 255){
                if(mask_gt.at<Vec3b>(i,j)[0] == 255 && mask_gt.at<Vec3b>(i,j)[1] == 255 && mask_gt.at<Vec3b>(i,j)[3] == 255){//correctly classified as hands
                    TP++;
                }
                else{//incorrectly classified as hands
                    FP++;
                }
            }else if(mask_gt.at<Vec3b>(i,j)[0] == 255 && mask_gt.at<Vec3b>(i,j)[1] == 255 && mask_gt.at<Vec3b>(i,j)[3] == 255){//incorrectly classified as not hands
                    FN++;  
                }
                else{//correctly classified as not hands
                    TN++;
                }
         }
    }
    
    double numerator = TP + TN;
    double denominator = TP +TN + FP + FN;
    //calculate accuracy
    double accuracy = numerator/denominator;
    return accuracy;
    }
