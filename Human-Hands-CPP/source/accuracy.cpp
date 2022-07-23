#include "../include/functions.h"

using namespace std;
using namespace cv;

double pixelAccuracy(Mat& mask, Mat& mask_got){
    double TP, TN, FP, FN;
    for(int i = 0; i < mask.rows && mask_got.rows; i++){//got
        for(int j = 0; j < mask.cols && mask_got.cols; j++){
           if(mask.at<Vec3b>(i,j)[0] == 255 && mask.at<Vec3b>(i,j)[1] == 255 && mask.at<Vec3b>(i,j)[3] == 255){
                if(mask_got.at<Vec3b>(i,j)[0] == 255 && mask_got.at<Vec3b>(i,j)[1] == 255 && mask_got.at<Vec3b>(i,j)[3] == 255){//correctly classified as hands
                    TP++;
                }
                else{//incorrectly classified as hands
                    FP++;
                }
            }else if(mask_got.at<Vec3b>(i,j)[0] == 255 && mask_got.at<Vec3b>(i,j)[1] == 255 && mask_got.at<Vec3b>(i,j)[3] == 255){//incorrectly classified as not hands
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

