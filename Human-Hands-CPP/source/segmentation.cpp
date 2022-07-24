
#include "../include/segmentation.h"

using namespace cv;
using namespace std;

void segmentation_rect(Mat& img, Rect& r, Mat& mask1){
    Mat bgdModel = Mat::zeros(1, 65, CV_64F);
    Mat fgdModel = Mat::zeros(1, 65, CV_64F);
    grabCut(img, mask1, r, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);//segmentation based on rect
} 

void segmentImg(Mat& originalImg, vector<Rect>& outBoxes, Mat& mask){
    for(int i = 0; i < outBoxes.size(); i++){ //for all bounding boxes in the image
        int margin = 5; //margin set to enlarge the bounding box in order to segment all the hand even if the bounding box is not precise
        
        Rect r = Rect(outBoxes.at(i).x - margin, outBoxes.at(i).y - margin, outBoxes.at(i).width + 2 * margin, outBoxes.at(i).height + 2 * margin);
        
        Mat mask1 = Mat::zeros(originalImg.rows, originalImg.cols, CV_8UC1);//all black, it will contain the mask of the single hand

        segmentation_rect(originalImg, r, mask1); //given the bounding box, segment the hand inside it
                  
        mask1 = (mask1==1)+(mask1==3); //color in white only the pixels labelled as sure foreground or probable forground
        mask = mask + mask1; //add to the output mask the pixels of the segmented hand
        
        Scalar random = Scalar(rand()%255,rand()%255,rand()%255); //choose a random color
        originalImg.setTo(random, mask1); //color the hand on the image with the chosen color
    }
}
