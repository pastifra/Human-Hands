#include <opencv2/imgproc.hpp>
#include "../include/segmentation.h"

using namespace cv;
using namespace std;

void segmentation_rect(Mat& img, Rect& r, Mat& mask1){
    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1);
            
    Mat imgLaplacian;
    Mat sharp = img.clone(); // copy source image to another temporary one
    
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    img.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    
    Mat bgdModel = Mat::zeros(1, 65, CV_64F);
    Mat fgdModel = Mat::zeros(1, 65, CV_64F);
    grabCut(imgResult, mask1, r, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);//segmentation based on rect
} 

void closing(Mat& mask1, Rect& r){
    int morph_size = 2;
    // Create structuring element
    Mat element = getStructuringElement( MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
      
    // Closing
    morphologyEx(mask1, mask1, MORPH_CLOSE, element, Point(-1, -1), 2);
}

void segmentImg(Mat& originalImg, vector<Rect>& outBoxes, Mat& mask){
    for(int i = 0; i < outBoxes.size(); i++){ //for all bounding boxes in the image
        int margin = 5; //margin set to enlarge the bounding box in order to segment all the hand even if the bounding box is not precise
        
        Rect r = Rect(outBoxes.at(i).x, outBoxes.at(i).y, outBoxes.at(i).width, outBoxes.at(i).height);
       
        Mat mask1 = Mat::zeros(originalImg.rows, originalImg.cols, CV_8UC1);//all black, it will contain the mask of the single hand

        segmentation_rect(originalImg, r, mask1); //given the bounding box, segment the hand inside it
                  
        mask1 = (mask1==1)+(mask1==3); //color in white only the pixels labelled as sure foreground or probable forground
        closing(mask1, r);
        mask = mask + mask1; //add to the output mask the pixels of the segmented hand
        
        /*erode(mask1, mask1, cv::Mat());
        dilate(mask1, mask1, cv::Mat());*/ 
        
        Scalar random = Scalar(rand()%255,rand()%255,rand()%255); //choose a random color
        originalImg.setTo(random, mask1); //color the hand on the image with the chosen color
    }
}
