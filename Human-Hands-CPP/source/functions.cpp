#include "../include/functions.h"

using namespace std;
using namespace cv;

void readNumbers(vector<int>& numbers, string filePath){
    ifstream file(filePath);
    string str; 
    while(getline(file, str)){//read all the lines and save the bb in numbers
        int number;
        istringstream ss(str);
        while(ss >> number){
            numbers.push_back(number);
        }
    }
}

void improveBySkin(Mat& img){
    Mat hsvImg, threshImg;         
    //TRY TO IMPROVE ACCURACY WITH DIVISION OF CHANNELS
    blur(img, img, Size(3,3));//increase a little bit accuracy
        
    //divide channels
    cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);
      
    //color skin
    inRange(hsvImg, Scalar(0, 58, 50), Scalar(30, 255, 255), threshImg);
    /*imshow("color",threshImg);
    waitKey(0);*/
    //grabCut requires image in rgb
    cv::cvtColor(threshImg, threshImg, cv::COLOR_RGBA2RGB);
}

void segmentation_rect(Mat& img, Rect& r, Mat& mask1){
    Mat bgdModel = Mat::zeros(1, 65, CV_64F);
    Mat fgdModel = Mat::zeros(1, 65, CV_64F);
    grabCut(img, mask1, r, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);//segmentation based on rect
} 



void improveMask(Rect& r, Mat& mask1){
        for(int m = r.x; m < r.x + r.width/20; m++){
            for(int n = r.y; n < r.y + r.height/20; n++){
                mask1.at<uchar>(n,m) = 0;
            }
        }
        
        for(int m = r.x; m < r.x + r.width/20; m++){
            for(int n = r.y + r.height; n > r.y + r.height - r.height/20; n--){
                mask1.at<uchar>(n,m) = 0;
            }
        }
        
        for(int m = r.x + r.width; m > r.x + r.width - r.width/20; m--){
            for(int n = r.y; n < r.y + r.height/20; n++){
                mask1.at<uchar>(n,m) = 0;
            }
        }
        
        for(int m = r.x + r.width; m > r.x + r.width - r.width/20; m--){
            for(int n = r.y + r.height; n > r.y + r.height - r.height/20; n--){
                mask1.at<uchar>(n,m) = 0;
            }
        }
}

void segmentation_mask(Mat& img,Mat& mask1){
        Rect r1;
        Mat bgdModel = Mat::zeros(1,65, CV_64F);
        Mat fgdModel = Mat::zeros(1,65, CV_64F);
        grabCut(img, mask1, r1, bgdModel, fgdModel, 5, GC_INIT_WITH_MASK);
} 

void calculate_avg(vector<double>& avg_intensity, Mat& img, Rect& r, Mat& mask1){
        int sum_1 = 0;
        int sum_2 = 0;
        int sum_3 = 0;
        int number_pixels = 0;
        for(int m = r.x; m < r.x + r.width; m++){
            for(int n = r.y; n < r.y + r.height; n++){
                if(mask1.at<uchar>(n,m) != 0){
                    number_pixels++;
                    sum_1 += img.at<Vec3b>(n,m)[0];
                    sum_2 += img.at<Vec3b>(n,m)[1];
                    sum_3 += img.at<Vec3b>(n,m)[2];
                }
            }
        }
        avg_intensity.push_back(sum_1/number_pixels);
        avg_intensity.push_back(sum_2/number_pixels);
        avg_intensity.push_back(sum_3/number_pixels);
}

void improveMask_avg(Mat& img,Rect& r,vector<double>& avg_intensity,Mat& mask1){
    double x_center = r.x + r.width/2;
    double y_center = r.y + r.height/2;
    for(int m = r.x; m < r.x + r.width; m++){
        for(int n = r.y; n < r.y + r.height; n++){
            double dist_x = abs(m - x_center);
            double dist_y = abs(n - y_center);
            double ray_x = r.width/8;
            double ray_y = r.height/8;
            vector<double> intensity_diff;
            intensity_diff.push_back(abs(img.at<Vec3b>(n,m)[0]-avg_intensity.at(0)));
            intensity_diff.push_back(abs(img.at<Vec3b>(n,m)[1]-avg_intensity.at(1)));
            intensity_diff.push_back(abs(img.at<Vec3b>(n,m)[2]-avg_intensity.at(2)));
            if((dist_x > ray_x || dist_y > ray_y) && intensity_diff[0] > 40 && intensity_diff[1] > 40 && intensity_diff[2] > 40){
                    mask1.at<uchar>(n,m) = 0;
            }
        }
   }
}
