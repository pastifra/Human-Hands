#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

    Mat img = imread("/home/local/ceccmar29554/Desktop/progetto/01.jpg");
    Mat img_got = imread("/home/local/ceccmar29554/Desktop/progetto/01.jpg");
    Rect_<int> rect, rect1;//to create a rectangle
    vector<int> numbers;//all the bounding boxes
    Mat mask_got = imread("/home/local/ceccmar29554/Desktop/progetto/01.png");
    Mat mask = Mat::zeros(img.rows,img.cols, CV_8UC1);//all black
    Mat gray_mat;
    Mat dst;

    ifstream file("/home/local/ceccmar29554/Desktop/progetto/01.txt");
    string str; 
    while(getline(file, str)){//read all the lines and save the bb in numbers
        int number;
        //std::cout << str <<endl;
        istringstream ss(str);
        while(ss>>number){
            //std::cout << number <<endl;
            numbers.push_back(number);
        }
    }
     

    for(int i = 0; i < numbers.size(); i = i + 4){
        Rect r = Rect(numbers.at(i),numbers.at(i+1), numbers.at(i+2), numbers.at(i+3));
    
        Mat mask1 = Mat::zeros(img.rows,img.cols, CV_8UC1);//all black
        Mat bgdModel = Mat::zeros(1,65, CV_64F);
        Mat fgdModel = Mat::zeros(1,65, CV_64F);

    
        grabCut(img,mask1,r,bgdModel,fgdModel,5,GC_INIT_WITH_RECT);//segmentation based on rec
        
        //Mat mask2 = Mat::zeros(img.rows,img.cols, CV_8UC1);
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

        /*cout << "x" << r.x << endl;
        cout << "y" << r.y << endl;
        cout << "width" << r.width/2 << endl;
        cout << "height" << r.height << endl;
        imshow("mask1", mask1);
        waitKey(0);*/
        
        Rect r1;
        grabCut(img,mask1, r1,bgdModel,fgdModel,5,GC_INIT_WITH_MASK);

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
        double avg_intensity_1 = sum_1/number_pixels;
        double avg_intensity_2 = sum_2/number_pixels;
        double avg_intensity_3 = sum_3/number_pixels;
        
        for(int m = r.x; m < r.x + r.width; m++){
            for(int n = r.y; n < r.y + r.height; n++){
                if(mask1.at<uchar>(n,m) != 0 && abs(img.at<Vec3b>(n,m)[0] - avg_intensity_1) > 10 && abs(img.at<Vec3b>(n,m)[1] - avg_intensity_2) > 10 && abs(img.at<Vec3b>(n,m)[2] - avg_intensity_3) > 100 && (m < r.x + r.width/4 || m > r.x + 3*r.width/4 || n < r.y + r.height/4 || n > r.y + r.height*3/4)){
                    mask1.at<uchar>(n,m) = 0;
                }
            }
        }
        
        /*int x_center = r.x + r.width/2;
        int y_center = r.y + r.height/2;
        for(int m = x_center - r.width/40; m < x_center + r.width/40; m++){
            for(int n = y_center - r.height/40; n < y_center + r.height/40; n++){
                mask1.at<uchar>(n,m) = 1;
            }
        }
        grabCut(img,mask1, r1,bgdModel,fgdModel,5,GC_INIT_WITH_MASK);*/
        mask1 = (mask1==1)+(mask1==3);
        mask = mask + mask1;
        img.copyTo(dst, mask);
    }
    
    Mat dst_got;
    img_got.copyTo(dst_got, mask_got);

    imshow("img",dst);
    waitKey(0);
    imshow("img_got", dst_got);
    waitKey(0);
    imshow("mask", mask);
    waitKey(0);
    imshow("mask_got", mask_got);
    waitKey(0);
    
    return 0;
}
