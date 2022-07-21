#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "functions.h"
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

    Mat img = imread("/home/local/ceccmar29554/Desktop/progetto/01.jpg");
    Mat img_got = imread("/home/local/ceccmar29554/Desktop/progetto/01.jpg");
    Mat mask_got = imread("/home/local/ceccmar29554/Desktop/progetto/01.png");
    string filePath = "/home/local/ceccmar29554/Desktop/progetto/01.txt";
    
    vector<int> numbers;//all the bounding boxes
    Mat mask = Mat::zeros(img.rows,img.cols, CV_8UC1);//all black
    Mat dst;
    
    readNumbers(numbers, filePath);

    for(int i = 0; i < numbers.size(); i = i + 4){

        Mat mask1 = Mat::zeros(img.rows,img.cols, CV_8UC1);//all black
        Rect r = Rect(numbers.at(i),numbers.at(i+1), numbers.at(i+2), numbers.at(i+3));
        
        segmentation_rect(img, r, mask1);
        improveMask(r, mask1);
        segmentation_mask(img, mask1);
        vector<double> avg_intensity;
        calculate_avg(avg_intensity, img, r, mask1);
        double global_avg = avg_intensity.at(0)+avg_intensity.at(1)+avg_intensity.at(2);
        improveMask_avg(img, r, avg_intensity, mask1);      
        
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
