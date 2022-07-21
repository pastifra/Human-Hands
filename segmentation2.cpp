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

    /*Mat img = imread("/home/local/ceccmar29554/Desktop/progetto/03.jpg");
    Mat img_got = imread("/home/local/ceccmar29554/Desktop/progetto/03.jpg");
    Rect_<int> rect, rect1;//to create a rectangle
    vector<int> numbers;//all the bounding boxes
    Mat mask_got = imread("/home/local/ceccmar29554/Desktop/progetto/03.png");


    ifstream file("/home/local/ceccmar29554/Desktop/progetto/03.txt");*/
    
    Mat img = imread("/home/local/furcann17861/Downloads/progetto/03.jpg");
    Mat img_got = imread("/home/local/furcann17861/Downloads/progetto/03.jpg");
    Rect_<int> rect, rect1;//to create a rectangle
    vector<int> numbers;//all the bounding boxes
    Mat mask_got = imread("/home/local/furcann17861/Downloads/progetto/03.png"); 
    Mat mask = Mat::zeros(img.rows,img.cols, CV_8UC1);//all black

    ifstream file("/home/local/furcann17861/Downloads/progetto/03.txt");
    string str; 
    while(getline(file, str)){//read all the lines and save the bb in numbers
        int number;
        std::cout << str <<endl;
        istringstream ss(str);
        while(ss>>number){
            std::cout << number <<endl;
            numbers.push_back(number);
        }
    }
     
    //TRY TO IMPROVE ACCURACY WITH DIVISION OF CHANNELS
    blur(img, img, Size(3,3));//increase a little bit accuracy
    
    //divide channels
    Mat hsvImg, threshImg;
    cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);
  
    inRange(hsvImg, Scalar(0, 58, 50), Scalar(30, 255, 255), threshImg);//color skin
    imshow("color",threshImg);
    waitKey(0);
    //grabCut requires image in rgb
    cv::cvtColor(threshImg, threshImg, cv::COLOR_RGBA2RGB);
    //END NEW CODE
    
    for(int i = 0; i < numbers.size(); i = i + 4){
        Rect r = Rect(numbers.at(i),numbers.at(i+1), numbers.at(i+2), numbers.at(i+3));
        
        //try to improve grabcut
        /*Mat contours;
        cv::Canny(img,contours,80,150);
        cv::namedWindow("Image");
        cv::imshow("Image",img);
        waitKey(0);*/
        
        Mat mask1 = Mat::zeros(img.rows,img.cols, CV_8UC1);//all black
        Mat bgdModel = Mat::zeros(1,65, CV_64F);
        Mat fgdModel = Mat::zeros(1,65, CV_64F);

    
        grabCut(img,mask1,r,bgdModel,fgdModel,5,GC_INIT_WITH_RECT);//segmentation based on rec
        
        mask = mask + (mask1 == 1) + (mask1 == 3);
    }
    
    Mat dst;
    img.copyTo(dst, mask);
    
    Mat dst_got;
    img_got.copyTo(dst_got, mask_got);

    double TP, TN, FP, FN;

    for(int i = 0; i < mask.rows && mask_got.rows; i++){//got
        for(int j = 0; j < mask.cols && mask_got.cols; j++){
           if(mask.at<Vec3b>(i,j)[0]==255 && mask.at<Vec3b>(i,j)[1]==255 && mask.at<Vec3b>(i,j)[3]==255){
                if(mask_got.at<Vec3b>(i,j)[0]==255 && mask_got.at<Vec3b>(i,j)[1]==255 && mask_got.at<Vec3b>(i,j)[3]==255){//correctly classified as hands
                    TP++;
                }else{//incorrectly classified as hands
                    FP++;
                }
            }else if(mask_got.at<Vec3b>(i,j)[0]==255 && mask_got.at<Vec3b>(i,j)[1]==255 && mask_got.at<Vec3b>(i,j)[3]==255){//incorrectly classified as not hands
                    FN++;  
                }else{//correctly classified as not hands
                    TN++;
                }
          }
    }
        
    //calculate accuracy
    double pixel_accuracy = (TP + TN)/(TP + TN + FP +FN);
    cout<<"Pixel accuracy:"<< pixel_accuracy<<endl;

    //calculate IoU
    double iou = (TP)/(TP+FP+FN);
    cout<<"IoU:"<< iou <<endl;
    
    cout<<"TP:"<< TP<< endl
        <<"TN:" <<TN<< endl
        <<"FP:" <<FP<< endl
        <<"FN:" <<FN<< endl;
  
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
