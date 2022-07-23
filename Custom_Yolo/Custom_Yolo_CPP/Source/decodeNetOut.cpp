#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "classes.h"

using namespace std;
using namespace cv;
using namespace dnn;

void decodeNetOut(cv::Mat output, cv::Mat image)
{
    typedef Vec<float, 5> Vec5f; //definition of template to get the outputs of the network
    
    int cell_size = 64; //Cell size is 448/7, needed to convert the values of the rect
    
    //Iterate over each grid cell to find the bboxes
    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            if(output.at<Vec5f>(i,j)[0] > 0.6) //Keep only boxes with a confidence bigger than 0.6
            {
                float x = output.at<Vec5f>(i,j)[1];
                float y = output.at<Vec5f>(i,j)[2];
                float w_cell = output.at<Vec5f>(i,j)[3];
                float h_cell = output.at<Vec5f>(i,j)[4];
                
                double ox = x*cell_size + cell_size*j;
                double oy = y*cell_size + cell_size*i;
                double w = w_cell*448;
                double h = h_cell*448;
                double lx = ox - w/2;
                double ly = oy - h/2;
                double bx = ox + w/2;
                double by = oy + h/2;
                rectangle(image, Point(static_cast<int>(lx),static_cast<int>(ly)), Point(static_cast<int>(bx),static_cast<int>(by)), Scalar(0,255,0), 1, LINE_8);
            }
        }
    }

    namedWindow("Image", cv::WINDOW_AUTOSIZE);
    imshow("Image", image);
    waitKey(0);

}
