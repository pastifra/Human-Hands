#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, char** argv ){
    dnn::Net net;
    string cfg_path = "/home/local/ceccmar29554/Desktop/progetto/darknet/yolov4-obj.cfg";
    string weights_path = "/home/local/ceccmar29554/Desktop/progetto/darknet/yolov4-obj_final.weights";
    net = dnn::readNetFromDarknet(cfg_path, weights_path);
    
}
