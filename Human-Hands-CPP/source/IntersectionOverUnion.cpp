#include <numeric>
#include "../include/detection.h"

using namespace cv;
using namespace std;

double calc_iou(Rect bBox, Rect gt){

    //Non overlapping cases then IoU = 0.0:
    //Bottom_right smaller than top left
    if((bBox.x + bBox.width) < gt.x)
        return 0.0;
    if((bBox.y + bBox.height) < gt.y)
        return 0.0;
    if((gt.x + gt.width) < bBox.x)
        return 0.0;
    if((gt.y + gt.height) < bBox.y)
        return 0.0;

    //Top left corner of intersection
    int x_in = max(bBox.x, gt.x);
    int y_in = max(bBox.y, gt.y);
    //Bottom right corner of intersection
    int x1_in = min(bBox.x + bBox.width, gt.x + gt.width);
    int y1_in = min(bBox.y + bBox.height, gt.y + gt.height);
    int w_in = x1_in - x_in;
    int h_in = y1_in - y_in;
    //Area intersection
    double area_in = w_in * h_in;
    //Area union
    double area_un = (bBox.width * bBox.height) + (gt.width * gt.height) - area_in;
    //Intersection Over Union
    double IoU = area_in / area_un;

    return IoU;
}

double single_img_results(vector<Rect> prop_bboxes, vector<Rect> true_bboxes){
    //This function returns the average IOU for a single image given its proposed bboxes and corresponding ground truth
    
    if(prop_bboxes.size() == 0)
        return 0.0;

    vector<double> maxIous;
    vector<double> Iou_results;

    //For each ground truth find the prop bbox that overlaps the most
    for(int i = 0; i < true_bboxes.size(); i++){
        Rect true_bbox = true_bboxes.at(i);
        for(int j = 0; j < prop_bboxes.size(); j++){
            Rect prop_bbox = prop_bboxes.at(j);
            Iou_results.push_back(calc_iou(prop_bbox,true_bbox));
        }
        double element = *max_element(Iou_results.begin(), Iou_results.end());
        maxIous.push_back(element);
    }

        //If there are more ground truth than len of max_Ious fill the list with 0s until their lenght are the same
    if(true_bboxes.size() > maxIous.size()){
        for(int m = 0; m < (true_bboxes.size() - maxIous.size()); m++){
            maxIous.push_back(0.0);
        }
    }
    
    double sum_of_elements = 0;
    sum_of_elements = std::accumulate(maxIous.begin(), maxIous.end(), 0.0);
    double avg_Iou = sum_of_elements/maxIous.size();
    return(avg_Iou);
}

