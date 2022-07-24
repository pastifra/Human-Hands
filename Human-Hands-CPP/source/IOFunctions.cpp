#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "../include/IOfunctions.h"

using namespace std;
using namespace cv;

string getPath(string indication){
    String output;
    cout << indication;
    cin >> output;
    return output;
}

void readNumbers(string filePath, vector<Rect>& gtBoxes){
    vector<int> numbers;
    ifstream file(filePath);
    string str; 
    while(getline(file, str)){//read all the lines and save the bb in numbers
        int number;
        istringstream ss(str);
        while(ss>>number){
            numbers.push_back(number);
        }
    }

    int counter = 0;
    for(int i = 0; i < numbers.size(); i = i + 4){
        Rect r = Rect(numbers.at(i),numbers.at(i+1), numbers.at(i+2), numbers.at(i+3));
        gtBoxes.push_back(r);
    }
}

void displayImg(string caption, Mat& image){
    // Show the output image with predictions
    namedWindow(caption);
    imshow(caption, image);
    waitKey(0);
}

void printBbox(Rect& r, string caption){
    cout << caption << endl;
    cout << "x: " << r.x << endl;
    cout << "y: " << r.y << endl;     
    cout << "width: " << r.width << endl;  
    cout << "height: " << r.height << "\n" << endl;
}
