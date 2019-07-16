#ifndef SPORT_TRACKER_H
#define SPORT_TRACKER_H
#include "opencv2/opencv.hpp"




#include <iostream>
using namespace std;

class sport_tracker
{
public:
    sport_tracker();
     vector<cv::Point> SURF_tracking(vector<cv::Mat> std_samples,cv::Mat image);
     void NNet_tracking(string xml_path);
     cv::Mat trackingRect(cv::Mat insrc,cv::Mat inrect,int steps_update);
};

#endif // SPORT_TRACKER_H
