#ifndef FACEDEAL_H
#define FACEDEAL_H
#include <opencv2/opencv.hpp>
#include <iostream>

#define PI 3.14159
const  cv::Scalar colors[] = { // 红橙黄绿青蓝紫
                                  CV_RGB(255, 0, 0),
                                  CV_RGB(255, 97, 0),
                                  CV_RGB(255, 255, 0),
                                  CV_RGB(0, 255, 0),
                                  CV_RGB(0, 255, 255),
                                  CV_RGB(0, 0, 255),
                                  CV_RGB(160, 32, 240)
                                  };

extern int out_width,out_height;
class facedeal
{
private:

    cv::CascadeClassifier face_cascade,eye_cascade;
    std::vector<cv::Rect> faceRect,eyeRect;           // all functions share
    int radio;
    double  reduce_radio;
    int arr_x,err_y;

//    cv::Mat facedata;

public:
    std::vector<cv::Rect> faces,eyes;  //
    facedeal();
    int* data_save(cv::Mat image);
    cv::Mat face_detection(cv::Mat image);
    cv::Mat eyes_detection(cv::Mat image);
    cv::Mat get_perspectiveMat(cv::Mat image,cv::Mat src,cv::Mat eyes);
    bool get_faceErr(cv::Mat image,cv::Mat eyes);
    cv::Mat skin_detection(cv::Mat image,cv::Mat facedata,int maxCr,int minCr,int maxCb,int minCb,int value1,int value2);

    cv::Mat hat_stick(cv::Mat hat0,cv::Mat facedata,cv::Mat image,cv::Mat eyesdata,bool c,float mask_w=1.5,float mask_h=0.9,float pan_h=0.8);
    cv::Mat glass_stick(cv::Mat glass0,cv::Mat image,cv::Mat eyesdata,bool c,float mask_w=1.2,float mask_h=1.2);
    cv::Mat mouth_stick(cv::Mat mouth0,cv::Mat image,cv::Mat nosedata,bool c,float mask_w=1.6,float mask_h=1.7,float pan_h=20);

    cv::Mat change_style(cv::Mat image,int style_choose);

    cv::Mat WarmCold_image(cv::Mat image,int value);

    cv::Mat emboss_image(cv::Mat image,int kernel);

    cv::Mat sketch_image(cv::Mat image,int value);

    cv::Mat cartoon_image(cv::Mat image,int value);

    cv::Mat add_Contrast(cv::Mat image,int value);
};

#endif // FACEDEAL_H
