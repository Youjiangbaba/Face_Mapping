#ifndef GLOBAL_VARIABLE_H
#define GLOBAL_VARIABLE_H
/*
 * 如何进行类直接的值传递？
 * //        for(int i = 0;i<image.cols/2;i++)                             //bgr 图像  at<Vec3b>
//            for(int j = 0;j<image.rows/2;j++)
//                image.at<cv::Vec3b>(j,i) =0;
*/

#include <QMutex>
#include <opencv2/opencv.hpp>
//  ../FaceDeal_Demo1.22/
//opencv xml
#define FACE_XML  "../FaceDeal_Demo1.22/xml/haarcascade_frontalface_alt.xml"
#define EYES_XML  "../FaceDeal_Demo1.22/xml/haarcascade_mcs_eyepair_big.xml"
#define EYES_BP   "../FaceDeal_Demo1.22/xml/eyes.xml"
#define NOSE_BP   "../FaceDeal_Demo1.22/xml/nose.xml"
#define HAT1      "../FaceDeal_Demo1.22/photo/hat1.jpeg"
#define HAT2      "../FaceDeal_Demo1.22/photo/hat2.png"
#define HAT3      "../FaceDeal_Demo1.22/photo/hat3.png"                      //1.1 , 0.8 ,0.8
#define GLASS1    "../FaceDeal_Demo1.22/photo/glass1.jpeg"                  //  1.3  1.5
#define GLASS2    "../FaceDeal_Demo1.22/photo/glass2.png"                   // 1.4  1.7
#define GLASS3    "../FaceDeal_Demo1.22/photo/glass3.png"                    // 1.6  2.3
#define GLASS4    "../FaceDeal_Demo1.22/photo/cute1.png"                     // 1.3  3.0
#define CUTE1    "../FaceDeal_Demo1.22/photo/cute2.png"                     //2.5  3.0  -45
#define CUTE2    "../FaceDeal_Demo1.22/photo/cute3.png"
#define CUTE3    "../FaceDeal_Demo1.22/photo/cute4.png"                      //3.0  3.0  -50
#define MOUTH1    "../FaceDeal_Demo1.22/photo/mouth1.png"
#define MOUTH2    "../FaceDeal_Demo1.22/photo/mouth2.png"                    //2.0  3.4 150

//无参数风格选择
#define FREEZING   100   //冰冻
#define CASTING    200   //熔铸
#define Nostalgic  300   //怀旧
#define Comic_strip 400  //连环画

#define CAPTURE   0

extern QMutex m;
extern cv::Mat image;                      //摄像头读取帧
extern     cv::VideoCapture capture;        //视频
extern bool flag_dealok;                    //处理完成标志，为1则开启标签显示

extern int beauty_value1,beauty_value2;
extern int crmax,crmin,cbmax,cbmin;    //肤色检测调试用

extern float hat_w,hat_h,hat_pan_h;  //帽子位置确定用
extern int Contrast;
extern int fps;
extern int flag_mask[10];
extern int style_choose,color_temperature,sobel_kernel,sketch_255,cartoon_10;


#endif // GLOBAL_VARIABLE_H
