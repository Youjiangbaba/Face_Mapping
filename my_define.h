#ifndef MY_DEFINE_H
#define MY_DEFINE_H

#include <opencv2/opencv.hpp>

#define IN_Dimension   3
#define OUT_Dimension   4
#define HID_LAYERS      12
#define PI 3.14159
const int EYES = 100;
const int NOSE = 200;
const int MOUTH = 300;
const int SMILE = 400;


#define eyes_xml   "../FaceDeal_Demo1.22/xml/eyes.xml"
#define eyes_train "/home/jiang/Repositories/FaceDeal_Class/train_data/eyes.txt" //"../FaceDeal_Demo1.22/train_dataold/eyes.txt"
#define eyes_test  "../FaceDeal_Demo1.22/train_dataold/testdata.txt"
#define eyes_result  "../FaceDeal_Demo1.22/resultfile.txt"
///home/jiang/Repositories/FaceDeal_Demo1.22/train_dataold/mydata.txt

#define nose_xml    "../FaceDeal_Demo1.22/xml/nose.xml"
#define nose_train  "/home/jiang/Repositories/FaceDeal_Class/train_data/nose.txt"
#define nose_test   "/home/jiang/Repositories/FaceDeal_Class/train_data/testnose.txt"
#define nose_result  "/home/jiang/Repositories/FaceDeal_Class/train_data/resultnose.txt"

#define mouth_xml     "mouth.xml"
#define mouth_train  "/home/jiang/Repositories/FaceDeal_Class/train_data/mouth.txt"
#define mouth_test  "/home/jiang/Repositories/FaceDeal_Class/train_data/testmouth.txt"
#define mouth_result  "/home/jiang/Repositories/FaceDeal_Class/train_data/resultmouth.txt"

#define smile_xml   "smile.xml"
#define smile_train  "/home/jiang/Repositories/FaceDeal_Class/train_data/smile.txt"
#define smile_test  "/home/jiang/Repositories/FaceDeal_Class/train_data/testsmile.txt"
#define smile_result  "/home/jiang/Repositories/FaceDeal_Class/train_data/resultsmile.txt"
extern int test_cvNNapi(int group_data,int organ_choose,int cmd,int mid_num = HID_LAYERS);


#endif // MY_DEFINE_H
