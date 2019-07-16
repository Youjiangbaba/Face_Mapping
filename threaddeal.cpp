#include "threaddeal.h"
#include "mainwindow.h"
#include "global_variable.h"
#include "facedeal.h"

#include <iostream>
#include <string>

#include <opencv2/ml/ml.hpp>
#include <my_define.h>

QMutex m;
cv::Mat image;
cv::VideoCapture capture;
bool flag_dealok = 0;
int beauty_value1 = 0,beauty_value2 = 0;

int crmax=173,crmin=150,cbmax=127,cbmin=80;       //173 129;127,80
float hat_w = 1.5,hat_h = 0.9,hat_pan_h = 0.8;
int fps = 0;
int flag_mask[10] = {0,0,0,0,0,0,0};
int style_choose = 0,color_temperature=0,sobel_kernel = 0,sketch_255 = 0,cartoon_10 =0;
int Contrast = 0;

std::string patheye = "../FaceDeal_Demo1.18/eyes/";

cv::Mat get_mapIMG(char *path)
{
    cv::Mat img = cv::imread(path);
    if(!img.data){
        while(1){
            cv::waitKey(200);
            std::cout<<"hat open error"<<std::endl;
        }
    }
    return img;
}

ThreadDeal::ThreadDeal()
{
    std::cout << "ok" <<std::endl;
}

//只面部识别  200-400fps ； 只眼部识别：120fps
#define  choose  1            //0 则分类器找面部 40-47fps ，屏蔽眼部分类器 70 - 80fps ； 1 神经网络找面部 52 - 56fps ;
void ThreadDeal::run()
{

//    std::vector<int> arr_err;
//    int this_arr = 0;
//    int steps = 100;
//    int start_mid = 2;
//    //训练出面部眼部、鼻子
//    for(int i = 0;i < steps;i++){
//        test_cvNNapi(900,EYES,1,start_mid+i);         //5 - 24
//        this_arr = test_cvNNapi(100,EYES,0);                         // 训练样本  1 训练；  0 预测，默认100
//        arr_err.push_back(this_arr);
//    }
//    for(int i = 0;i < steps;i++){
//        std::cout << i+start_mid<<": "<<arr_err[i]<<std::endl;
//    }


    test_cvNNapi(200,EYES,1);

    cv::Mat last_image;

    cv::Mat mask_image,mask_lastimage,err_image;
    cv::Scalar result_err = 0;

    char* bpxml1 = EYES_BP;


    cv::Ptr<cv::ml::ANN_MLP> bp1 = cv::ml::ANN_MLP::load(bpxml1);


    facedeal deal;                     //脸部处理类
    cv::Mat facedata(1, 3, CV_32FC1);                // x,y,w
    cv::Mat last_facedata(1, 3, CV_32FC1);
    cv::Mat err_face(1, 3, CV_32FC1);
    cv::Scalar mean_err = 0;

    cv::Mat eyesdata(1, 4, CV_32FC1);
    cv::Mat nosedata(1, 4, CV_32FC1);

    cv::Mat hat0 = get_mapIMG(HAT1);
    cv::Mat hat1 = get_mapIMG(HAT3);
    cv::Mat glass0 = get_mapIMG(GLASS1);
    cv::Mat glass1 = get_mapIMG(GLASS2);
    cv::Mat glass2 = get_mapIMG(GLASS3);
    cv::Mat glass3 = get_mapIMG(GLASS4);
    cv::Mat cute0 = get_mapIMG(CUTE1);
    cv::Mat cute1 = get_mapIMG(CUTE3);
    cv::Mat mouth0 = get_mapIMG(MOUTH1);
    cv::Mat mouth1 = get_mapIMG(MOUTH2);
//    window = new MainWindow;
    while(1){
        msleep(30);
        capture>>image;                     //先捕捉一帧存在last
        last_image = image.clone();
        if(last_image.data)
            break;
    }
    int noface = 0;
    while(1){
        if(!flag_dealok){
            if(!capture.isOpened()){
                continue;
            }
            capture>>image;
            cv::resize(image,image,cv::Size(640,480),cv::INTER_LINEAR);               //神经网络是以（６４０，４８０）训练的
 //           cv::Mat test = cv::imread("/home/jiang/图片/2019-05-14 10-07-46 的屏幕截图.png");
 //           if(!test.data){
 //               std::cout << "open err"<<std::endl;
 //           }
 //           image = test.clone();
            if(image.data)
            {
                cv::flip(image,image,2);            //镜像
                long t = cv::getTickCount();
                //对比度调节
                image = deal.add_Contrast(image,Contrast);

                //检测面部 返回面部矩阵信息
                facedata = deal.face_detection(image);

        #if 1  //消除闪烁
                if(!facedata.at<float>(0,2)){           //这一帧没检测到
                    cv::cvtColor(image,mask_image,CV_BGR2GRAY);
                    cv::cvtColor(last_image,mask_lastimage,CV_BGR2GRAY);
                    absdiff(mask_image,mask_lastimage,err_image);
                    //cv::imshow("ds",err_image);
                    result_err = mean(err_image);
                    //std::cout << "err: "<<result_err[0] <<std::endl;
                    if(result_err[0] < 1.5){
                        facedata = last_facedata.clone();
                    }
                }
                last_image = image.clone();

        #endif
        #if 1      //解决贴图抖动问题
                absdiff(facedata,last_facedata,err_face);       //两个矩阵差的绝对值
                mean_err = cv::sum(err_face);                  //mean求平均，<10 不显著
                //std::cout <<"last:"<<last_facedata<<"new:"<<facedata<< "err: "<<mean_err <<std::endl;
                if(mean_err[0] < 30)
                    facedata = last_facedata.clone();
                last_facedata = facedata.clone();


                //std::cout << result_err << "  "<<mean_err<<std::endl;
                mean_err[0] = 0;
                result_err[0] = 0;
        #endif



                eyesdata = (cv::Mat_<float>(1,4) << 0,0,0,0);                 //  x y w h

                if(facedata.at<float>(0,2) > 0)  //存在面部
                {
                     bp1->predict(facedata,eyesdata);
                     std::cout << eyesdata <<std::endl;
                     deal.get_faceErr(image,eyesdata);                   //class 中的 err值变换.用来得到仿射角度
                     if(flag_mask[0]){
                         deal.hat_stick(hat0,image,facedata,eyesdata,1,1.6,0.9,0.6);
                     }
                     if(flag_mask[1]){
                         deal.hat_stick(hat1,image,facedata,eyesdata,0,1.2,0.8,0.7);
                     }
    /*
                     //保存眼部区域，进行特征匹配训练，得出透视变换关系。
                     static char eyerect_num[]  = "000.jpg";
                     static int num_eye = 0;
                     cv::Mat roieyes;
                     num_eye++;
                     if(num_eye>=1000)
                         num_eye = 0;
                     else if(num_eye < 10){
                         eyerect_num[2] = num_eye + '0';
                     }
                     else if(num_eye < 100 && num_eye >= 10){
                         eyerect_num[2] = num_eye%10 + '0';
                         eyerect_num[1] = num_eye/10 + '0';
                     }
                     else if(num_eye >= 100 && num_eye < 1000){
                         eyerect_num[2] = num_eye%10 + '0';
                         eyerect_num[1] = num_eye/10 - num_eye/100*10 + '0';
                         eyerect_num[0] = num_eye/100 + '0';
                     }
                     roieyes = image(cv::Rect(eyesdata.at<float>(0,0)-10,eyesdata.at<float>(0,1)-10,\
                                              eyesdata.at<float>(0,2)+20,eyesdata.at<float>(0,3)+20));
                     cv::imwrite(patheye + eyerect_num,roieyes);
   */
                     //rectangle(image,cv::Point2d(facedata.at<float>(0,0),facedata.at<float>(0,1)),cv::Point2d(facedata.at<float>(0,0)+facedata.at<float>(0,2),facedata.at<float>(0,1)+facedata.at<float>(0,2)),cv::Scalar(0, 0, 255), 2);
                     //rectangle(image,cv::Point2d(eyesdata.at<float>(0,0),eyesdata.at<float>(0,1)),cv::Point2d(eyesdata.at<float>(0,0)+eyesdata.at<float>(0,2),eyesdata.at<float>(0,1)+eyesdata.at<float>(0,3)),cv::Scalar(255, 0, 255), 2);
                     if(flag_mask[2])
                        deal.glass_stick(glass0,image,eyesdata,1,1.3,1.5);
                     if(flag_mask[3])
                        deal.glass_stick(glass1,image,eyesdata,0,1.4,1.7);
                     if(flag_mask[8])
                        deal.glass_stick(glass2,image,eyesdata,0,1.6,2.3);
                     if(flag_mask[9])
                        deal.glass_stick(glass3,image,eyesdata,0,1.3,3.2);

                    //bp2.predict(facedata,nosedata);
                    nosedata.at<float>(0,0) = eyesdata.at<float>(0,0) + 0.25*eyesdata.at<float>(0,2);
                    nosedata.at<float>(0,1) = eyesdata.at<float>(0,1) + eyesdata.at<float>(0,3);
                    nosedata.at<float>(0,2) = 0.5*eyesdata.at<float>(0,2);
                    nosedata.at<float>(0,3) = 1.5*eyesdata.at<float>(0,3);
      //画出器官矩形
//                    rectangle(image,cv::Point2d(facedata.at<float>(0,0),facedata.at<float>(0,1)),\
//                    cv::Point2d(facedata.at<float>(0,0)+facedata.at<float>(0,2),facedata.at<float>(0,1)\
//                                +facedata.at<float>(0,2)),cv::Scalar(255, 0, 0), 2);
//                    rectangle(image,cv::Point2d(eyesdata.at<float>(0,0),eyesdata.at<float>(0,1)),\
//                    cv::Point2d(eyesdata.at<float>(0,0)+eyesdata.at<float>(0,2),eyesdata.at<float>(0,1)\
//                                +eyesdata.at<float>(0,3)),cv::Scalar(255, 100, 100), 2);
//                    rectangle(image,cv::Point2d(nosedata.at<float>(0,0),nosedata.at<float>(0,1)),\
//                    cv::Point2d(nosedata.at<float>(0,0)+nosedata.at<float>(0,2),nosedata.at<float>(0,1)\
//                                +nosedata.at<float>(0,3)),cv::Scalar(255, 255, 0), 2);

//                    rectangle(image,cv::Point2d((nosedata.at<float>(0,0)-10),(nosedata.at<float>(0,1))+50),\
//                    cv::Point2d((nosedata.at<float>(0,0)+10)+nosedata.at<float>(0,2),(nosedata.at<float>(0,1)+50)\
//                                +nosedata.at<float>(0,3)),cv::Scalar(122, 255, 90), 2);


                    if(flag_mask[4]){
                       deal.mouth_stick(cute0,image,nosedata,0,2.5,3.0,-45);
                    }
                    if(flag_mask[5]){
                       deal.mouth_stick(cute1,image,nosedata,0,3.0,3.0,-50);
                    }
                    if(flag_mask[6]){
                       deal.mouth_stick(mouth0,image,nosedata,1);
                    }
                    if(flag_mask[7]){
                       deal.mouth_stick(mouth1,image,nosedata,0,2,3.4,150);
                    }

                }

                else
                    std::cout << ++noface<<std::endl;

                //std::cout<< eyesdata <<std::endl;
                //deal.eyes_detection(image);//眼部直接检测，很慢，已经不用了


                image = deal.skin_detection(image,facedata,crmax,crmin,cbmax,cbmin,beauty_value1,beauty_value2);  // beauty

                image = deal.change_style(image,style_choose);                                                    //style

                if(!style_choose)                       //没有其他风格，冷暖、浮雕
                {
                    image = deal.WarmCold_image(image,color_temperature);
                    image = deal.emboss_image(image,sobel_kernel);
                    image = deal.sketch_image(image,sketch_255);
                    image = deal.cartoon_image(image,cartoon_10);
                }

                fps = cv::getTickFrequency()/(cv::getTickCount() - t);
                //std::cout<<" fps :"<< fps <<std::endl;
                flag_dealok = 1;
                //std::cout << "flag_dealok" <<std::endl;
               }        //捕捉到了图片
            }           //已经显示完？没必要
                msleep(1);
        }           //循环

    }


