/*************************************************************************
    > File Name: backgraound_change.cpp
    > Author: jiang
    > Mail: 760021776@qq.com 
    > Created Time: 2019年03月11日 星期一 17时05分32秒
 ************************************************************************/

#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;

int main(int argc,char *argv[])
{
    if(argc <= 1)
        return 1;
    else{
        cv::Mat image,gray;
        for(int i = 1;i < argc-1;i++){
                image = cv::imread(argv[i]);
                cv::cvtColor(image,gray,CV_BGR2GRAY);
                for(int x = 0;x < image.cols;x++){
                    for(int y = 0;y<image.cols;y++){
                        if(gray.at<uchar>(y,x) > 200){
                            image.at<cv::Vec3b>(y,x) = {0,0,0};
                        }
                        else
                            break;
                        }
                    for(int yy = image.rows-1;yy > 0;yy--){
                        if(gray.at<uchar>(yy,x) > 200){
                            image.at<cv::Vec3b>(yy,x) = {0,0,0};
                        }
                        else
                            break;                  // 一列扫描完 针对列扫描无死角的
                        }
                    }
                    cv::imwrite(to_string(i)+".png",image);
                    cv::imshow("show",image);
                    cv::waitKey(0);
            }
        }
        return 0;
    }

