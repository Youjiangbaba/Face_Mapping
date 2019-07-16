#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include<string>
using namespace cv;
using namespace cv::ml;



#include "my_define.h"

using namespace std;
using namespace cv;


int main(){

    // std::vector<int> arr_err;
    // int this_arr = 0;
    // int steps = 15;
    // int start_mid = 2;
    // //训练出面部眼部、鼻子
    // for(int i = 0;i < steps;i++){
    //     test_cvNNapi(800,EYES,1,start_mid+i);         //5 - 24
    //     this_arr = test_cvNNapi(100,EYES,0);                         // 训练样本  1 训练；  0 预测，默认100
    //    arr_err.push_back(this_arr);
    // }
    // for(int i = 0;i < steps;i++){
    //     std::cout << i+start_mid<<": "<<arr_err[i]<<std::endl;
    // }

    test_cvNNapi(800,EYES,1,10); 
    //下载神经网络
    Mat eyesdata = (cv::Mat_<float>(1,4) << 0,0,0,0); 
    Mat facedata = (cv::Mat_<float>(1,3) << 0,0,0);
    Ptr<ANN_MLP> bp = ANN_MLP::load("eyes.xml");//创建; //bp网络

	VideoCapture cap(0);
	cv::CascadeClassifier face_cascade,eye_cascade;
    	std::vector<cv::Rect> faceRect;           // all functions share
   	if(!face_cascade.load("haarcascade_frontalface_alt.xml"))
       {
        while(1)
             cout << "Load frontalface_alt.xml failed!" << endl;
        }

	Mat image,image_gray,image_face;
    int radio = 5;
    float reduce_radio = 1.0/radio;
	while(1){
	    cap>>image;
        if(!image.data)
            continue;
        cv::resize(image,image,cv::Size(640,480),cv::INTER_LINEAR);
	    cv::cvtColor(image,image_gray,CV_BGR2GRAY);
        cv::equalizeHist(image_gray,image_gray);                             // little effect
        cv::resize(image_gray,image_face,cv::Size(),reduce_radio,reduce_radio,cv::INTER_LINEAR);
        face_cascade.detectMultiScale(image_face, faceRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        for(size_t i = 0;i < faceRect.size();i ++)
        {
            //cout << "0: " << faceRect[i].width <<","<< faceRect[i].height <<"("<<faceRect[i].x <<","<<faceRect[i].y<<endl;
            faceRect[i] = faceRect[i] + cv::Point((radio-1) *faceRect[i].x, (radio-1) *faceRect[i].y); //平移左上顶点Point
            faceRect[i] = faceRect[i] + cv::Size((radio-1) *faceRect[i].width, (radio-1) *faceRect[i].height);  //缩放，左上顶点不变，宽高
        // faceRect[i] = faceRect[i] + cv::Point(radio *faceRect[i].x, radio *faceRect[i].y); //平移左上顶点Point
        // cout << "1: " << faceRect[i].width <<","<< faceRect[i].height <<"("<<faceRect[i].x <<","<<faceRect[i].y<<endl;
            //rectangle(image, faceRect[i], cv::Scalar(0, 0, 255), 1);  //画出识别的face
            facedata = (cv::Mat_<float>(1,3) << faceRect[0].x,faceRect[0].y,faceRect[0].width);
            break;
        }
        bp->predict(facedata,eyesdata);	
        rectangle(image,cv::Point2d(facedata.at<float>(0,0),facedata.at<float>(0,1)),\
                        cv::Point2d(facedata.at<float>(0,0)+facedata.at<float>(0,2),facedata.at<float>(0,1)\
                                    +facedata.at<float>(0,2)),cv::Scalar(255, 0, 0), 2);
        rectangle(image,cv::Point2d(eyesdata.at<float>(0,0),eyesdata.at<float>(0,1)),\
                        cv::Point2d(eyesdata.at<float>(0,0)+eyesdata.at<float>(0,2),eyesdata.at<float>(0,1)\
                                    +eyesdata.at<float>(0,3)),cv::Scalar(255, 100, 100), 2);
        imshow("image",image);
        int k = waitKey(1);
        if(k==27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

/*
*  神经网络
*  cmd  1   训练；样本数：group_data
*  cmd  0   预测；默认100
*  文件格式； ,out[0],[1],[2],[3],in[0],in[1],in[2],
*  organ_choose : EYES,NOSE,MOUTH,SMILE
*/
int test_cvNNapi(int group_data,int organ_choose,int cmd,int mid_num)
{

    char* bpxml;                    //训练好的网络
    string datafile;           //训练文件
    string testfile;          //测试文件
    string resultfile;   //预测和标准结果对比文件
    switch (organ_choose) {
        case EYES:
            bpxml = eyes_xml;
            datafile  = eyes_train;
            testfile =  eyes_test;
            resultfile = eyes_result;
        break;
        case NOSE:
            bpxml = nose_xml;
            datafile  = nose_train;
            testfile =  nose_test;
            resultfile = nose_result;
        break;
        case MOUTH:
            bpxml = mouth_xml;
            datafile  = mouth_train;
            testfile =  mouth_test;
            resultfile = mouth_result;
        break;
        case SMILE:
            bpxml = smile_xml;
            datafile  = smile_train;
            testfile =  smile_test;
            resultfile = smile_result;
        break;
    }
    char buffer[50];
    if(cmd == 1)
    {

        Ptr<ANN_MLP> bp = ANN_MLP::create();//创建; //bp网络
        // trainning
        int *IN_data = new int[IN_Dimension];
        int *OUT_data = new int[OUT_Dimension];
        //建立一个标签矩阵
        Mat labelsMat(group_data, OUT_Dimension, CV_32FC1);
        //建立一个训练样本矩阵
        Mat trainingDataMat(group_data, IN_Dimension, CV_32FC1);           // cols = 3 ; rows = group_data  at(y,x)

        fstream ifile;
        int count_hang = 0;
        ifile.open(datafile,ios::in);
        for(int i=0;i<group_data;i++)
        {
            //每行格式          ,101,22,333,120,333,12,340
            ifile.getline(buffer, 50, '\n');     //getline(char *,int,char) 表示该行字符达到 50 个或遇到换行就结束;
            int num_count = -1;
            for(int j = 0; j < 50; ){
                if(num_count >=6 )
                    break;
                else if(buffer[j] == ','){
                    num_count++;
                    if(num_count < 4){
                        if(buffer[j+4] == ','){
                            OUT_data[num_count] = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            OUT_data[num_count] = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            OUT_data[num_count] = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                    else {
                        if(buffer[j+4] == ','){
                            IN_data[num_count-4] = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            IN_data[num_count-4] = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            IN_data[num_count-4] = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                }
            }
            //cout << count_hang++ <<endl;
            //cout << OUT_data[0] << ","<< OUT_data[1] << ","<< OUT_data[2]<<","<< OUT_data[3] <<" ; " << IN_data[0] << ","<< IN_data[1] << ","<< IN_data[2] << endl;
            //以上得出一行的数据
            for(int k = 0;k < 3;k ++){                          //at (y,x)
                trainingDataMat.at<float>(i,k) = IN_data[k];
            }
            for(int k = 0;k < 4;k ++){
                labelsMat.at<float>(i,k) = OUT_data[k];                            //at (y,x)
            }
//            labelsMat = (cv::Mat_<float>(i,4) << OUT_data[0],OUT_data[1],OUT_data[2],OUT_data[3]);
//            trainingDataMat = (cv::Mat_<float>(i,3) << IN_data[0],IN_data[1],IN_data[2]);
        }
        //cout << labelsMat <<endl;

        ifile.close();
        cout << "训练中..."<<endl;
    //    cout << labelsMat <<endl;
    //    cout << trainingDataMat <<endl;

        /*定义神经网络及参数*/
        //create the neural network
        Mat_<int> layerSizes(1, 3);
        layerSizes(0, 0) = IN_Dimension;
        layerSizes(0, 1) = mid_num;
        layerSizes(0, 2) = OUT_Dimension;

        bp->setLayerSizes(layerSizes);//设置层数
        bp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);//激活函数
        bp->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);//训练方法


//        Mat layerSizes = (Mat_<int>(1, 3) << IN_Dimension,HID_LAYERS,OUT_Dimension);//含有两个隐含层的网络结构，输入、输出层各3个节点
//        /*layerSizes设置了有两个隐含层的网络结构：输入层，两个隐含层，输出层。输入层和输出层节点数均为2，中间隐含层每层有两个节点
//          create第二个参数可以设置每个神经节点的激活函数，默认为CvANN_MLP::SIGMOID_SYM
//        */
//        bp->setLayerSizes(layerSizes);//设置层数
//        bp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.1, 0.1);//激活函数
//        bp->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);//训练方法

//        //激活函数为SIGMOID函数，还可使用高斯函数(CvANN_MLP::GAUSSIAN)，阶跃函数(CvANN_MLP::IDENTITY)

        Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);//创建训练数据，ROW_SAMPLE表示data中每行为一个样本
        bp->train(trainData);//训练的接口train()

        bp->save(bpxml);//存储模型

        cout << "训练完成！"<<endl;
        delete [] IN_data;
        delete [] OUT_data;

    }
    else
    {

        Ptr<ANN_MLP> bp = ANN_MLP::load(bpxml);//读取模型
        cout << "下载完成！开始测试！"<<endl;
        //	/*使用训练好的网络结构分类新的数据*/
        Mat sampleMat(1, IN_Dimension,CV_32FC1); //一组测试数据
        Mat stdresultMat(1, OUT_Dimension,CV_32FC1); //一组测试数据的标准结果
        Mat responseMat; //一组预测结果
        fstream ifile2,ofile;
        ifile2.open(testfile,ios::in);
        ofile.open(resultfile,ios::out);
        float err[OUT_Dimension] = {0,0,0};
        float errerr = 0;
        float sum_err = 0.0;
        //100组数据测试
        for(int i=0;i<100;i++)
        {
            //每行格式          ,101,22,333,120,333,12,200,
            ifile2.getline(buffer, 50, '\n');     //getline(char *,int,char) 表示该行字符达到 50 个或遇到换行就结束;
            int num1_count = -1;
            for(int j = 0; j < 50; ){
                if(num1_count >=6 )
                    break;
                if(buffer[j] == ','){
                    num1_count++;                      // -1 + 1 = 0 从角标0开始
                    if(num1_count < 4){
                        if(buffer[j+4] == ','){
                            stdresultMat.at<float>(0,num1_count) = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            stdresultMat.at<float>(0,num1_count) = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            stdresultMat.at<float>(0,num1_count) = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                    else {
                        if(buffer[j+4] == ','){
                            sampleMat.at<float>(0,num1_count-4) = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            sampleMat.at<float>(0,num1_count-4) = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            sampleMat.at<float>(0,num1_count-4) = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                }
            }
            //cout<<  sampleMat << "    "<< stdresultMat <<endl;
            bp->predict(sampleMat, responseMat);
            //cout<< responseMat <<endl;
            //cout << "第 " << i << " 组：误差：";
            for(int k=0;k<OUT_Dimension;k++){
                err[k] = responseMat.at<float>(0,k) - stdresultMat.at<float>(0,k);
                ofile << responseMat.at<float>(0,k) <<","<<stdresultMat.at<float>(0,k)<<";  ";
                //cout <<err[k]<< " ";
                errerr += err[k]*err[k];  //方差
                errerr /= OUT_Dimension;
            }
            //cout <<"                        "<<errerr<<endl;


            ofile << endl;
            sum_err += errerr;
            errerr = 0.0;
        }
        //cout << "sum_err = "<<sum_err <<"  arr_err ="<<sum_err/100<<endl;
        ofile << "sum_err = "<<sum_err <<"  arr_err ="<<sum_err/100<<endl;
        ofile.close();
        ifile2.close();
        return int(sum_err/100);

    }
    return 0;
}

//cmd :
//g++ `pkg-config --cflags opencv` -o Cv_NNapi Cv_NNapi.cpp `pkg-config --libs opencv`
