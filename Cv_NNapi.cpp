#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>   
#include <fstream>
#include<string>
using namespace std;
using namespace cv;

//从眼部得到全脸  Mat eyesdata(Mat_<float>(1,3) << eyesRect.x,eyesRect.y,eyesRect.width)
cv::Mat  getFace_Bpnet(char *bpxml,cv::Mat eyesdata)
{
   CvANN_MLP bp; //bp网络
   bp.load(bpxml);//读取模型
   Mat facedata; //一组预测结果  facedata(1, 3,CV_32FC1)(Mat_<int>(1, 3) << 3,18,3)
   bp.predict(eyesdata, facedata);
   return facedata;
}

int test_cvNNapi(int group_data,int cmd)
{
    CvANN_MLP bp; //bp网络
    string datafile = "/home/jiang/Repositories/FaceDeal_Class/train_data/mydata.txt";
    string testfile = "/home/jiang/Repositories/FaceDeal_Class/train_data/testdata.txt";
    string resultfile = "/home/jiang/Repositories/FaceDeal_Class/train_data/resultfile.txt";
    char buffer[50];
    if(cmd == 1)
    {                      // trainning
        int *IN_data = new int[3];
        int *OUT_data = new int[3];
        //建立一个标签矩阵
        Mat labelsMat(group_data, 3, CV_32FC1);
        //建立一个训练样本矩阵
        Mat trainingDataMat(group_data, 3, CV_32FC1);           // cols = 3 ; rows = group_data  at(y,x)

        fstream ifile;
        int count_hang = 0;
        ifile.open(datafile,ios::in);
        for(int i=0;i<group_data;i++)
        {
            //每行格式          ,101,22,333,120,333,12,
            ifile.getline(buffer, 50, '\n');     //getline(char *,int,char) 表示该行字符达到 50 个或遇到换行就结束;
            int num_count = -1;
            for(int j = 0; j < 50; ){
                if(num_count >=5 )
                    break;
                else if(buffer[j] == ','){
                    num_count++;
                    if(num_count < 3){
                        if(buffer[j+4] == ','){
                            IN_data[num_count] = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            IN_data[num_count] = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            IN_data[num_count] = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                    else {
                        if(buffer[j+4] == ','){
                            OUT_data[num_count-3] = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            OUT_data[num_count-3] = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            OUT_data[num_count-3] = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                }
            }
            cout << count_hang++ <<endl;
            cout << IN_data[0] << ","<< IN_data[1] << ","<< IN_data[2] << " ; "<< OUT_data[0] << ","<< OUT_data[1] << ","<< OUT_data[2] << endl;
            //以上得出一行的数据
            for(int k = 0;k < 3;k ++){
                labelsMat.at<float>(i,k) = OUT_data[k];                            //at (y,x)
                trainingDataMat.at<float>(i,k) = IN_data[k];
            }
        }
        cout << labelsMat <<endl;

        ifile.close();
        cout << "训练中..."<<endl;
    //    cout << labelsMat <<endl;
    //    cout << trainingDataMat <<endl;

        /*定义神经网络及参数*/
        //CvANN_MLP bp; //bp网络
        CvANN_MLP_TrainParams params; //bp网络参数
        params.train_method = CvANN_MLP_TrainParams::BACKPROP;		//使用简单的BP算法，还可使用RPROP
        /*BACKPROP表示使用back-propagation的训练方法，使用BACKPROP有两个相关参数：bp_dw-scale,bp_moment_scale
          RPROP即最简单的propagation训练方法,使用PRPOP有四个相关参数：rp_dw0,rp-dw_plus,rp_dw_minus,rp_dw_min,rp_dw_max
         一个是权值更新率bp_dw_scale和权值更新冲量bp_moment_scale。
         这两个量一般情况设置为0.1就行了；太小了网络收敛速度会很慢，太大了可能会让网络越过最小值点
        */
        params.bp_dw_scale = 0.1;
        params.bp_moment_scale = 0.1;
        //params.term_crit=cvTermCriteria(CV_TerMCrIT_ITER+CV_TERMCRIT_EPS,5000,0.01);

        /*设置网络层数，训练数据*/
        Mat layerSizes = (Mat_<int>(1, 3) << 3,18,3);//含有两个隐含层的网络结构，输入、输出层各3个节点
        /*layerSizes设置了有两个隐含层的网络结构：输入层，两个隐含层，输出层。输入层和输出层节点数均为2，中间隐含层每层有两个节点
          create第二个参数可以设置每个神经节点的激活函数，默认为CvANN_MLP::SIGMOID_SYM
        */
        bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);//激活函数为SIGMOID函数，还可使用高斯函数(CvANN_MLP::GAUSSIAN)，阶跃函数(CvANN_MLP::IDENTITY)
        bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);//训练的接口train()

        bp.save("bp.xml");//存储模型

        cout << "训练完成！"<<endl;
        delete [] IN_data;
        delete [] OUT_data;

    }
    else
    {
        bp.load("bp.xml");//读取模型
        cout << "下载完成！开始测试！"<<endl;
        //	/*使用训练好的网络结构分类新的数据*/
        Mat sampleMat(1, 3,CV_32FC1); //一组测试数据
        Mat stdresultMat(1, 3,CV_32FC1); //一组测试数据
        Mat responseMat; //一组预测结果
        fstream ifile2,ofile;
        ifile2.open(testfile,ios::in);
        ofile.open(resultfile,ios::out);
        float err[3] = {0,0,0};
        float errerr = 0;
        float sum_err = 0.0;
        //100组数据测试
        for(int i=0;i<100;i++)
        {
            //每行格式          ,101,22,333,120,333,12,
            ifile2.getline(buffer, 50, '\n');     //getline(char *,int,char) 表示该行字符达到 50 个或遇到换行就结束;
            int num1_count = -1;
            for(int j = 0; j < 50; ){
                if(num1_count >=5 )
                    break;
                else if(buffer[j] == ','){
                    num1_count++;
                    if(num1_count < 3){
                        if(buffer[j+4] == ','){
                            sampleMat.at<float>(0,num1_count) = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            sampleMat.at<float>(0,num1_count) = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            sampleMat.at<float>(0,num1_count) = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                    else {
                        if(buffer[j+4] == ','){
                            stdresultMat.at<float>(0,num1_count-3) = (buffer[j+1] - '0')*100 + (buffer[j+2] - '0')*10 + (buffer[j+3] - '0')*1;
                            j = j+4;
                        }

                        else if (buffer[j+3] == ','){
                            stdresultMat.at<float>(0,num1_count-3) = (buffer[j+1] - '0')*10 + (buffer[j+2] - '0')*1;
                            j = j+3;
                        }

                        else if(buffer[j+2] == ','){
                            stdresultMat.at<float>(0,num1_count-3) = (buffer[j+1] - '0')*1;
                            j = j+2;
                        }
                        else
                            break;
                    }
                }
            }
            cout<<  sampleMat << "    "<< stdresultMat <<endl;
            bp.predict(sampleMat, responseMat);
            cout<< responseMat <<endl;
            cout << "第 " << i << " 组：误差：";
            for(int k=0;k<3;k++){
                err[k] = responseMat.at<float>(0,k) - stdresultMat.at<float>(0,k);
                ofile << responseMat.at<float>(0,k) <<","<<stdresultMat.at<float>(0,k)<<";";
                cout <<err[k]<< " ";
                errerr += err[k]*err[k];  //方差
                errerr /= 3;
            }
            cout <<"                        "<<errerr<<endl;
            ofile << endl;
            sum_err += errerr;
            errerr = 0.0;
        }
        cout << "sum_err = "<<sum_err <<"  arr_err ="<<sum_err/100<<endl;
        ofile.close();
        ifile2.close();
    }
    return 0;
}


//cmd : 
//g++ `pkg-config --cflags opencv` -o Cv_NNapi Cv_NNapi.cpp `pkg-config --libs opencv`

