#include "facedeal.h"
#include <math.h>
//ctrl + i 自动缩进
#include "global_variable.h"
using namespace std;

extern cv::Mat  Rot_PerspectiveTrans(cv::Mat src,int an_x,int an_y,int anz,bool c);   //透视变换，在perspectivetrans.cpp里




facedeal::facedeal(){
    radio = 5;
    reduce_radio = 1.0/radio;
    if(!face_cascade.load(FACE_XML))
     {
        while(1)
             cout << "Load frontalface_alt.xml failed!" << endl;
        }
    if(!eye_cascade.load(EYES_XML))
     {
        while(1)
             cout << "Load eye_alt.xml failed!" << endl;
        }
}

// 返回 int[6]  目的：得出眼部和面部的坐标，通过神经网络训练出关系。（眼部识别更快速,用来推测出面部上边界和宽度）
//             int[0]代表识别到的区域的类型，2代表都识别到了
//             int[1] int[2]眼部的左上  int[3]眼部宽度
//             int[4] int[5]面部的左上  int[6]面部宽度
int *facedeal::data_save(cv::Mat image)        //参考https://www.cnblogs.com/walter-xh/p/6192800.html
{
    int *data = new int[7];
    cv::Mat image_gray,image_face;
    cv::cvtColor(image,image_gray,CV_BGR2GRAY);
    cv::equalizeHist(image_gray,image_gray);                             // little effect
    cv::resize(image_gray,image_face,cv::Size(),reduce_radio,reduce_radio,cv::INTER_LINEAR);

    data[0] = 0;
    //检测关于face位置 !注意：这里是缩放后的位置
        face_cascade.detectMultiScale(image_face, faceRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    for(size_t i = 0;i < faceRect.size();i ++){
        data[0]++;
        if (i >= 1)
            continue;
        faceRect[i] = faceRect[i] + cv::Point((radio-1) *faceRect[i].x, (radio-1) *faceRect[i].y); //平移左上顶点Point
        faceRect[i] = faceRect[i] + cv::Size((radio-1) *faceRect[i].width, (radio-1) *faceRect[i].height);  //缩放，左上顶点不变，宽高
        rectangle(image, faceRect[i], cv::Scalar(0, 0, 255), 1);  //画出识别的face
        data[4] = faceRect[i].x;
        data[5] = faceRect[i].y;
        data[6] = faceRect[i].width;
    }
    eye_cascade.detectMultiScale(image_gray, eyeRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    for(size_t i = 0;i < eyeRect.size();i ++){
        data[0]++;
        if (i >= 1)
            continue;
        rectangle(image, eyeRect[i], cv::Scalar(0, 0, 255), 2);  //画出识别的eyes
        data[1] = eyeRect[i].x;
        data[2] = eyeRect[i].y;
        data[3] = eyeRect[i].width;
    }
    return data;
}

//检测脸部，输入图像作为处理
cv::Mat facedeal::face_detection(cv::Mat image)
{
    cv::Mat image_gray,image_face,img;
    img = image.clone();
    cv::Mat facedata = (cv::Mat_<float>(1,3) << 0,0,0);
    cv::cvtColor(img,image_gray,CV_BGR2GRAY);

    cv::resize(image_gray,image_gray,cv::Size(),reduce_radio,reduce_radio,cv::INTER_LINEAR);
    cv::equalizeHist(image_gray,image_face);                             // little effect

    //检测关于face位置 !注意：这里是缩放后的位置
        face_cascade.detectMultiScale(image_face, faceRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    for(size_t i = 0;i < faceRect.size();i ++){
        //cout << "0: " << faceRect[i].width <<","<< faceRect[i].height <<"("<<faceRect[i].x <<","<<faceRect[i].y<<endl;
        faceRect[i] = faceRect[i] + cv::Point((radio-1) *faceRect[i].x, (radio-1) *faceRect[i].y); //平移左上顶点Point
        faceRect[i] = faceRect[i] + cv::Size((radio-1) *faceRect[i].width, (radio-1) *faceRect[i].height);  //缩放，左上顶点不变，宽高
       // faceRect[i] = faceRect[i] + cv::Point(radio *faceRect[i].x, radio *faceRect[i].y); //平移左上顶点Point
       // cout << "1: " << faceRect[i].width <<","<< faceRect[i].height <<"("<<faceRect[i].x <<","<<faceRect[i].y<<endl;
        //rectangle(image, faceRect[i], cv::Scalar(0, 0, 255), 1);  //画出识别的face
        facedata = (cv::Mat_<float>(1,3) << faceRect[0].x,faceRect[0].y,faceRect[0].width);
        break;
    }
    return facedata;
}

//检测眼部，输入图像进行处理
cv::Mat facedeal::eyes_detection(cv::Mat image)
{
    cv::Mat image_gray,image_eyes;
    cv::Mat eyesdata(1, 3, CV_32FC1);
    eyesdata = (cv::Mat_<float>(1,3) << 0,0,0);
    cv::cvtColor(image,image_gray,CV_BGR2GRAY);
    //cv::resize(image_gray,image_gray,cv::Size(),1,1,cv::INTER_LINEAR);
    cv::equalizeHist(image_gray,image_eyes);
    eye_cascade.detectMultiScale(image_eyes, eyeRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    for(size_t i = 0;i < eyeRect.size();i ++){
        rectangle(image, eyeRect[i], cv::Scalar(0, 0, 255), 2);  //画出识别的eyes
        eyesdata = (cv::Mat_<float>(1,3) << eyeRect[0].x,eyeRect[0].y,eyeRect[0].width);
        break;
    }
    return  eyesdata;
}

//从脸部得到x透视角度，从eye得出旋转角度
bool facedeal::get_faceErr(cv::Mat image,cv::Mat eyes)  //eyes <float>类型，其他类型会错误
{
    if(!faceRect.data()){
        err_y = 0;
        arr_x = 0;
        return 0;
    }
    cv::Mat img = image.clone();
    cv::Mat eyes0 = img(cv::Rect(eyes.at<float>(0,0),  eyes.at<float>(0,1) , eyes.at<float>(0,2), eyes.at<float>(0,3)));
    cv::Mat eyes1;
    cv::Point p1(0,0),p2(0,0);
    int num1=0,num2=0;
    cv::cvtColor(eyes0,eyes1,CV_BGR2GRAY);
    cv::threshold(eyes1,eyes1,70, 255, CV_THRESH_BINARY);
//    cv::imshow("eyemask",eyes0);
//    cv::imshow("mask",eyes1);
    //cv::imshow("eyes",eyes1);            // **************************************************************************************************************
    for(int ix = 0;ix < eyes1.cols/2;ix++)
        for(int iy = 0;iy <eyes1.rows;iy++){
            if(!eyes1.at<uchar>(iy,ix))   {
                p1.x += ix;
                p1.y +=iy;
                num1 ++;
            }
        }
    if(p1.x>0&&p1.y>0&&num1>0){	//防止 0/
        p1.x = p1.x/num1;
        p1.y = p1.y/num1;
    }
    for(int ix = eyes1.cols/2;ix < eyes1.cols;ix++)
        for(int iy = 0;iy <eyes1.rows;iy++){
            if(!eyes1.at<uchar>(iy,ix))   {
                p2.x += ix;
                p2.y +=iy;
                num2 ++;
            }
        }
    if(p2.x>0&&p2.y>0&&num2>0){	//防止 0/
        p2.x = p2.x/num2;
        p2.y = p2.y/num2;
    }
    err_y = p2.y - p1.y;             // > 0 : 左高右低



    int sum_err = 0;
    int start_y,end_y;
    cv::Mat faceroi = img(faceRect[0]);

    cv::Mat face2;
    cv::cvtColor(faceroi,face2,CV_BGR2GRAY);


    start_y = 0.5*faceroi.rows - 10;
    end_y = 0.5*faceroi.rows + 10;
    vector<int> face_left(end_y - start_y),face_right(end_y - start_y),face_err(end_y - start_y);
    cv::cvtColor(faceroi,faceroi,CV_BGR2YCrCb);
    for(int i = start_y;i <  end_y; i ++)
    {
        for(int j = 0;j < faceroi.cols; j++){
            if((faceroi.at<cv::Vec3b>(i,j)[1] > crmax)||(faceroi.at<cv::Vec3b>(i,j)[1] < crmin)
                    ||(faceroi.at<cv::Vec3b>(i,j)[2] > cbmax)||(faceroi.at<cv::Vec3b>(i,j)[2] < cbmin))
                ;
            else {
                face_left[i- start_y] = j;
                face2.at<uchar>(i,j) = 255;


                //break;

            }
        }
        for(int j = faceroi.cols;j > 0 ;j--){
            if((faceroi.at<cv::Vec3b>(i,j)[1] > crmax)||(faceroi.at<cv::Vec3b>(i,j)[1] < crmin)
                    ||(faceroi.at<cv::Vec3b>(i,j)[2] > cbmax)||(faceroi.at<cv::Vec3b>(i,j)[2] < cbmin))
                ;
            else {
                face_right[i - start_y] = faceroi.cols - j;
                face2.at<uchar>(i,j) = 255;

                //break;
            }
        }
        face_err[i - start_y] = face_left[i - start_y] - face_right[ i- start_y];
        sum_err += face_err[i - start_y];
    }


    //cv::imshow("face",face2);            // **************************************************************************************************************

    arr_x = 0.5*sum_err/(end_y - start_y);                        // > 0 : 左多右少
    //cout << "x "<<arr_x<<"    y "<<err_y<<endl;
    return  1;
}
/*
 *  源图像、贴图图像、眼部位置矩阵；输出贴图透视变换矩阵  x、y；没有z rot
 *  面部通过肤色和位置矩阵找出x方向透视变换关系
 *  眼部通过眼镜的mask，找出y方向关系
*/
cv::Mat facedeal::get_perspectiveMat(cv::Mat image,cv::Mat src,cv::Mat eyes)  //eyes <float>类型，其他类型会错误
{
    cv::Mat img = image.clone();
    cv::Mat eyes0 = img(cv::Rect(eyes.at<float>(0,0),  eyes.at<float>(0,1) , eyes.at<float>(0,2), eyes.at<float>(0,3)));
    cv::Mat eyes1;
    cv::Point p1(0,0),p2(0,0);
    int num1=0,num2=0;
    cv::cvtColor(eyes0,eyes1,CV_BGR2GRAY);
    cv::threshold(eyes1,eyes1,70, 255, CV_THRESH_BINARY);
//    cv::imshow("eyemask",eyes0);
    //cv::imshow("mask",eyes1);            // **************************************************************************************************************
    for(int ix = 0;ix < eyes1.cols/2;ix++)
        for(int iy = 0;iy <eyes1.rows;iy++){
            if(!eyes1.at<uchar>(iy,ix))   {
                p1.x += ix;
                p1.y +=iy;
                num1 ++;
            }
        }
    if(p1.x>0&&p1.y>0&&num1>0){	//防止 0/
        p1.x = p1.x/num1;
        p1.y = p1.y/num1;
    }
    for(int ix = eyes1.cols/2;ix < eyes1.cols;ix++)
        for(int iy = 0;iy <eyes1.rows;iy++){
            if(!eyes1.at<uchar>(iy,ix))   {
                p2.x += ix;
                p2.y +=iy;
                num2 ++;
            }

        }
    if(p2.x>0&&p2.y>0&&num2>0){	//防止 0/
        p2.x = p2.x/num2;
        p2.y = p2.y/num2;
    }
    err_y = p2.y - p1.y;             // > 0 : 左高右低


    int sum_err = 0;
    int start_y,end_y;
    cv::Mat faceroi = img(faceRect[0]);
    start_y = 0.5*faceroi.rows - 10;
    end_y = 0.5*faceroi.rows + 10;
    vector<int> face_left(end_y - start_y),face_right(end_y - start_y),face_err(end_y - start_y);
    cv::cvtColor(faceroi,faceroi,CV_BGR2YCrCb);
    for(int i = start_y;i <  end_y; i ++)
    {
        for(int j = 0;j < faceroi.cols; j++){
            if((faceroi.at<cv::Vec3b>(i,j)[1] > crmax)||(faceroi.at<cv::Vec3b>(i,j)[1] < crmin)
                    ||(faceroi.at<cv::Vec3b>(i,j)[2] > cbmax)||(faceroi.at<cv::Vec3b>(i,j)[2] < cbmin))
                ;
            else {
                face_left[i- start_y] = j;
                break;
            }
        }
        for(int j = faceroi.cols;j > 0 ;j--){
            if((faceroi.at<cv::Vec3b>(i,j)[1] > crmax)||(faceroi.at<cv::Vec3b>(i,j)[1] < crmin)
                    ||(faceroi.at<cv::Vec3b>(i,j)[2] > cbmax)||(faceroi.at<cv::Vec3b>(i,j)[2] < cbmin))
                ;
            else {
                face_right[i - start_y] = faceroi.cols - j;
                break;
            }
        }
        face_err[i - start_y] = face_left[i - start_y] - face_right[ i- start_y];
        sum_err += face_err[i - start_y];
    }

    arr_x = sum_err/(end_y - start_y);                        // > 0 : 左多右少

    cout << err_y <<"     "<< arr_x<<endl;
    if(arr_x <= 25 && arr_x >= -25)
        arr_x = 0;
    if(err_y <= 10 && err_y >= -10)
        err_y = 0;

    if(!arr_x)
    {
        return (cv::Mat_<float>(3,3) << 0,0,0,0,0,0,0,0,0);
    }
    float anx,any;
    anx = -0.002*arr_x;    //x边动，y中轴不变
    any = 0;                    //
    vector<cv::Point2f> corners(4),corners_trans(4);
    corners[0] = cv::Point2f(0,0);
    corners[1] = cv::Point2f(src.cols-1,0);
    corners[2] = cv::Point2f(src.cols-1,src.rows-1);
    corners[3] = cv::Point2f(0,src.rows-1);

    int dis_x,dis_y,dis_xx1,dis_yy1,dis_xx2,dis_yy2;
    dis_x = 0.5*src.rows*cos(any)*sin(any);
    dis_y = 0.5*src.rows*cos(any)*cos(any);
    dis_xx1 = 0.5*(src.cols + 2*dis_x)*cos(anx)*cos(anx);
    dis_yy1 = 0.5*(src.cols + 2*dis_x)*cos(anx)*sin(anx);
    dis_xx2 = 0.5*(src.cols - 2*dis_x)*cos(anx)*cos(anx);
    dis_yy2 = 0.5*(src.cols - 2*dis_x)*cos(anx)*sin(anx);

    corners_trans[0].x = 0.5*src.cols - dis_xx2;
    corners_trans[0].y = (0.5*src.rows - dis_y) - dis_yy2;

    corners_trans[1].x =0.5*src.cols + dis_xx2;
    corners_trans[1].y = (0.5*src.rows - dis_y) + dis_yy2;

    corners_trans[2].x = 0.5*src.cols + dis_xx1;
    corners_trans[2].y = src.rows - (dis_yy1 +  (0.5*src.rows - dis_y));

    corners_trans[3].x = 0.5*src.cols - dis_xx1;
    corners_trans[3].y = src.rows + (dis_yy1 -  (0.5*src.rows - dis_y));

    //求外切矩形 左x比小、右x比大；上y比小，下y比大。

    cv::Point2f change_point = cv::Point2f(0,0);

    if(corners_trans[0].y == corners_trans[1].y)   // anx = 0   上下平行
    {
        out_height = corners_trans[3].y - corners_trans[0].y;
        if(corners_trans[0].x > corners_trans[3].x){
            //cout << "下大"<<endl;
            out_width = corners_trans[2].x - corners_trans[3].x;
            change_point.x = 0 - corners_trans[3].x;
            change_point.y = out_height  - corners_trans[3].y;
        }

        else{
            //cout << "上大"<<endl;
            out_width = corners_trans[1].x - corners_trans[0].x;
            change_point.x = 0 - corners_trans[0].x;
            change_point.y = 0 - corners_trans[0].y;
        }
        corners_trans[0] += change_point;
        corners_trans[1] += change_point;
        corners_trans[2] +=change_point;
        corners_trans[3] += change_point;

    }
    else if(corners_trans[0].x == corners_trans[3].x)  // any = 0   左右平行
    {
        out_width = corners_trans[1].x - corners_trans[0].x;
        if(corners_trans[0].y > corners_trans[1].y){
            //cout << "右大"<<endl;
            out_height = corners_trans[2].y - corners_trans[1].y;
            change_point.x = out_width - corners_trans[1].x;
            change_point.y = 0  - corners_trans[1].y;
        }

        else{
            //cout << "左大"<<endl;
            out_height = corners_trans[3].y - corners_trans[0].y;
            change_point.x = 0 - corners_trans[0].x;
            change_point.y = 0 - corners_trans[0].y;
        }
        corners_trans[0] += change_point;
        corners_trans[1] += change_point;
        corners_trans[2] += change_point;
        corners_trans[3] += change_point;
    }
    else if(corners_trans[0].x < corners_trans[3].x && corners_trans[0].y < corners_trans[1].y)  //左上 0
    {
        //cout << "左上"<<endl;
        out_width = corners_trans[1].x - corners_trans[0].x;
        out_height = corners_trans[3].y - corners_trans[0].y;

        change_point.x = 0 - corners_trans[0].x;
        change_point.y = 0 - corners_trans[0].y;

        corners_trans[1] += change_point;
        corners_trans[2] += change_point;
        corners_trans[3] += change_point;
        corners_trans[0] += change_point;

    }
    else if(corners_trans[1].x > corners_trans[2].x && corners_trans[1].y < corners_trans[0].y)  //右上 1
    {
        //cout << "右上"<<endl;
        out_width = corners_trans[1].x - corners_trans[0].x;
        out_height = corners_trans[2].y - corners_trans[1].y;

        change_point.x = out_width  - corners_trans[1].x;
        change_point.y = 0 - corners_trans[1].y;

        corners_trans[0] += change_point;
        corners_trans[1] += change_point;
        corners_trans[2] += change_point;
        corners_trans[3] += change_point;
    }
    else if(corners_trans[2].x > corners_trans[1].x && corners_trans[2].y > corners_trans[3].y)  //右下 2
    {
        //cout << "右下"<<endl;
        out_width = corners_trans[2].x - corners_trans[3].x;
        out_height = corners_trans[2].y - corners_trans[1].y;

        change_point.x = out_width   - corners_trans[2].x;
        change_point.y = out_height  - corners_trans[2].y;

        corners_trans[0] += change_point;
        corners_trans[1] += change_point;
        corners_trans[2] += change_point;
        corners_trans[3] += change_point;
    }
    else if(corners_trans[3].x < corners_trans[0].x && corners_trans[3].y > corners_trans[2].y)  //左下 3
    {
        //cout << "左下"<<endl;
        out_width = corners_trans[2].x - corners_trans[3].x;
        out_height = corners_trans[3].y - corners_trans[0].y;

        change_point.x = 0  - corners_trans[3].x;
        change_point.y = out_height  - corners_trans[3].y;

        corners_trans[0] += change_point;
        corners_trans[1] += change_point;
        corners_trans[2] += change_point;
        corners_trans[3] += change_point;
    }

    //cout << corners_trans <<endl;
    //调用opencv四点函数得出矩阵 getPerspectiveTransform(const Point2f src[], const Point2f dst[])
    cv::Mat mat33 = getPerspectiveTransform(corners,corners_trans);
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(0.5*eyes.at<float>(0,2),0.5*eyes.at<float>(0,3)), err_y/eyes.at<float>(0,2), 1);


    //cout << mat33 <<endl;
    return  mat33;
}

//肤色检测 输入脸部区域 + 美颜
cv::Mat facedeal::skin_detection(cv::Mat image,cv::Mat facedata,int maxCr,int minCr,int maxCb,int minCb,int value1,int value2)
{

    if((!value1)&&(!value2))
        return image;
    cv::Mat img;
    img = image.clone();
    cv::Mat faceroi;
//        if(!facedata.at<float>(0,2)){
//            cout << "no face"<<endl;
//            return img;
//        }
//        int face_x0,face_y0,face_x1,face_y1;
//        face_x0 = (facedata.at<float>(0,0)>0)?facedata.at<float>(0,0):0;
//        face_y0 = (facedata.at<float>(0,1)>0)?facedata.at<float>(0,1):0;
//        face_x1 = face_x0 + facedata.at<float>(0,2);
//        face_y1 = face_y0 + facedata.at<float>(0,2);
//        face_x1 = (face_x1<img.cols)?face_x1:img.cols;
//        face_y1 = (face_y1<img.rows)?face_y1:img.rows;

//        rectangle(img, cv::Point2f(face_x0,face_y0), cv::Point2f(face_x1,face_y1), cv::Scalar(0, 255, 255), 2);
//        faceroi = img(cv::Rect(face_x0,face_y0,face_x1-face_x0,face_y1-face_y0));

        if(!faceRect.data()){
                    cout << "no face"<<endl;
                    return img;
        }
        //扩大一点
        faceroi = img(cv::Rect(faceRect[0].x,faceRect[0].y-30,faceRect[0].width,faceRect[0].height+30));
    if(value2){
    cv::cvtColor(faceroi,faceroi,CV_BGR2YCrCb);
//    cv::Mat Y, Cr, Cb;
//	vector<cv::Mat> channels;
//    split(faceroi, channels);
//    Cr = channels.at(1);
//    Cb = channels.at(2);

    //遍历每个像素点
//    for(int i =0;i < faceroi.rows;i++){
//        for(int j = 0;j < faceroi.cols;j++){
//            if((faceroi.at<cv::Vec3b>(i,j)[1] > maxCr)||(faceroi.at<cv::Vec3b>(i,j)[1] < minCr)
//                    ||(faceroi.at<cv::Vec3b>(i,j)[2] > maxCb)||(faceroi.at<cv::Vec3b>(i,j)[2] < minCb)){
//                //faceroi.at<cv::Vec3b>(i,j) = 0;
//                ;
//            }
//            else  //肤色
//            {
//                faceroi.at<cv::Vec3b>(i,j)[0] += 15;
//            }
//        }
//    }
    //逐行扫描，边界满足则下一行；左右各一次  :  提高扫描效率，且眼镜等默认为肤色  //173 129;127,80
        int j2 = 0;
        for(int i =0;i < faceroi.rows;i++){
            for(int j = 0;j < faceroi.cols;j++){
                if((faceroi.at<cv::Vec3b>(i,j)[1] > maxCr)||(faceroi.at<cv::Vec3b>(i,j)[1] < minCr)
                        ||(faceroi.at<cv::Vec3b>(i,j)[2] > maxCb)||(faceroi.at<cv::Vec3b>(i,j)[2] < minCb)){
                    ;//faceroi.at<cv::Vec3b>(i,j) = 0;
                }
                else {
                    if(j < faceroi.cols-1){
                        for(int k = faceroi.cols-1; k > j;k--){                 //同一行扫描第二次
                            if((faceroi.at<cv::Vec3b>(i,k)[1] > maxCr)||(faceroi.at<cv::Vec3b>(i,k)[1] < minCr)
                                    ||(faceroi.at<cv::Vec3b>(i,k)[2] > maxCb)||(faceroi.at<cv::Vec3b>(i,k)[2] < minCb)){
                                ;//faceroi.at<cv::Vec3b>(i,k) = 0;
                            }
                            else{
                                j2 = k;
                                break;
                            }
                        }
                        // j - k 之间进行美白
                        for(int n = j; n < j2;n++)
                            faceroi.at<cv::Vec3b>(i,n)[0] += value2;          //这里直接进行亮度增加
                            //faceroi.at<cv::Vec3b>(i,n)[0] = 255;   //test
                    }
                    break;                                                  //下一行
                }
            }
        }
        cv::cvtColor(faceroi,faceroi,CV_YCrCb2BGR);
    }

    //////////////////////////////////////////// airbrushing ////////////////////////////////////////////////
    if(value1){
        cv::Mat dst;
        int dx = value1 * 5;    //双边滤波参数之一
        double fc = value1*12.5; //双边滤波参数之一
        int p = 50; //透明度
        cv::Mat temp1, temp2, temp3, temp4;
        //双边滤波
        cv::bilateralFilter(faceroi, temp1, dx, fc, fc);
        temp2 = (temp1 - faceroi + 128);
        //高斯模糊
        if(value2 == 0)  value2 = 1;
        cv::GaussianBlur(temp2, temp3, cv::Size(2 * 2 - 1, 2 * 2 - 1), 0, 0);  //ksize 1 3 5 7
        temp4 = faceroi + 2 * temp3 - 255;
        dst = (faceroi*(100 - p) + temp4*p) / 100;
        dst.copyTo(faceroi);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    return img;

}


//贴帽子
cv::Mat facedeal::hat_stick(cv::Mat hat0,cv::Mat image,cv::Mat facedata,cv::Mat eyesdata,bool c,float mask_w,float mask_h,float pan_h)
{

    cv::Mat hat1,hat_g,hat_mask1,hat_mask2;
    cv::Mat imageROI;

 //   for (size_t i = 0; i < faceRect.size();i++)
    {
        int width2,height2;
        cv::Point p1,p2;

//        if(i >= 1)
//            continue;      //只找一个
        p1.x =   cvRound(facedata.at<float>(0,0));
        p1.y =   cvRound(facedata.at<float>(0,1));
        p2.x =   cvRound((facedata.at<float>(0,0) +facedata.at<float>(0,2)));
        p2.y =   cvRound((facedata.at<float>(0,1) +facedata.at<float>(0,2)));

        width2 =  facedata.at<float>(0,2);
        height2 = facedata.at<float>(0,2);

        /**********************************************透视变换************************************/
//        if(faceRect.data()){
//            cv::Mat perspectiveMat( 3, 3, CV_32FC1 );
//            perspectiveMat = get_perspectiveMat(image,hat0,eyesdata);
//            if(perspectiveMat.at<float>(0,0)){
//               cv::warpPerspective(hat0,hat1,perspectiveMat,cv::Size(out_width,out_height),1,0,cv::Scalar(255,255,255));
//               cv::resize(hat1,hat1,cv::Size(cvRound(mask_w*width2),cvRound(mask_h*height2)));  //比例根据mask样式调节,如何自适应？   --大小控制
//            }
//            else
//                cv::resize(hat0,hat1,cv::Size(cvRound(mask_w*width2),cvRound(mask_h*height2)));  //比例根据mask样式调节,如何自适应？   --大小控制
//        }
//        else
//            cv::resize(hat0,hat1,cv::Size(cvRound(mask_w*width2),cvRound(mask_h*height2)));  //比例根据mask样式调节,如何自适应？   --大小控制

        if(abs(arr_x) < 25)
             arr_x = 0;
        if(abs(err_y) < 2)
             err_y = 0;

        hat1 = Rot_PerspectiveTrans(hat0,arr_x,0,err_y,c);
        cv::resize(hat1,hat1,cv::Size(cvRound(mask_w*width2),cvRound(mask_h*height2)));  //比例根据mask样式调节,如何自适应？   --大小控制

//cv::imshow("0",hat1);
        int ROI_x,ROI_y;
        ROI_x = cvRound(p1.x-(hat1.cols-width2)/2) + arr_x;         //增加透视变换水平误差
        ROI_y = cvRound(p1.y-hat1.rows*pan_h);		   //感兴趣区域的y0                          --上下平移控制

        int Dx[2],Dy[2];
        Dx[0] = image.cols - (ROI_x + hat1.cols);       //imageROI的x_max是否超过image.cols
        Dx[1] = ROI_x;                                  //x_min 是否小于0
        Dy[0] = image.rows - (ROI_y + hat1.rows);		      //y_max
        Dy[1] = ROI_y;                                             //y_min

        // 不考虑两个以上边界的情况
        //cout << "脸："<< "("<<ROI_x<<","<<ROI_y<<endl;
        if(Dx[0]>=0&&Dx[1]>=0&&Dy[0]>=0&&Dy[1]>=0){
            imageROI = image(cv::Rect(ROI_x, ROI_y, hat1.cols, hat1.rows));
            cv::cvtColor(hat1,hat_g,CV_BGR2GRAY);
            if(c == 0)
                cv::threshold(hat_g,hat_mask1, 50, 255, CV_THRESH_BINARY_INV);
            else if(c==1)
                cv::threshold(hat_g,hat_mask1, 220, 255, CV_THRESH_BINARY);
            cv::bitwise_not(hat_mask1,hat_mask2);
        }
        else{
            for(int j = 0; j < 2;j ++)
            {
                if(Dx[j] >= 0)
                    Dx[j] = 0;
                if(Dy[j] >= 0)
                    Dy[j] = 0;
            }

            hat1(cv::Rect(-Dx[1],-Dy[1], hat1.cols + Dx[0] +Dx[1], hat1.rows + Dy[0] +Dy[1])).copyTo(hat1);
            imageROI = image(cv::Rect(ROI_x - Dx[1],ROI_y - Dy[1],hat1.cols, hat1.rows));                   //显示不完整
            cv::cvtColor(hat1,hat_g,CV_BGR2GRAY);
            if(c == 0)
                cv::threshold(hat_g,hat_mask1, 50, 255, CV_THRESH_BINARY_INV);
            else if(c==1)
                cv::threshold(hat_g,hat_mask1, 220, 255, CV_THRESH_BINARY);
            cv::bitwise_not(hat_mask1,hat_mask2);
            //cv::rectangle(image, p1,p2,  colors[3]);
        }

        //cv::imshow("1",hat_mask1);
        //cv::imshow("2",hat_mask2);
        cv::Mat i1(cv::Size(hat_mask1.cols,hat_mask1.rows),CV_8UC3);
        cv::addWeighted(hat1,1.0,imageROI,0.4,0.,i1);                    //与原图比例加
        //cv::imshow("3",i1);
        hat1.copyTo(imageROI,hat_mask2);
        //cv::imshow("4",imageROI);
        cv::medianBlur(imageROI,imageROI,5);
        //cv::imshow("5",imageROI);
   }

    return image;
}

//贴眼镜  mask要求 ： 眼镜中心为矩形中心
cv::Mat facedeal::glass_stick(cv::Mat glass0,cv::Mat image,cv::Mat eyesdata,bool c,float mask_w,float mask_h)
{
    cv::Mat imageROI;
    cv::Mat glass1,glass_g,glass_mask1,glass_mask2;

    int eyesx = eyesdata.at<float>(0,0);
    int eyesy = eyesdata.at<float>(0,1);
    int eyewidth =  eyesdata.at<float>(0,2);
    int eyeheight = eyesdata.at<float>(0,3);

    cv::Point p3,p4;
    p3.x =  eyesx;
    p3.y =  eyesy;
    p4.x =  eyesx + eyewidth;
    p4.y =  eyesy + eyeheight;

    //rectangle(image, p3[i],p4[i],  colors[1]);
    //cv::resize(glass0,glass1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);

    /**********************************************透视变换************************************/
    //     if(faceRect.data()){
    //         cv::Mat perspectiveMat( 3, 3, CV_32FC1 );
    //         perspectiveMat = get_perspectiveMat(image,glass0,eyesdata);
    //         if(perspectiveMat.at<float>(0,0)){
    //            cv::warpPerspective(glass0,glass1,perspectiveMat,cv::Size(out_width,out_height),1,0,cv::Scalar(255,255,255));
    //            cv::resize(glass1,glass1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);
    //         }
    //         else
    //             cv::resize(glass0,glass1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);
    //     }
    //     else
    //         cv::resize(glass0,glass1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);

    if(abs(arr_x) < 25)
        arr_x = 0;
    if(abs(err_y) < 3)
        err_y = 0;
    glass1 = Rot_PerspectiveTrans(glass0,arr_x,0,err_y,c);
    cv::resize(glass1,glass1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);

    //cv::imshow("1",glass1);


    //mask中心对两眼中心
    int Roi1x0;
    if(arr_x >=0 )
        Roi1x0 = cvRound(p3.x - (glass1.cols*0.5 - eyewidth*0.5) + 0.5*arr_x);        //增加透视变换水平误差
    else
        Roi1x0 = cvRound(p3.x - (glass1.cols*0.5 - eyewidth*0.5));
    int Roi1y0 = cvRound(p3.y - (glass1.rows*0.5 - eyeheight*0.5));

    //以下为防止glass roi超过img
    int ROI_x,ROI_y;
    ROI_x = Roi1x0;         //默认完整显示，感兴趣区域的x0
    ROI_y = Roi1y0;		   //感兴趣区域的y0
    //cout << "眼："<< "("<<ROI_x<<","<<ROI_y<<endl;

    int Dx[2],Dy[2];
    Dx[0] = image.cols - (ROI_x + glass1.cols);       //imageROI的x_max是否超过image.cols
    Dx[1] = ROI_x;                                  //x_min 是否小于0
    Dy[0] = image.rows - (ROI_y + glass1.rows);		      //y_max
    Dy[1] = ROI_y;                                             //y_min

    // 不考虑两个以上边界的情况
    if(Dx[0]>=0&&Dx[1]>=0&&Dy[0]>=0&&Dy[1]>=0){

        imageROI = image(cv::Rect(ROI_x,ROI_y, glass1.cols, glass1.rows));
    }
    else{
        for(int j = 0; j <2;j++)
        {
            if(Dx[j] >= 0)
                Dx[j] = 0;
            if(Dy[j] >= 0)
                Dy[j] = 0;
        }

        glass1(cv::Rect(-Dx[1],-Dy[1], glass1.cols + Dx[0] +Dx[1], glass1.rows + Dy[0] +Dy[1])).copyTo(glass1);
        imageROI = image(cv::Rect(ROI_x - Dx[1],ROI_y - Dy[1],glass1.cols, glass1.rows));        //显示不完整
    }
    cv::cvtColor(glass1,glass_g,CV_BGR2GRAY);
    if(c == 0)
        cv::threshold(glass_g,glass_mask1, 50, 255, cv::THRESH_BINARY_INV);
    else if(c == 1)
        cv::threshold(glass_g,glass_mask1, 100, 255, cv::THRESH_BINARY);
    cv::bitwise_not(glass_mask1,glass_mask2);


    cv::Mat i1(cv::Size(glass1.cols,glass0.rows),CV_8UC3);
    cv::addWeighted(glass1,1.0,imageROI,0.3,0.,i1);                    //与原图比例加
    //贴mask
    i1.copyTo(imageROI,glass_mask2);
    //*****************************************************************************************************************
//    cv::imshow("0",glass1);
//    cv::imshow("1",glass_mask1);
//    cv::imshow("2",glass_mask2);
//    cv::imshow("3",i1);
//    cv::imshow("4",imageROI);
//*****************************************************************************************************************

    cv::medianBlur(imageROI,imageROI,5);

 //   cv::imshow("5",imageROI);
    return image;
}

//贴嘴巴，通过鼻子的位置
cv::Mat facedeal::mouth_stick(cv::Mat mouth0,cv::Mat image,cv::Mat nosedata,bool c,float mask_w,float mask_h,float pan_h)
{
    //cv::imshow("dasasda",mouth0);
    cv::Mat imageROI;
    cv::Mat mouth1,mouth_g,mouth_mask1,mouth_mask2;

    int eyesx = nosedata.at<float>(0,0);
    int eyesy = nosedata.at<float>(0,1);
    int eyewidth =  nosedata.at<float>(0,2);
    int eyeheight = nosedata.at<float>(0,3);



    //rectangle(image, p3[i],p4[i],  colors[1]);
    //cv::resize(mouth0,mouth1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);

    /**********************************************透视变换************************************/
    //     if(faceRect.data()){
    //         cv::Mat perspectiveMat( 3, 3, CV_32FC1 );
    //         perspectiveMat = get_perspectiveMat(image,mouth0,eyesdata);
    //         if(perspectiveMat.at<float>(0,0)){
    //            cv::warpPerspective(mouth0,mouth1,perspectiveMat,cv::Size(out_width,out_height),1,0,cv::Scalar(255,255,255));
    //            cv::resize(mouth1,mouth1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);
    //         }
    //         else
    //             cv::resize(mouth0,mouth1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);
    //     }
    //     else
    //         cv::resize(mouth0,mouth1,cv::Size(),mask_w*eyewidth/glass0.cols,mask_h*eyeheight/glass0.rows,cv::INTER_LINEAR);

    if(abs(arr_x) < 25)
        arr_x = 0;
    if(abs(err_y) < 3)
        err_y = 0;
    mouth1 = Rot_PerspectiveTrans(mouth0,arr_x,0,err_y,c);
    cv::resize(mouth1,mouth1,cv::Size(),mask_w*eyewidth/mouth0.cols,mask_h*eyeheight/mouth0.rows,cv::INTER_LINEAR);
    //cv::imshow("ma",mouth1);

    //mask中心x对nose中心x
    int Roi1x0 = cvRound(eyesx - (mouth1.cols*0.5 - eyewidth*0.5));
    int Roi1y0 = cvRound(eyesy - (mouth1.rows*0.5 - eyeheight*0.5));
    //以下为防止mouth roi超过img
    int ROI_x,ROI_y;
    ROI_x = Roi1x0;         //默认完整显示，感兴趣区域的x0
    ROI_y = Roi1y0 + 0.5*eyeheight + pan_h;                //感兴趣区域的y0



    int Dx[2],Dy[2];
    Dx[0] = image.cols - (ROI_x + mouth1.cols);       //imageROI的x_max是否超过image.cols
    Dx[1] = ROI_x;                                  //x_min 是否小于0
    Dy[0] = image.rows - (ROI_y + mouth1.rows);               //y_max
    Dy[1] = ROI_y;                                             //y_min

    // 不考虑两个以上边界的情况
    if(Dx[0]>=0&&Dx[1]>=0&&Dy[0]>=0&&Dy[1]>=0){

        imageROI = image(cv::Rect(ROI_x,ROI_y, mouth1.cols, mouth1.rows));
    }
    else{
        for(int j = 0; j <2;j++)
        {
            if(Dx[j] >= 0)
                Dx[j] = 0;
            if(Dy[j] >= 0)
                Dy[j] = 0;
        }

        mouth1(cv::Rect(-Dx[1],-Dy[1], mouth1.cols + Dx[0] +Dx[1], mouth1.rows + Dy[0] +Dy[1])).copyTo(mouth1);
        imageROI = image(cv::Rect(ROI_x - Dx[1],ROI_y - Dy[1],mouth1.cols, mouth1.rows));        //显示不完整
    }
    cv::cvtColor(mouth1,mouth_g,CV_BGR2GRAY);
    if(c==0)
        cv::threshold(mouth_g,mouth_mask1, 50, 255, cv::THRESH_BINARY_INV);
    else if(c==1)
        cv::threshold(mouth_g,mouth_mask1, 220, 255, cv::THRESH_BINARY );



    cv::bitwise_not(mouth_mask1,mouth_mask2);

    cv::Mat i1(cv::Size(mouth1.cols,mouth0.rows),CV_8UC3);
    cv::addWeighted(mouth1,1.0,imageROI,0.3,0.,i1);                    //与原图比例加
    //贴mask
    i1.copyTo(imageROI,mouth_mask2);
    cv::medianBlur(imageROI,imageROI,5);

    return image;
}

/*  风格变换
*********××××××××××× 以下为图像风格处理（0,1,2,3）  无入口参数调节 **************************
********* 参考  https://blog.csdn.net/yangtrees/article/details/9116337
*怀旧：
*           float newB=0.272*R+0.534*G+0.131*B;
            float newG=0.349*R+0.686*G+0.168*B;
            float newR=0.393*R+0.769*G+0.189*B;
 连环画：
            R = |g – b + g + r| * r / 256;
            G = |b – g + b + r| * r / 256;
            B = |b – g + b + r| * g / 256;
 熔铸：
            r = r*128/(g+b +1);
            g = g*128/(r+b +1);
            b = b*128/(g+r +1);
 冰冻:
            r = (r-g-b)*3/2;
            g = (g-r-b)*3/2;
            b = (b-g-r)*3/2;
*/
cv::Mat facedeal::change_style(cv::Mat image,int style_choose)
{
    if(!style_choose)
        return  image;
    int radio_resize = 1;                                    // 模糊处理（缩小增加帧率）
    cv::Mat img;
    img = image.clone();
    //img.convertTo(img,CV_32FC3);
    //分离通道，整张操作
    std::vector<cv::Mat> channels,new_channels;   // b g r    单通道操作比像素点遍历快200帧！
    cv::split(img, new_channels);
    cv::split(img, channels);
    switch (style_choose)
    {
    case Nostalgic:
        new_channels[0] = 0.272*channels[2]+0.534*channels[1]+0.131*channels[0];
        new_channels[1] = 0.349*channels[2]+0.686*channels[1]+0.168*channels[0];
        new_channels[2] = 0.393*channels[2]+0.769*channels[1]+0.189*channels[0];
        cv::merge(new_channels,img);
        break;
    case Comic_strip:
        //cv::cvtColor(img,img,CV_BGR2GRAY);
        //cv::imshow("g",img);
        new_channels[2] = abs(channels[0] - channels[1] + channels[1] +channels[2]).mul(0.0039*channels[2]);
        new_channels[1] = abs(channels[1] - channels[0] + channels[0] +channels[2]).mul(0.0039*channels[2]);
        new_channels[0] = abs(channels[1] - channels[0] + channels[0] +channels[2]).mul(0.0039*channels[1]);
        cv::merge(new_channels,img);

        //        //slow
//        for(int i = 0;i<3;i++){
//            channels[i].convertTo(channels[i],CV_32FC1);
//            new_channels[i].convertTo(new_channels[i],CV_32FC1);
//        }
//        multiply(abs(channels[1] - channels[0] + channels[1] +channels[2]),0.0039*channels[2],new_channels[2]);        //<CV_32FC1>
//        //new_channels[2] *= 0.0039;
//        multiply(abs(channels[0] - channels[1] + channels[0] +channels[2]),0.0039*channels[2],new_channels[1]);
//        //new_channels[1] /= 256;
//        multiply(abs(channels[0] - channels[1] + channels[0] +channels[2]),0.0039*channels[1],new_channels[0]);
//        //new_channels[0] /= 256;
//        for(int i = 0;i<3;i++){
//            channels[i].convertTo(channels[i],CV_8UC1);
//            new_channels[i].convertTo(new_channels[i],CV_8UC1);
//        }
//                for (int y=0; y<img.rows; y++)
//                {
//                    uchar* P0  = img.ptr<uchar>(y);
//                    uchar* P1  = img.ptr<uchar>(y);
//                    for (int x=0; x<img.cols; x++)
//                    {
//                        int B=P0[3*x];
//                        int G=P0[3*x+1];
//                        int R=P0[3*x+2];
//                        int newB=static_cast<int>(abs(B-G+B+R)*G/256);
//                        int newG=static_cast<int>(abs(B-G+B+R)*R/256);
//                        int newR=static_cast<int>(abs(G-B+G+R)*R/256);
//                        if(newB<0)newB=0;
//                        if(newB>255)newB=255;
//                        if(newG<0)newG=0;
//                        if(newG>255)newG=255;
//                        if(newR<0)newR=0;
//                        if(newR>255)newR=255;
//                        P1[3*x] = (uchar)newB;
//                        P1[3*x+1] = (uchar)newG;
//                        P1[3*x+2] = (uchar)newR;
//                    }
//                }
        break;
    case FREEZING:
        new_channels[2] = 1.5*abs(channels[2]-channels[1]-channels[0]);
        new_channels[1] = 1.5*abs(channels[1]-channels[2]-channels[0]);
        new_channels[0] = 1.5*abs(channels[0]-channels[1]-channels[2]);
        cv::merge(new_channels,img);
        break;

    case CASTING:
//        for(int i = 0;i<3;i++){
//            channels[i].convertTo(channels[i],CV_32FC1);
//            new_channels[i].convertTo(new_channels[i],CV_32FC1);
//        }
//        divide(channels[0],(channels[1]+channels[2]+1),new_channels[0],128.0);
//        divide(channels[1],(channels[0]+channels[2]+1),new_channels[1],128.0);
//        divide(channels[2],(channels[1]+channels[0]+1),new_channels[2],128.0);
//        for(int i = 0;i<3;i++){
//            channels[i].convertTo(channels[i],CV_8UC1);
//            new_channels[i].convertTo(new_channels[i],CV_8UC1);
//        }
//        cv::merge(new_channels,img);

                for (int y=0;y<img.rows;y++)    // 23 lenvo
                {
                    uchar* imgP=img.ptr<uchar>(y);
                    uchar* dstP=img.ptr<uchar>(y);
                    for (int x=0;x<img.cols;x++)
                    {
                        int b0=imgP[3*x];
                        int g0=imgP[3*x+1];
                        int r0=imgP[3*x+2];

                        int b = static_cast<int>(b0*128/(g0+r0+1));
                        int g = static_cast<int>(g0*128/(b0+r0+1));
                        int r = static_cast<int>(r0*128/(g0+b0+1));

                        r = (r>255 ? 255 : (r<0? 0 : r));
                        g = (g>255 ? 255 : (g<0? 0 : g));
                        b = (b>255 ? 255 : (b<0? 0 : b));

                        dstP[3*x] = (uchar)b;
                        dstP[3*x+1] = (uchar)g;
                        dstP[3*x+2] = (uchar)r;
                    }
                }
        break;

    }

 /*
  * 遍历像素点 改变像素值
 */
//    cv::resize(img,img,cv::Size(),1.0/radio_resize,1.0/radio_resize,CV_INTER_LINEAR);          //缩小处理
//    switch (style_choose) {
//    case Nostalgic:
//        for (int y=0; y<img.rows; y++)
//        {
//            uchar* P0  = img.ptr<uchar>(y);
//            uchar* P1  = img.ptr<uchar>(y);
//            for (int x=0; x<img.cols; x++)
//            {
//                int B=P0[3*x];
//                int G=P0[3*x+1];
//                int R=P0[3*x+2];
//                int newB=static_cast<int>(0.272*R+0.534*G+0.131*B);
//                int newG=static_cast<int>(0.349*R+0.686*G+0.168*B);
//                int newR=static_cast<int>(0.393*R+0.769*G+0.189*B);
//                if(newB<0)newB=0;
//                if(newB>255)newB=255;
//                if(newG<0)newG=0;
//                if(newG>255)newG=255;
//                if(newR<0)newR=0;
//                if(newR>255)newR=255;
//                P1[3*x] = (uchar)newB;
//                P1[3*x+1] = (uchar)newG;
//                P1[3*x+2] = (uchar)newR;
//            }
//        }
//        break;
//    case Comic_strip:
//        for (int y=0; y<img.rows; y++)
//        {
//            uchar* P0  = img.ptr<uchar>(y);
//            uchar* P1  = img.ptr<uchar>(y);
//            for (int x=0; x<img.cols; x++)
//            {
//                int B=P0[3*x];
//                int G=P0[3*x+1];
//                int R=P0[3*x+2];
//                int newB=static_cast<int>(abs(B-G+B+R)*G/256);
//                int newG=static_cast<int>(abs(B-G+B+R)*R/256);
//                int newR=static_cast<int>(abs(G-B+G+R)*R/256);
//                if(newB<0)newB=0;
//                if(newB>255)newB=255;
//                if(newG<0)newG=0;
//                if(newG>255)newG=255;
//                if(newR<0)newR=0;
//                if(newR>255)newR=255;
//                P1[3*x] = (uchar)newB;
//                P1[3*x+1] = (uchar)newG;
//                P1[3*x+2] = (uchar)newR;
//            }
//        }
//        break;
//    case CASTING:
//        for (int y=0;y<img.rows;y++)
//        {
//            uchar* imgP=img.ptr<uchar>(y);
//            uchar* dstP=img.ptr<uchar>(y);
//            for (int x=0;x<img.cols;x++)
//            {
//                int b0=imgP[3*x];
//                int g0=imgP[3*x+1];
//                int r0=imgP[3*x+2];

//                int b = static_cast<int>(b0*128/(g0+r0+1));
//                int g = static_cast<int>(g0*128/(b0+r0+1));
//                int r = static_cast<int>(r0*128/(g0+b0+1));

//                r = (r>255 ? 255 : (r<0? 0 : r));
//                g = (g>255 ? 255 : (g<0? 0 : g));
//                b = (b>255 ? 255 : (b<0? 0 : b));

//                dstP[3*x] = (uchar)b;
//                dstP[3*x+1] = (uchar)g;
//                dstP[3*x+2] = (uchar)r;
//            }
//        }

//        break;
//    case FREEZING:
//        for (int y=0;y<img.rows;y++)
//        {
//            uchar* imgP=img.ptr<uchar>(y);
//            uchar* dstP=img.ptr<uchar>(y);
//            for (int x=0;x<img.cols;x++)
//            {
//                int b0=imgP[3*x];
//                int g0=imgP[3*x+1];
//                int r0=imgP[3*x+2];

//                int b = static_cast<int>((b0-g0-r0)*1.5);
//                int g = static_cast<int>((g0-b0-r0)*1.5);
//                int r = static_cast<int>((r0-g0-b0)*1.5);

//                r = (r>255 ? 255 : (r<0? -r : r));
//                g = (g>255 ? 255 : (g<0? -g : g));
//                b = (b>255 ? 255 : (b<0? -b : b));
//    // 			r = (r>255 ? 255 : (r<0? 0 : r));
//    // 			g = (g>255 ? 255 : (g<0? 0 : g));
//    // 			b = (b>255 ? 255 : (b<0? 0 : b));
//                dstP[3*x] = (uchar)b;
//                dstP[3*x+1] = (uchar)g;
//                dstP[3*x+2] = (uchar)r;
//            }
//        }
//        break;
//    }

//    cv::resize(img,img,cv::Size(),radio_resize,radio_resize,CV_INTER_LINEAR);  //还原

    return  img;
}
/*
*  冷暖色调：
* 暖色调 ：一幅暖色调的图片的时候通常是因为这张图色调偏黄。没有黄色的通道，但红色和绿色混合起来就是黄色，增加这两个通道值，然后蓝色通道值不变。
* 冷色调 ：冷色调的图片应该就是偏蓝色。该方法只增加蓝色通道的值，红色和绿色的值不变。
*/
cv::Mat facedeal::WarmCold_image(cv::Mat image,int value)
{
    if(!value)
        return image;
    int radio_resize = 1;                                    // 模糊处理（缩小增加帧率）
    cv::Mat img;
    img = image.clone();
    std::vector<cv::Mat> channels;   // b g r    单通道操作比像素点遍历快200帧！
    cv::split(img, channels);
    if(value > 0){
        channels[1] += value;
        channels[2] += value;
    }
    else
        channels[0] -= value;

    cv::merge(channels,img);
//    cv::resize(img,img,cv::Size(),1.0/radio_resize,1.0/radio_resize,CV_INTER_LINEAR);          //缩小处理
//    if(value > 0)   //warm
//    {
//        for (int y=0; y<img.rows; y++)
//        {
//            uchar* P0  = img.ptr<uchar>(y);
//            uchar* P1  = img.ptr<uchar>(y);
//            for (int x=0; x<img.cols; x++)
//            {
//                int G=P0[3*x+1];
//                int R=P0[3*x+2];
//                int newG=G+value;
//                int newR=R+value;
//                if(newG<0)newG=0;
//                if(newG>255)newG=255;
//                if(newR<0)newR=0;
//                if(newR>255)newR=255;
//                P1[3*x+1] = (uchar)newG;
//                P1[3*x+2] = (uchar)newR;
//            }
//        }
//    }
//    else                  //cold
//    {
//        for (int y=0; y<img.rows; y++)
//        {
//            uchar* P0  = img.ptr<uchar>(y);
//            uchar* P1  = img.ptr<uchar>(y);
//            for (int x=0; x<img.cols; x++)
//            {
//                int B=P0[3*x];
//                int newB=B-value;
//                if(newB<0)newB=0;
//                if(newB>255)newB=255;

//                P1[3*x] = (uchar)newB;
//            }
//        }
//    }
//    cv::resize(img,img,cv::Size(),radio_resize,radio_resize,CV_INTER_LINEAR);  //还原
    return  img;
}
/*
 * 1、浮雕效果  sobel算子实现
 * 2、
*/
cv::Mat facedeal::emboss_image(cv::Mat image,int kernel)
{
    // Compute Sobel X derivative

    if(!kernel)
        return image;
//    kernel = 2*kernel - 1;
    cv::Mat src,sobelX;
//    //cv::cvtColor(image,img,CV_BGR2GRAY);
//    cv::Sobel(image,  // input
//            sobelX,    // output
//            CV_8U,     // image type
//            1, 0,      // kernel specification
//            kernel,         // size of the square kernel
//            0.5,96); // scale and offset

    src = image.clone();
    for (int y=1; y<src.rows-1; y++)
    {
        uchar *p0 = image.ptr<uchar>(y);
        uchar *p1 = image.ptr<uchar>(y+1);

        uchar *q0 = src.ptr<uchar>(y);
        for (int x=1; x<src.cols-1; x++)
        {
            for (int i=0; i<3; i++)
            {
                int tmp0 = p1[3*(x+1)+i]-p0[3*(x-1)+i]+ 28 + kernel;//浮雕
                if (tmp0<0)
                    q0[3*x+i]=0;
                else if(tmp0>255)
                    q0[3*x+i]=255;
                else
                    q0[3*x+i]=tmp0;
            }
        }

    }
    return src;
    //return sobelX;
}
/*
 * 参考https://blog.csdn.net/shuiyixin/article/details/81095564
* 素描效果
* 第一步是需要先将图像转为灰度图像。然后应用中值滤波，中值滤波法是一种非线性平滑技术，它将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值平滑处理。
* 边缘检測算法实现。如经常使用的基于Sobel、Canny、Robert、Prewitt、Laplacian等算子的滤波器均能够实现这一操作
*
*/
cv::Mat facedeal::sketch_image(cv::Mat image,int value)   // value 0 - 255
{
    if(!value)
        return  image;
    cv::Mat Frame;
    Frame = image.clone();
    if(Frame.channels() == 3)
    {
        //error: (-215) scn == 3 || scn == 4 in function cvtColor 源图像不是BGR，则转换就会出问题
        cv::cvtColor(Frame, Frame, CV_BGR2GRAY);
    }
    //blur(Frame, Frame, cv::Size(3, 3), cv::Point(-1, -1));
     //cv::GaussianBlur(Frame, Frame, cv::Size(3, 3), 3, 3);

    cv::resize(Frame,Frame,cv::Size(),0.5,0.5,CV_INTER_LINEAR);
    cv::medianBlur(Frame, Frame, 3); //有点慢 消除椒盐噪声
    cv::resize(Frame,Frame,cv::Size(),2.0,2.0,CV_INTER_LINEAR);


    std::vector<std::vector<cv::Point>> contours;
    const int LAPLACIAN_FILTER_SIZE = 5;
    Laplacian(Frame, Frame, CV_8U, LAPLACIAN_FILTER_SIZE);
    cv::threshold(Frame, Frame, value, 255, CV_THRESH_BINARY_INV);
    //Frame = Frame < 100;  //颜色反转

    return Frame;
}

//卡通效果
cv::Mat facedeal::cartoon_image(cv::Mat image,int value)   // value 0 - 255
{
    if(!value)
        return  image;
    cv::Mat Frame,gray;
    Frame = image.clone();
    cv::Mat mask;
    mask = sketch_image(image,value);
    cv::Mat src0,src;
    Frame.copyTo(src0,mask);
    float radio_add = 3.0;
    cv::resize(src0,src0,cv::Size(),1.0/radio_add,1.0/radio_add,CV_INTER_LINEAR);

    bilateralFilter(src0, src, 6, 20, 20, cv::BORDER_DEFAULT);

    src = add_Contrast(src,30);

    cv::resize(src,src,cv::Size(),radio_add,radio_add,CV_INTER_LINEAR);


//    cv::Mat edges_Image;
//    cv::Canny(Frame, edges_Image, CV_8U, 5);

//    edges_Image = edges_Image <100; //颜色反转


    //加入双边滤波  但很耗时间

    return src;
}

//增加对比度
cv::Mat facedeal::add_Contrast(cv::Mat image,int value)
{
    if(!value)
        return image;
    cv::Mat img = image.clone();
    cv::cvtColor(img,img,CV_BGR2HLS);
    vector<cv::Mat> hls;
    cv::split(img,hls);
    hls[2] += value;
    cv::merge(hls,img);
    cv::cvtColor(img,img,CV_HLS2BGR);
    return img;
}
