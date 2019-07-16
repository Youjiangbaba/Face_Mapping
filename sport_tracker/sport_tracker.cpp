#include "sport_tracker.h"
#include "opencv2/opencv.hpp"
using namespace cv;
sport_tracker::sport_tracker()
{

}


///*
//    算法：利用SURF特征，输入一组已知有目标的函数，得到一组置信度较高的特征点。
//    输入： 一组样本,测试图片
//    输出： 测试图片的特征点位置
//*/
// vector<cv::Point> sport_tracker::SURF_tracking(vector<cv::Mat> std_samples,cv::Mat image)
//{
//     static int start_flag = 0;
//     vector<cv::Mat> samples_gray;
//     vector<cv::KeyPoint> keyPoint[std_samples.size()];
//     for(int i = 0;i < std_samples.size();i++)
//     {
//         cvtColor(std_samples[i], samples_gray[i], CV_RGB2GRAY);
//         //提取特征点
//         cv::SurfFeatureDetector surfDetector(1000);  // 海塞矩阵阈值，在这里调整精度，值越大点越少，越精准
//         surfDetector.detect(std_samples[i], keyPoint[i]);
//     }
//     //下面进行特征点匹配，每个都要互相匹配到
//     for(int i = 0;i < std_samples.size();i++)
//     {
//         for(int j = i;j < std_samples.size();j++){
//             if(i != j){
//                 //特征点描述，为下边的特征点匹配做准备
//                 cv::SurfDescriptorExtractor SurfDescriptor;
//                 cv::Mat imageDesc1, imageDesc2;
//                 SurfDescriptor.compute(samples_gray[i], keyPoint[i], imageDesc1);
//                 SurfDescriptor.compute(samples_gray[j], keyPoint[j], imageDesc2);
//             }
//         }
//     }
//}

/*
    算法：存储目标矩阵的Mat
*/
 cv::Mat get_Targetdata(cv::Rect target,cv::Mat image)
 {
      cv::Mat img;
      cv::cvtColor(image,img,CV_BGR2GRAY);
      return img(target);
//     int *data = new int[5];
//     int sumpixels = 0;
//     cv::Mat img;
//     cv::cvtColor(image,img,CV_BGR2GRAY);
//     for(int i = target.x;i<(target.width + target.x);i++)
//        for(int j = target.y;j < (target.height + target.y);j++)
//        {
//            sumpixels += img.at<uchar>(j,i);
//        }
//     data[0] = target.x;
//     data[1] = target.y;
//     data[2] = target.width;
//     data[3] = target.height;
//     data[4] = sumpixels;
 }


/*
    算法：上次的目标矩阵，二值化筛选出最好的;针对每次目标大小相近且能二值化得出最优矩阵的的情况
    输入：标准目标截取的矩阵，标准目标的灰度图，; 需要判断的图像
    输出：目标矩阵
*/
cv::Rect best_pixelsrect(cv::Rect std_rect,cv::Mat std_target,cv::Mat image)
{
    cv::Mat img ;
    cv::cvtColor(image,img,CV_BGR2BGRA);
    //以下进行扫描：扫描算法？
    cv::Rect scan_rect;
    cv::Mat scan_target;
    cv::Mat err_target;
    cv::Scalar mean_err = 0,last_meaneerr;          //mean 的返回值
    int stepx = 3;
    int stepy = 2;
    for( ; ;){
        scan_rect = std_rect;
        scan_target = img(scan_rect);
        absdiff(std_target,scan_target,err_target);
        mean_err = mean(err_target);
        //if()
    }
//    for(int x = 0;x < image.cols-std_rect.width; x += stepx){
//        for(int y = 0;y < image.rows - std_rect.height; y += stepy){
//        }
//    }
}


//cv::Mat sport_tracker::trackingRect(cv::Mat insrc,cv::Mat inrect,int steps_update)
//{
//    // List of tracker types in OpenCV 3.4.1
//    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
//    // vector <string> trackerTypes(types, std::end(types));

//    // Create a tracker
//    string trackerType = trackerTypes[2];


//    Ptr<TrackerMIT> tracker;

//    #if (CV_MINOR_VERSION < 3)
//    {
//        tracker = Tracker::create(trackerType);
//    }
//    #else
//    {
//        if (trackerType == "BOOSTING")
//            tracker = TrackerBoosting::create();
//        if (trackerType == "MIL")
//            tracker = TrackerMIL::create();
//        if (trackerType == "KCF")
//            tracker = TrackerKCF::create();
//        if (trackerType == "TLD")
//            tracker = TrackerTLD::create();
//        if (trackerType == "MEDIANFLOW")
//            tracker = TrackerMedianFlow::create();
//        if (trackerType == "GOTURN")
//            tracker = TrackerGOTURN::create();
//        if (trackerType == "MOSSE")
//            tracker = TrackerMOSSE::create();
//        if (trackerType == "CSRT")
//            tracker = TrackerCSRT::create();
//    }
//    #endif
//    // Read video
//    VideoCapture video("videos/chaplin.mp4");

//    // Exit if video is not opened
//    if(!video.isOpened())
//    {
//        cout << "Could not read video file" << endl;
//        return 1;
//    }

//    // Read first frame
//    Mat frame;
//    bool ok = video.read(frame);

//    // Define initial bounding box
//    Rect2d bbox(287, 23, 86, 320);

//    // Uncomment the line below to select a different bounding box
//    // bbox = selectROI(frame, false);
//    // Display bounding box.
//    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );

//    imshow("Tracking", frame);
//    tracker->init(frame, bbox);

//    while(video.read(frame))
//    {
//        // Start timer
//        double timer = (double)getTickCount();

//        // Update the tracking result
//        bool ok = tracker->update(frame, bbox);

//        // Calculate Frames per second (FPS)
//        float fps = getTickFrequency() / ((double)getTickCount() - timer);

//        if (ok)
//        {
//            // Tracking success : Draw the tracked object
//            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
//        }
//        else
//        {
//            // Tracking failure detected.
//            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
//        }

//        // Display tracker type on frame
//        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

//        // Display FPS on frame
//        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

//        // Display frame.
//        imshow("Tracking", frame);

//        // Exit if ESC pressed.
//        int k = waitKey(1);
//        if(k == 27)
//        {
//            break;
//        }

//    }
//}


