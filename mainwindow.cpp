#include "mainwindow.h"
#include <QApplication>
#include <QLCDNumber>
#include "ui_mainwindow.h"
#include "global_variable.h"
//#include <QMutex>
#include "facedeal.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer=new QTimer(this);
    thread1 = new ThreadDeal;
    connect(timer,SIGNAL(timeout()),this,SLOT(ShowImg_Label()));    //定时显示
    ui->pushButton_1->setText("hat0");
    ui->pushButton_2->setText("hat1");
    ui->pushButton_3->setText("glass0");
    ui->pushButton_4->setText("glass1");
    ui->pushButton_5->setText("cute0");
    ui->pushButton_6->setText("cute1");
    ui->pushButton_7->setText("mouth0");
    ui->pushButton_8->setText("mouth1");

}

//加的label显示图像
void MainWindow::ShowImg_Label()
{
//    facedeal deal;
//    while(1){
//      if(capture.isOpened())
//        break;
//    }
//    capture>>image;
//    //m.lock();

//    deal.face_detection(image);
//    deal.eyes_detection(image);
//    image = deal.skin_detection(image,crmax,crmin,cbmax,cbmin,beauty_value1,beauty_value2);
//    flag_dealok = 1;


    //ui->lcdNumber->display(fps);

    ui->label_6->setNum(fps);
    if(flag_dealok){

        cv::resize(image,image,cv::Size(800,600));
        if(image.channels() == 1)
            cvtColor( image, image, CV_GRAY2RGB);
        else
            cvtColor( image, image, CV_BGR2RGB);
        QImage img = QImage((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888 );//.rgbSwapped()
        ui->label->setPixmap(QPixmap::fromImage(img));
        //ui->label->resize(ui->label->size());
        flag_dealok = 0;
    }
//    else{
//        std::cout << "no image" <<std::endl;
//    }
}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_startButton_clicked()
{
    if(!capture.isOpened())
        capture.open(CAPTURE);
     if(!thread1->isRunning())
        thread1->start();
     timer->start(15);
     ui->startButton->setEnabled(0);          //不可用
     ui->stopButton->setEnabled(1);
     ui->pushButton_1->setEnabled(1);
     ui->pushButton_2->setEnabled(1);
     ui->pushButton_3->setEnabled(1);
     ui->pushButton_4->setEnabled(1);
     ui->pushButton_5->setEnabled(1);
     ui->pushButton_6->setEnabled(1);
     ui->pushButton_7->setEnabled(1);
     ui->pushButton_8->setEnabled(1);
     ui->pushButton_17->setEnabled(1);
     ui->pushButton_18->setEnabled(1);
     ui->beautySlider->setEnabled(1);
     ui->beautySlider_2->setEnabled(1);
     ui->label_2->setEnabled(1);
     ui->label_3->setEnabled(1);
     ui->label_4->setEnabled(1);
     ui->label_5->setEnabled(1);
     ui->label_6->setEnabled(1);

     ui->label_7->setEnabled(1);
     ui->pushButton_9->setEnabled(1);
     ui->pushButton_10->setEnabled(1);
     ui->pushButton_11->setEnabled(1);
     ui->pushButton_12->setEnabled(1);

     ui->pushButton_13->setEnabled(1);
     ui->pushButton_14->setEnabled(1);
     ui->pushButton_15->setEnabled(1);
     ui->pushButton_16->setEnabled(1);
}



void MainWindow::on_stopButton_clicked()
{

    timer->stop();         // 停止显示
    ui->startButton->setEnabled(1);
    ui->stopButton->setEnabled(0);
    ui->pushButton_1->setEnabled(0);
    ui->pushButton_2->setEnabled(0);
    ui->pushButton_3->setEnabled(0);
    ui->pushButton_4->setEnabled(0);
    ui->pushButton_5->setEnabled(0);
    ui->pushButton_6->setEnabled(0);
    ui->pushButton_7->setEnabled(0);
    ui->pushButton_8->setEnabled(0);
    ui->pushButton_17->setEnabled(1);
    ui->pushButton_18->setEnabled(1);
    ui->beautySlider->setEnabled(0);
    ui->beautySlider_2->setEnabled(0);

    ui->label_2->setEnabled(0);
    ui->label_3->setEnabled(0);
    ui->label_4->setEnabled(0);
    ui->label_5->setEnabled(0);
    ui->label_6->setEnabled(0);

    ui->label_7->setEnabled(0);
    ui->pushButton_9->setEnabled(0);
    ui->pushButton_10->setEnabled(0);
    ui->pushButton_11->setEnabled(0);
    ui->pushButton_12->setEnabled(0);

    ui->pushButton_13->setEnabled(0);
    ui->pushButton_14->setEnabled(0);
    ui->pushButton_15->setEnabled(0);
    ui->pushButton_16->setEnabled(0);
}





void MainWindow::on_beautySlider_sliderMoved(int position)
{
    beauty_value1 = position;
    std::cout << position <<std::endl;
}
void MainWindow::on_beautySlider_2_sliderMoved(int position)
{
    beauty_value2 = position;
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    hat_w = 0.1*arg1;
}

void MainWindow::on_spinBox_2_valueChanged(int arg1)
{
    hat_h =0.1*arg1;
}

void MainWindow::on_spinBox_3_valueChanged(int arg1)
{
    hat_pan_h = 0.1*arg1;
}

void MainWindow::on_spinBox_4_valueChanged(int arg1)
{
    Contrast = arg1;
    std::cout << Contrast <<std::endl;
}



void MainWindow::on_quitButton_clicked()
{
    capture.release();//释放内存；
    thread1->exit();       // 停止处理
    timer->stop();         // 停止显示
    QApplication::closeAllWindows();
}

void MainWindow::on_pushButton_1_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[0] = 1;
    else
        flag_mask[0] = 0;
}

void MainWindow::on_pushButton_2_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[1] = 1;
    else
        flag_mask[1] = 0;
}

void MainWindow::on_pushButton_3_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[2] = 1;
    else
        flag_mask[2] = 0;
}

void MainWindow::on_pushButton_4_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[3] = 1;
    else
        flag_mask[3] = 0;
}

void MainWindow::on_pushButton_5_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[4] = 1;
    else
        flag_mask[4] = 0;
}

void MainWindow::on_pushButton_6_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[5] = 1;
    else
        flag_mask[5] = 0;
}

void MainWindow::on_pushButton_7_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[6] = 1;
    else
        flag_mask[6] = 0;
}

void MainWindow::on_pushButton_8_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[7] = 1;
    else
        flag_mask[7] = 0;
}


void MainWindow::on_pushButton_9_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2){
        style_choose = FREEZING;
        ui->pushButton_10->setEnabled(0);
        ui->pushButton_11->setEnabled(0);
        ui->pushButton_12->setEnabled(0);
        ui->pushButton_13->setEnabled(0);
        ui->pushButton_14->setEnabled(0);
        ui->pushButton_15->setEnabled(0);
        ui->pushButton_16->setEnabled(0);
    }
    else{
        style_choose = 0;
        ui->pushButton_13->setEnabled(1);
        ui->pushButton_10->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_12->setEnabled(1);
        ui->pushButton_14->setEnabled(1);
        ui->pushButton_15->setEnabled(1);
        ui->pushButton_16->setEnabled(1);
    }
}

void MainWindow::on_pushButton_10_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2){
        style_choose = CASTING;
        ui->pushButton_9->setEnabled(0);
        ui->pushButton_11->setEnabled(0);
        ui->pushButton_12->setEnabled(0);
        ui->pushButton_13->setEnabled(0);
        ui->pushButton_14->setEnabled(0);
        ui->pushButton_15->setEnabled(0);
        ui->pushButton_16->setEnabled(0);
    }
    else{
        style_choose = 0;
        ui->pushButton_13->setEnabled(1);
        ui->pushButton_9->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_12->setEnabled(1);
        ui->pushButton_14->setEnabled(1);
        ui->pushButton_15->setEnabled(1);
        ui->pushButton_16->setEnabled(1);

    }
}

void MainWindow::on_pushButton_11_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2){
        style_choose = Nostalgic;
        ui->pushButton_10->setEnabled(0);
        ui->pushButton_9->setEnabled(0);
        ui->pushButton_12->setEnabled(0);
        ui->pushButton_13->setEnabled(0);
        ui->pushButton_14->setEnabled(0);
        ui->pushButton_15->setEnabled(0);
        ui->pushButton_16->setEnabled(0);
    }
    else{
        style_choose = 0;
            ui->pushButton_13->setEnabled(1);
            ui->pushButton_10->setEnabled(1);
            ui->pushButton_9->setEnabled(1);
            ui->pushButton_12->setEnabled(1);
            ui->pushButton_14->setEnabled(1);
            ui->pushButton_15->setEnabled(1);
            ui->pushButton_16->setEnabled(1);

    }
}

void MainWindow::on_pushButton_12_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2){
        style_choose = Comic_strip;
        ui->pushButton_10->setEnabled(0);
        ui->pushButton_11->setEnabled(0);
        ui->pushButton_9->setEnabled(0);
        ui->pushButton_13->setEnabled(0);
        ui->pushButton_14->setEnabled(0);
        ui->pushButton_15->setEnabled(0);
        ui->pushButton_16->setEnabled(0);
       }
    else{
        style_choose = 0;
        ui->pushButton_13->setEnabled(1);
        ui->pushButton_10->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_9->setEnabled(1);
        ui->pushButton_14->setEnabled(1);
        ui->pushButton_15->setEnabled(1);
        ui->pushButton_16->setEnabled(1);

       }
}

//以下调节用到了推杆
int slider_choose = 0;
void MainWindow::on_horizontalScrollBar_sliderMoved(int position)
{
    if(slider_choose==13)
        color_temperature = position - 49;
    else if(slider_choose == 14)
        sobel_kernel = position;
    else if(slider_choose == 15)
        sketch_255 = position;
    else if(slider_choose == 16)
        cartoon_10 = position;

}

void MainWindow::on_pushButton_13_clicked()
{
    slider_choose = 13;
    static int nums = 0;
    nums ++;
    if(nums%2){
         ui->label_8->setEnabled(1);
         ui->label_8->setText("冷暖色调");
         ui->horizontalScrollBar->setEnabled(1);
         ui->horizontalScrollBar->setMinimum(0);
         ui->horizontalScrollBar->setMaximum(99);
         ui->horizontalScrollBar->setValue(49);
         ui->pushButton_9->setEnabled(0);
         ui->pushButton_10->setEnabled(0);
         ui->pushButton_11->setEnabled(0);
         ui->pushButton_12->setEnabled(0);
         ui->pushButton_14->setEnabled(0);
         ui->pushButton_15->setEnabled(0);
         ui->pushButton_16->setEnabled(0);
    }
    else{
        ui->label_8->setEnabled(0);
        ui->horizontalScrollBar->setEnabled(0);
        ui->horizontalScrollBar->setValue(0);
        color_temperature = 0;
        ui->pushButton_9->setEnabled(1);
        ui->pushButton_10->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_12->setEnabled(1);
        ui->pushButton_14->setEnabled(1);
        ui->pushButton_15->setEnabled(1);
        ui->pushButton_16->setEnabled(1);
    }
}

void MainWindow::on_pushButton_14_clicked()
{
    slider_choose = 14;
    static int nums = 0;
    nums ++;
    if(nums%2){
         sobel_kernel = 100;
         ui->label_8->setEnabled(1);
         ui->label_8->setText("浮雕变换");
         ui->horizontalScrollBar->setEnabled(1);
         ui->horizontalScrollBar->setMinimum(1);
         ui->horizontalScrollBar->setMaximum(200);
         ui->horizontalScrollBar->setValue(60);
         ui->pushButton_9->setEnabled(0);
         ui->pushButton_10->setEnabled(0);
         ui->pushButton_11->setEnabled(0);
         ui->pushButton_12->setEnabled(0);
         ui->pushButton_13->setEnabled(0);
         ui->pushButton_15->setEnabled(0);
         ui->pushButton_16->setEnabled(0);
    }
    else{
        ui->label_8->setEnabled(0);
        ui->horizontalScrollBar->setEnabled(0);
        ui->horizontalScrollBar->setValue(0);
        sobel_kernel = 0;
        ui->pushButton_9->setEnabled(1);
        ui->pushButton_10->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_12->setEnabled(1);
        ui->pushButton_13->setEnabled(1);
        ui->pushButton_15->setEnabled(1);
        ui->pushButton_16->setEnabled(1);
    }
}

void MainWindow::on_pushButton_15_clicked()
{
    slider_choose = 15;
    static int nums = 0;
    nums ++;
    if(nums%2){
         sketch_255 = 100;
         ui->label_8->setEnabled(1);
         ui->label_8->setText(" 素 描");
         ui->horizontalScrollBar->setEnabled(1);
         ui->horizontalScrollBar->setMinimum(1);
         ui->horizontalScrollBar->setMaximum(255);
         ui->horizontalScrollBar->setValue(100);
         ui->pushButton_9->setEnabled(0);
         ui->pushButton_10->setEnabled(0);
         ui->pushButton_11->setEnabled(0);
         ui->pushButton_12->setEnabled(0);
         ui->pushButton_13->setEnabled(0);
         ui->pushButton_14->setEnabled(0);
         ui->pushButton_16->setEnabled(0);
    }
    else{
        ui->label_8->setEnabled(0);
        ui->horizontalScrollBar->setEnabled(0);
        ui->horizontalScrollBar->setValue(0);
        sketch_255 = 0;
        ui->pushButton_9->setEnabled(1);
        ui->pushButton_10->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_12->setEnabled(1);
        ui->pushButton_13->setEnabled(1);
        ui->pushButton_14->setEnabled(1);
        ui->pushButton_16->setEnabled(1);
    }
}

void MainWindow::on_pushButton_16_clicked()
{
    slider_choose = 16;
    static int nums = 0;
    nums ++;
    if(nums%2){
         cartoon_10 = 1;
         ui->label_8->setEnabled(1);
         ui->label_8->setText(" 卡 通");
         ui->horizontalScrollBar->setEnabled(1);
         ui->horizontalScrollBar->setMinimum(1);
         ui->horizontalScrollBar->setMaximum(250);
         ui->horizontalScrollBar->setValue(100);
         ui->pushButton_9->setEnabled(0);
         ui->pushButton_10->setEnabled(0);
         ui->pushButton_11->setEnabled(0);
         ui->pushButton_12->setEnabled(0);
         ui->pushButton_13->setEnabled(0);
         ui->pushButton_14->setEnabled(0);
         ui->pushButton_15->setEnabled(0);
    }
    else{
        ui->label_8->setEnabled(0);
        ui->horizontalScrollBar->setEnabled(0);
        ui->horizontalScrollBar->setValue(0);
        cartoon_10 = 0;
        ui->pushButton_9->setEnabled(1);
        ui->pushButton_10->setEnabled(1);
        ui->pushButton_11->setEnabled(1);
        ui->pushButton_12->setEnabled(1);
        ui->pushButton_13->setEnabled(1);
        ui->pushButton_14->setEnabled(1);
        ui->pushButton_15->setEnabled(1);
    }
}

void MainWindow::on_pushButton_17_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[8] = 1;
    else
        flag_mask[8] = 0;
}

void MainWindow::on_pushButton_18_clicked()
{
    static int nums = 0;
    nums ++;
    if(nums%2)
        flag_mask[9] = 1;
    else
        flag_mask[9] = 0;
}
