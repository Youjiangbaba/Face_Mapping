#ifndef THREADDEAL_H
#define THREADDEAL_H
#include <QWidget>
#include <QThread>
#include <opencv2/opencv.hpp>

class ThreadDeal : public QThread
{
public:
    ThreadDeal();


protected:
    void run();
};


#endif // THREADDEAL_H
