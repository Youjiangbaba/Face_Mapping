#-------------------------------------------------
#
# Project created by QtCreator 2019-01-07T14:09:39
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FaceDeal_Demo
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    threaddeal.cpp \
    facedeal.cpp \
    PerspectiveTrans.cpp \
    sport_tracker/sport_tracker.cpp \
    get_depthimg.cpp \
    facedeal_bymlp.cpp

HEADERS += \
        mainwindow.h \
    threaddeal.h \
    global_variable.h \
    facedeal.h \
    sport_tracker/sport_tracker.h \
    get_depthimg.h \
    my_define.h

FORMS += \
        mainwindow.ui

#opencv
INCLUDEPATH += /home/jiang/opencv/opencv-3.4.4/include/opencv \
                 /home/jiang/opencv/opencv-3.4.4/include/opencv2

LIBS += /usr/local/lib/*.so.3.4.4

DISTFILES += \
    haarcascade_frontalface_alt.xml
#
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
