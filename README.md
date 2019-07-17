# Face_Mapping
Graduation Design in UESTC in 2019


﻿###############################################################################

**主要任务：**
设计基于Raspberry Pi的实时人脸检测和处理的人脸贴图系统，主要任务：

1、实时的人脸检测；

2、基于QT的界面设计；

3、设计一套实时人脸处理算法，包括但不局限于skin变化、贴图变换等。
 
预期成果或目标：

1、完成硬体平台的搭建；

2、完成实时检测算法的设计；

3、完成图像处理算法的设计；

4、完成界面软件的设计；

5、完成论文的编撰。

结果：
完成了稳定、实时的人脸检测。

完成了快速、准确的脸部器官定位。

完成了稳定逼真、实时的人脸贴图处理。

完成了简易的美颜功能和一系列滤镜功能。

完成了树莓派端的移植和优化。

###############################################################################

**开发环境：**
ubuntu18.04

opencv2.4.10 -> opencv3.4.4(最终增加了跟踪部分，换成了opencv3)

c++11

qt 5.8


嵌入式端：

Raspberry Pi 3B+

###############################################################################

人脸检测 -> 器官定位　->　目标跟踪　->　美颜相机功能

人脸贴图 ->　透视变换 ->　图像融合

美颜功能 -> 肤色检测 ->　美白、保边滤波磨皮

风格变换 -> 像素操作

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071620033729.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNzY4Njc5,size_16,color_FFFFFF,t_70)
透视变换学习：https://blog.csdn.net/qq_32768679/article/details/85294100

人脸贴图处理：https://blog.csdn.net/qq_32768679/article/details/88425079

细节：

神经网络面部器官定位、跟踪＋检测定位人脸、透视变换贴图处理、Mask贴图、一系列的图像像素高效扫描

**效果图:**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716200402106.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNzY4Njc5,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071620042359.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNzY4Njc5,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716200603925.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNzY4Njc5,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716200620187.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNzY4Njc5,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716200742458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyNzY4Njc5,size_16,color_FFFFFF,t_70)


