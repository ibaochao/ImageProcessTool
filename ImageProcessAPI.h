#ifndef IMAGEPROCESSAPI_H
#define IMAGEPROCESSAPI_H

#include <QDebug>
#include <QVector>
#include <QImage>
#include <QColor>
#include <QPainter>
#include <QDateTime>
#include <QtMath>
#include <QRgb>

#include <iostream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


#define max2(a,b) (a>b?a:b)
#define max3(a,b,c) (a>b?max2(a,c):max2(b,c))
#define min2(a,b) (a<b?a:b)
#define min3(a,b,c) (a<b?min2(a,c):min2(b,c))


class ImageProcessAPI
{
public:
    ImageProcessAPI();

    static cv::Mat mQimageToMat(const QImage &image);  // QImage转cv::Mat
    static QImage mMatToQImage(const cv::Mat &mat);  // cv::Mat转QImage
    template<typename T>
    static T mPixelValueOutOfBoundProcess(T data, T range_left, T range_right);  // 像素值越界处理

    static QImage mGrayImage(const QImage &image);  // 灰度图
    static QImage mImageSharpen(const QImage &image);  // 锐化
    static QImage mGrayLevelHistogram(const QImage &image);  // 灰度直方图
    static QImage mImageQualizeHistogram(const QImage &image);  // 直方图均衡化

    static QImage mHorizontalFlip(const QImage &image);  // 水平翻转
    static QImage mVerticalFlip(const QImage &image);  // 垂直翻转
    static QImage mHorizontalVerticalFlip(const QImage &image);  // 水平垂直翻转

    static QImage mThreshold(const QImage &image);  // 二值化
    static QImage mInverseThreshold(const QImage &image);  // 反二值化
    static QImage mAdaptiveThreshold(const QImage &image);  // 自适应二值化
    static QImage mOldtySle(const QImage &image);  // 老照片
    static QImage mInverseColor(const QImage &image);  // 反色
    static QImage mWarmTone(const QImage &image, int offset);  // 暖色调
    static QImage mColdTone(const QImage &image, int offset);  // 冷色调

    static QImage mImageFusion(const QImage &image1, const QImage &image2);  // 图像融合
    static QImage mImageCartoon(const QImage &image);  // 漫画效果
    static QImage mImageFourierTransform(const QImage &image);  // 傅里叶变换
    static QImage mImageFourierTransformFilter(const QImage &image, bool flag);  // 傅里叶变换滤波

    static QImage mMeanFilter(const QImage &image);  // 均值滤波
    static QImage mMedianFilter(const QImage &image);  // 中值滤波
    static QImage mBoxFilter(const QImage &image);  // 方框滤波
    static double mGenerateGaussian(int x, int y, double sigma); // 生成高斯滤波核
    static void mGenerateGaussianFilter(double kernel[][5], double sigma); // 生成高斯滤波器
    static QImage mGaussianFilter(const QImage &image);  // 高斯滤波
    static double mGaussian(double x, double sigma);  // 高斯函数
    static double mBilateralFilterWeight(const QImage &inputImage, int x, int y, int centerX, int centerY, double sigmaS, double sigmaR);  // 计算双边滤波权重
    static QImage mBilateralFilter(const QImage &image);  // 双边滤波
    static QImage mSepFilter(const QImage &image);  // 可分离滤波

    static int mSobelOperator(const QImage &image, int x, int y);  // Sobel算子
    static QImage mSobelEdgeDetection(const QImage &image);  // sobel边缘检测
    static QImage mScharrEdgeDetection(const QImage &image);  // scharr边缘检测
    static QImage mCannyEdgeDetection(const QImage &image);  // canny边缘检测
    static QImage mPrewittEdgeDetection(const QImage &image);  // Prewitt边缘检测
    static QImage mLaplacianEdgeDetection(const QImage &image);  // laplacian边缘检测

    static QImage mBinaryzation(const QImage &image);  // 二值化
    static QImage mContourExtraction(const QImage &image);  // 常规轮廓提取
    static QImage mWatershedAlgorithmAutomatic(const QImage &image);  // 分水岭算法-自动
    static QImage mSuperpixelSegmentation(const QImage &image);  // 超像素分割SLIC
    static QImage mCornerHarris(const QImage &image);  // 角点检测

    static QImage mImageMosaic(const QImage &image, int blockSize);  // 马赛克
    static QImage mImageGaussianBlur(const QImage &image, int radius);  // 高斯模糊
    static QImage mImageScale(const QImage &image, float scale);  // 放缩

    static QImage mImageBrightnessContrastAdjust(const QImage &image, int brightness, int contrast);  // 亮度对比度调整
    static QImage mImageSaturationAdjust(const QImage &image, int saturation);  // 饱和度调整
    static QImage mImageTransparencyAdjust(const QImage &image, int transparency);  // 透明度调整

    static QImage mMorphErode(const QImage &image);  // 腐蚀
    static QImage mMorphDilate(const QImage &image);  // 膨胀
    static QImage mMorphOpen(const QImage &image);  // 开运算
    static QImage mMorphClose(const QImage &image);  // 闭运算
    static QImage mMorphTopHat(const QImage &image);  // 顶帽运算
    static QImage mMorphBlockHat(const QImage &image);  // 黑帽运算
    static QImage mMorphGradient(const QImage &image);  // 基本梯度运算


};

#endif // IMAGEPROCESSAPI_H
