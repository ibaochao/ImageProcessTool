#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QFileDialog>
#include <QStandardPaths>
#include <QDateTime>
#include <QRegularExpression>
#include <QIntValidator>
#include <QMessageBox>

#include "ImageProcessAPI.h"  // 导入图像处理API


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_open_clicked();  // 打开图片

    void on_pushButton_save_clicked();  // 保存图片

    void on_pushButton_close_clicked();  // 关闭图片

    void on_pushButton_hdt_clicked();  // 灰度图
    void on_pushButton_shapness_clicked();  // 锐化
    void on_pushButton_hdzft_clicked();  // 灰度直方图
    void on_pushButton_zftjhh_clicked();  // 直方图均衡化

    void on_pushButton_spf_clicked();  // 水平翻转
    void on_pushButton_czf_clicked();  // 垂直翻转
    void on_pushButton_spczf_clicked();  // 水平垂直翻转

    void on_pushButton_ez_clicked();  // 二值化
    void on_pushButton_fez_clicked();  // 反二值化
    void on_pushButton_aez_clicked();  // 自适应二值化
    void on_pushButton_lzp_clicked();  // 老照片
    void on_pushButton_fanse_clicked();  // 反色
    void on_pushButton_ns_clicked();  // 暖色调
    void on_pushButton_ls_clicked();  // 冷色调

    void on_pushButton_fusion_clicked();  // 图像融合
    void on_pushButton_cartoon_clicked();  // 漫画效果
    void on_pushButton_fly_clicked();  // 傅里叶变换
    void on_pushButton_flyd_clicked();  // 傅里叶变换低通滤波
    void on_pushButton_flyg_clicked();  // 傅里叶变换高通滤波

    void on_pushButton_jzlb_clicked();  // 均值滤波
    void on_pushButton_zzlb_clicked();  // 中值滤波
    void on_pushButton_fklb_clicked();  // 方框滤波
    void on_pushButton_gslb_clicked();  // 高斯滤波
    void on_pushButton_sblb_clicked();  // 双边滤波
    void on_pushButton_kfllb_clicked();  // 可分离滤波

    void on_pushButton_sobel_clicked();  // sobel边缘检测
    void on_pushButton_scharr_clicked();  // scharr边缘检测
    void on_pushButton_canny_clicked();  // canny边缘检测
    void on_pushButton_prewitt_clicked();  // Prewitt边缘检测
    void on_pushButton_laplacian_clicked();  // laplacian边缘检测

    void on_pushButton_cg_clicked();  // 常规轮廓提取
    void on_pushButton_fsl_clicked();  // 分水岭算法-自动
    void on_pushButton_seg_clicked();  // 超像素分割SLIC
    void on_pushButton_jd_clicked();  // 角点检测

    void on_pushButton_msk_clicked();  // 马赛克
    void on_pushButton_mh_clicked();  // 高斯模糊
    void on_pushButton_fs_clicked();  // 放缩

    void on_horizontalSlider_ld_valueChanged(int value);  // 亮度调整
    void on_horizontalSlider_dbd_valueChanged(int value);  // 对比度调整
    void on_horizontalSlider_bhd_valueChanged(int value);  // 饱和度调整
    void on_horizontalSlider_tmd_valueChanged(int value);  // 透明度调整


    void on_pushButton_fushi_clicked();  // 腐蚀
    void on_pushButton_pz_clicked();  // 膨胀
    void on_pushButton_kys_clicked();  // 开运算
    void on_pushButton_bys_clicked();  // 闭运算
    void on_pushButton_dmys_clicked();  // 顶帽运算
    void on_pushButton_hmys_clicked();  // 黑帽运算
    void on_pushButton_tdys_clicked();  // 基本梯度运算

private:
    QPixmap scaleImage(QImage img);
    void releaseImg0();  // 释放原图
    void releaseImg1();  // 释放处理后图
    void showProcessedImage();  // 显示处理图
    void showImageInfo(QString imagePath);  // 显示图片相关信息

private:
    Ui::MainWindow *ui;
    QStatusBar *statusBar=nullptr;  // 状态栏
    QLabel *fileInfoLabel=nullptr;  // 文件信息

    QImage *m_img0=nullptr;  // 原图
    QImage *m_img1=nullptr;  // 处理后图
    QPixmap *pixmap=nullptr;  // 展示图
};
#endif // MAINWINDOW_H
