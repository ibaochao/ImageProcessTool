#include "mainwindow.h"
#include "./ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // this->setWindowTitle("ImageProcessTool-https://github.com/ibaochao/ImageProcessTool");
    this->setWindowTitle("ImageProcessTool");
    this->setLayout(ui->verticalLayout_main);
    setFixedSize(1335, 870);

    statusBar = new QStatusBar(this);  // 创建状态栏
    fileInfoLabel = new QLabel(this);  // 创建用于显示文件属性的标签
    statusBar->addWidget(fileInfoLabel);  // 将标签添加到状态栏
    setStatusBar(statusBar);  // 设置主窗口的状态栏
}

MainWindow::~MainWindow()
{
    delete ui;
}


// 公共函数
/*****************************************************************************************/


// 放缩
QPixmap MainWindow::scaleImage(QImage img)
{
    if(pixmap != nullptr){
        delete pixmap;
        pixmap=nullptr;
    }
    pixmap = new QPixmap(QPixmap::fromImage(img));
    QSize qsize = pixmap->size();
    if(qsize.width()>640 || qsize.height()>480){
        *pixmap = pixmap->scaled(640, 480, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    return *pixmap;
}


// 释放原图
void MainWindow::releaseImg0()
{
    // 不为空则关闭资源
    if (m_img0 != nullptr) {
        delete m_img0;
        m_img0 = nullptr;
    }
    ui->label_img0->clear();
    ui->label_img0->setText("原图");
    ui->label_img0->setStyleSheet("background-color: rgb(177, 177, 177); font-size:18pt");
    ui->label_img0->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
}


// 释放处理后图
void MainWindow::releaseImg1()
{
    // 不为空则关闭资源
    if (m_img1 != nullptr) {
        delete m_img1;
        m_img1 = nullptr;
    }
    ui->label_img1->clear();
    ui->label_img1->setText("处理后");
    ui->label_img1->setStyleSheet("background-color: rgb(177, 177, 177); font-size:18pt");
    ui->label_img1->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

}


// 显示处理图
void MainWindow::showProcessedImage()
{
    if (m_img1 != nullptr) {
        ui->label_img1->setPixmap(scaleImage(*m_img1));
        ui->label_img1->setAlignment(Qt::AlignCenter);
        update();
    }
}


// 显示图片相关信息
void MainWindow::showImageInfo(QString imagePath)
{
    QFileInfo imageInfo(imagePath);
    float imageSize = imageInfo.size() / 1024;  // B->KB
    QSize resolution = m_img0->size();
    QString iamgeProperties;
    if(imageSize > 1024){
        iamgeProperties = QString("原图大小: %1 MB    分辨率(宽度x高度): %2 x %3    最后修改时间: %4    路径: %5")
                              .arg(QString::number(imageSize / 1024, 'f', 2))
                              .arg(resolution.width())
                              .arg(resolution.height())
                              .arg(imageInfo.lastModified().toString("yyyy-MM-dd hh:mm:ss"))
                              .arg(imagePath);
    }else{
        iamgeProperties = QString("原图大小: %1 KB    分辨率(宽度x高度): %2 x %3    最后修改时间: %4    路径: %5")
                              .arg(QString::number(imageSize, 'f', 2))
                              .arg(resolution.width())
                              .arg(resolution.height())
                              .arg(imageInfo.lastModified().toString("yyyy-MM-dd hh:mm:ss"))
                              .arg(imagePath);
    }
    // 更新状态栏标签以显示文件属性
    fileInfoLabel->setText(iamgeProperties);
    fileInfoLabel->setStyleSheet("background-color: rgb(56, 56, 56); font-size:11pt; color:white");
    fileInfoLabel->setAlignment(Qt::AlignLeft);
}


// 打开保存关闭
/*****************************************************************************************/


// 打开图片
void MainWindow::on_pushButton_open_clicked()
{
    // 选择一张图片
    QString defaultPicturePath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/Pictures";
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), defaultPicturePath, tr("Images (*.png *.jpg *.jpeg *.bmp)"));
    if (filename.isEmpty()) {
        QMessageBox::warning(this, tr("警告信息"), tr("未选择图片!"), QMessageBox::Ok);
        return;
    }

    // 释放资源
    releaseImg0();
    releaseImg1();

    m_img0 = new QImage(filename);

    // 放缩后显示
    ui->label_img0->setPixmap(scaleImage(*m_img0));
    ui->label_img0->setAlignment(Qt::AlignCenter);

    // 在状态栏显示文件相关信息
    showImageInfo(filename);

    update();
}


// 保存图片
void MainWindow::on_pushButton_save_clicked()
{
    if (m_img1 == nullptr) {
        QMessageBox::warning(this, tr("警告信息"), tr("右图为空, 不可保存!"), QMessageBox::Ok);
        return;
    }
    QString defaultPicturePath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/Pictures";
    QString filename = QFileDialog::getSaveFileName(this, tr("Save Image"), defaultPicturePath, tr("Images (*.png)"));  // 只能保存为png格式
    if (filename.isEmpty()) {
        QMessageBox::warning(this, tr("警告信息"), tr("文件名无效!"), QMessageBox::Ok);
        return;
    }
    if (!filename.contains(".png")) {
        filename = filename + ".png";
    }
    m_img1->save(filename);
    QMessageBox::information(this, tr("提示信息"), tr("图像保存成功!"), QMessageBox::Ok);
}


// 关闭图片
void MainWindow::on_pushButton_close_clicked()
{
    if (m_img0 != nullptr) {
        delete m_img0;
        m_img0 = nullptr;
    }else{
        QMessageBox::warning(this, tr("提示信息"), tr("无需清空!"), QMessageBox::Ok);
        return;
    }
    if (m_img1 != nullptr) {
        delete m_img1;
        m_img1 = nullptr;
    }
    if (pixmap != nullptr) {
        delete pixmap;
        pixmap = nullptr;
    }
    ui->label_img0->clear();
    ui->label_img0->setText("原图");
    ui->label_img0->setStyleSheet("background-color: rgb(177, 177, 177); font-size:18pt");
    ui->label_img0->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    ui->label_img1->clear();
    ui->label_img1->setText("处理后");
    ui->label_img1->setStyleSheet("background-color: rgb(177, 177, 177); font-size:18pt");
    ui->label_img1->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    fileInfoLabel->clear();

    QMessageBox::information(this, tr("提示信息"), tr("已清空!"), QMessageBox::Ok);
}


// 灰度图翻转
/*****************************************************************************************/


// 灰度图
void MainWindow::on_pushButton_hdt_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mGrayImage(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<灰度图>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 锐化
void MainWindow::on_pushButton_shapness_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageSharpen(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<锐化>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 灰度直方图
void MainWindow::on_pushButton_hdzft_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mGrayLevelHistogram(*m_img0));
        // showProcessedImage();
        if (m_img1 != nullptr) {  // 此处要单独处理，不能直接调用，不然图很窄
            if(pixmap != nullptr){
                delete pixmap;
                pixmap=nullptr;
            }
            pixmap = new QPixmap(QPixmap::fromImage(*m_img1));
            *pixmap = pixmap->scaled(480, 360, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);  // 不能用Qt::KeepAspectRatio否则图很小

            ui->label_img1->setPixmap(*pixmap);
            ui->label_img1->setAlignment(Qt::AlignCenter);
            update();
        }else{
            return;
        }

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<灰度直方图>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 直方图均衡化
void MainWindow::on_pushButton_zftjhh_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageQualizeHistogram(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<直方图均衡化>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 水平翻转
void MainWindow::on_pushButton_spf_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mHorizontalFlip(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<水平翻转>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 垂直翻转
void MainWindow::on_pushButton_czf_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mVerticalFlip(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<垂直翻转>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 水平垂直翻转
void MainWindow::on_pushButton_spczf_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mHorizontalVerticalFlip(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<水平垂直翻转>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 色调处理
/*****************************************************************************************/


// 二值化
void MainWindow::on_pushButton_ez_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mThreshold(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<二值化>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 反二值化
void MainWindow::on_pushButton_fez_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mInverseThreshold(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<反二值化>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 自适应二值化
void MainWindow::on_pushButton_aez_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mAdaptiveThreshold(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<自适应二值化>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 老照片
void MainWindow::on_pushButton_lzp_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mOldtySle(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<老照片>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 反色
void MainWindow::on_pushButton_fanse_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mInverseColor(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<反色>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 暖色调
void MainWindow::on_pushButton_ns_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mWarmTone(*m_img0, 30));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<暖色调>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 冷色调
void MainWindow::on_pushButton_ls_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mColdTone(*m_img0, 30));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<冷色调>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 扩展操作
/*****************************************************************************************/


// 图像融合
void MainWindow::on_pushButton_fusion_clicked()
{
    // 图像处理
    if (m_img0 == nullptr) {
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开第一张图片再点此选择第二张图片!"), QMessageBox::Ok);
        return;
    }

    // 选择第二张图片
    QString defaultPicturePath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/Pictures";
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), defaultPicturePath, tr("Images (*.png *.jpg *.jpeg *.bmp)"));
    if (filename.isEmpty()) {
        QMessageBox::warning(this, tr("警告信息"), tr("未选择图片!"), QMessageBox::Ok);
        return;
    }
    QImage image2(filename);

    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageFusion(*m_img0, image2));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<图像融合>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 漫画效果
void MainWindow::on_pushButton_cartoon_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageCartoon(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<漫画效果>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 傅里叶变换
void MainWindow::on_pushButton_fly_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageFourierTransform(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<傅里叶变换>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 傅里叶变换低通滤波
void MainWindow::on_pushButton_flyd_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageFourierTransformFilter(*m_img0, false));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<傅里叶变换低通滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 傅里叶变换高通滤波
void MainWindow::on_pushButton_flyg_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageFourierTransformFilter(*m_img0, true));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<傅里叶变换高通滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 滤波处理
/*****************************************************************************************/


// 均值滤波
void MainWindow::on_pushButton_jzlb_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMeanFilter(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<均值滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 中值滤波
void MainWindow::on_pushButton_zzlb_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMedianFilter(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<中值滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 方框滤波
void MainWindow::on_pushButton_fklb_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mBoxFilter(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<方框滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 高斯滤波
void MainWindow::on_pushButton_gslb_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mGaussianFilter(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<高斯滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 双边滤波
void MainWindow::on_pushButton_sblb_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mBilateralFilter(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<双边滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 可分离滤波
void MainWindow::on_pushButton_kfllb_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mSepFilter(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<可分离滤波>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 边缘检测
/*****************************************************************************************/


// sobel边缘检测
void MainWindow::on_pushButton_sobel_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mSobelEdgeDetection(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<sobel边缘检测>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// scharr边缘检测
void MainWindow::on_pushButton_scharr_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mScharrEdgeDetection(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<scharr边缘检测>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// canny边缘检测
void MainWindow::on_pushButton_canny_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mCannyEdgeDetection(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<canny边缘检测>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// Prewitt边缘检测
void MainWindow::on_pushButton_prewitt_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mPrewittEdgeDetection(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<Prewitt边缘检测>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// laplacian边缘检测
void MainWindow::on_pushButton_laplacian_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mLaplacianEdgeDetection(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<laplacian边缘检测>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 轮廓提取
/*****************************************************************************************/


// 常规轮廓提取
void MainWindow::on_pushButton_cg_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mContourExtraction(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<常规轮廓提取>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 分水岭算法-自动
void MainWindow::on_pushButton_fsl_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mWatershedAlgorithmAutomatic(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<分水岭算法-自动>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 超像素分割SLIC 图片太大或内容不合适均会导致程序崩溃, 慎重使用
void MainWindow::on_pushButton_seg_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像预处理
        if(pixmap != nullptr){
            delete pixmap;
            pixmap=nullptr;
        }
        pixmap = new QPixmap(QPixmap::fromImage(*m_img0));
        QSize qsize = pixmap->size();
        if(qsize.width()>300 || qsize.height()>300){  // 只处理300像素以下图像, 因为算法特别费时
            *pixmap = pixmap->scaled(300, 300, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }
        QImage image(pixmap->toImage());

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mSuperpixelSegmentation(image));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<超像素分割SLIC>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 角点检测
void MainWindow::on_pushButton_jd_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mCornerHarris(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<角点检测>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 基本数值操作
/*****************************************************************************************/


// 马赛克
void MainWindow::on_pushButton_msk_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageMosaic(*m_img0, ui->lineEdit_msk->text().toInt()));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<马赛克>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 高斯模糊
void MainWindow::on_pushButton_mh_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageGaussianBlur(*m_img0, ui->lineEdit_mh->text().toInt()));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<高斯模糊>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 放缩
void MainWindow::on_pushButton_fs_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageScale(*m_img0, ui->lineEdit_fs->text().toFloat()));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<放缩>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 亮度调整
void MainWindow::on_horizontalSlider_ld_valueChanged(int value)
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageBrightnessContrastAdjust(*m_img0, value, ui->horizontalSlider_dbd->value()));  // 还需要获取对比度值
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<亮度调整>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 对比度调整
void MainWindow::on_horizontalSlider_dbd_valueChanged(int value)
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageBrightnessContrastAdjust(*m_img0, ui->horizontalSlider_ld->value(), value));  // 还需要获取亮度值
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<对比度调整>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 饱和度调整
void MainWindow::on_horizontalSlider_bhd_valueChanged(int value)
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理

        showProcessedImage();
        m_img1 = new QImage(ImageProcessAPI::mImageSaturationAdjust(*m_img0, value));
        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<饱和度调整>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 透明度调整
void MainWindow::on_horizontalSlider_tmd_valueChanged(int value)
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mImageTransparencyAdjust(*m_img0, value));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<透明度调整>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 形态学处理
/*****************************************************************************************/


// 腐蚀
void MainWindow::on_pushButton_fushi_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphErode(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<腐蚀>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 膨胀
void MainWindow::on_pushButton_pz_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphDilate(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<膨胀>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 开运算
void MainWindow::on_pushButton_kys_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphOpen(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<开运算>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 闭运算
void MainWindow::on_pushButton_bys_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphClose(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<闭运算>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 顶帽运算
void MainWindow::on_pushButton_dmys_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphTopHat(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<顶帽运算>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 黑帽运算
void MainWindow::on_pushButton_hmys_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphBlockHat(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<黑帽运算>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


// 基本梯度运算
void MainWindow::on_pushButton_tdys_clicked()
{
    if (m_img0 != nullptr) {
        // 开始计时
        QDateTime startTime = QDateTime::currentDateTime();

        // 图像处理
        m_img1 = new QImage(ImageProcessAPI::mMorphGradient(*m_img0));
        showProcessedImage();

        // 结束计时
        QDateTime endTime = QDateTime::currentDateTime();
        qint64 costTime = startTime.msecsTo(endTime);
        qDebug().noquote() << endTime.toString("yyyy-MM-dd hh:mm:ss") << "<基本梯度运算>" << "操作用时:" << costTime << "ms";
    }else{
        QMessageBox::warning(this, tr("警告信息"), tr("请先打开一张图片!"), QMessageBox::Ok);
    }
}


