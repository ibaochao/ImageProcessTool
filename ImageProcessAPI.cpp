#include "ImageProcessAPI.h"


ImageProcessAPI::ImageProcessAPI(){}


// 公共函数
/*****************************************************************************************/


// QImage转cv::Mat
cv::Mat ImageProcessAPI::mQimageToMat(const QImage &image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_ARGB32:  // 8-bit, 4 channel
    case QImage::Format_ARGB32_Premultiplied:  // 8-bit, 4 channel
    {
        // std::cout<<"QImage::Format_ARGB32 or QImage::Format_ARGB32_Premultiplied:"<<std::endl;
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        break;
    }
    case QImage::Format_RGB32:  // 8-bit, 4 channel
    {
        // std::cout<<"QImage::Format_RGB32:"<<std::endl;
        // 不去掉alpha通道
        // mat = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        // 去掉alpha通道
        cv::Mat mat0 = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        // 方式1
        cv::cvtColor(mat0, mat, cv::COLOR_BGRA2BGR);
        // cv::cvtColor(mat0, mat, cv::COLOR_BGRA2GRAY);
        // 方式2
        // std::vector<cv::Mat> channels;
        // split(mat0, channels);
        // channels.pop_back();
        // cv::merge(channels, mat);
        break;
    }
    case QImage::Format_RGB888:  // 8-bit, 3 channel
    {
        // std::cout<<"QImage::Format_RGB888:"<<std::endl;
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    }
    case QImage::Format_Indexed8:  // 8-bit, 1 channel
    {
        // std::cout<<"QImage::Format_Indexed8:"<<std::endl;
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
        break;
    }
    default:
    {
        // std::cout<<"QImage::default:"<<std::endl;
        qWarning() << "Unsupported image format";
    }
    }
    return mat;
}


// cv::Mat转QImageM
QImage ImageProcessAPI::mMatToQImage(const cv::Mat &mat)
{
    QImage image;
    switch (mat.type())
    {
    case CV_8UC4:  // 8-bit, 4 channel
    {
        // std::cout<<"cv::Mat CV_8UC4:"<<std::endl;
        image = QImage((const uchar*)mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
        break;
    }
    case CV_8UC3:  // 8-bit, 3 channel
    {
        // std::cout<<"cv::Mat CV_8UC3:"<<std::endl;
        image = QImage((const uchar*)mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
        image = image.rgbSwapped();
        break;
    }
    case CV_8UC1:  // 8-bit, 1 channel
    {
        // std::cout<<"cv::Mat CV_8UC1:"<<std::endl;
        // 方式1，只要转灰度图就发生程序崩溃，转其它颜色空间就不会，不知为何???
        // image = QImage((const uchar*)mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);  // 这种方式不行
        // 方式2，这种可以
        image = QImage(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);
        for(int i = 0; i < 256; i++){
            image.setColor(i, qRgb(i, i, i));
        }
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; ++row){
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        break;
    }
    default:
    {
        // std::cout<<"cv::Mat default:"<<std::endl;
    }
    }
    return image;
}


// 像素值越界处理
template<typename T>
T ImageProcessAPI::mPixelValueOutOfBoundProcess(T data, T range_left, T range_right)
{
    if (data > range_right) {
        return range_right;
    }
    if (data < range_left) {
        return range_left;
    }
    return data;
}


// 灰度图翻转
/*****************************************************************************************/


// 灰度图
QImage ImageProcessAPI::mGrayImage(const QImage &image)
{
#if 0
    // Qt函数实现
    QImage grayImage = image.convertToFormat(QImage::Format_Grayscale8);
    return grayImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::cvtColor(src, dist, cv::COLOR_BGR2GRAY);
    return mMatToQImage(dist);
#endif
}


// 锐化
QImage ImageProcessAPI::mImageSharpen(const QImage &image)
{
#if 0
    // Qt函数实现
    QImage imgCopy;
    int width = image.width();
    int height = image.height();
    int window[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(width, height, QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    QImage imgCopyrgbImg = QImage(image).convertToFormat(QImage::Format_RGB888);
    uint8_t *rgbImg = imgCopyrgbImg.bits();
    uint8_t *rgb = imgCopy.bits();

    int nRowBytes = (width * 24 + 31) / 32 * 4;
    int  lineNum_24 = 0;
    for (int x = 1; x < image.width(); x++) {
        for (int y = 1; y < image.height(); y++) {
            int sumR = 0;
            int sumG = 0;
            int sumB = 0;

            for (int m = x - 1; m <= x + 1; m++)
                for (int n = y - 1; n <= y + 1; n++) {
                    if (m >= 0 && m < width && n < height) {
                        lineNum_24 = n * nRowBytes;
                        sumR += rgbImg[lineNum_24 + m * 3] * window[n - y + 1][m - x + 1];
                        sumG += rgbImg[lineNum_24 + m * 3 + 1] * window[n - y + 1][m - x + 1];
                        sumB += rgbImg[lineNum_24 + m * 3 + 2] * window[n - y + 1][m - x + 1];
                    }
                }

            int old_r = rgbImg[lineNum_24 + x * 3];
            sumR += old_r;
            sumR = qBound(0, sumR, 255);

            int old_g = rgbImg[lineNum_24 + x * 3 + 1];
            sumG += old_g;
            sumG = qBound(0, sumG, 255);

            int old_b = rgbImg[lineNum_24 + x * 3 + 2];
            sumB += old_b;
            sumB = qBound(0, sumB, 255);
            lineNum_24 = y * nRowBytes;
            rgb[lineNum_24 + x * 3] = sumR;
            rgb[lineNum_24 + x * 3 + 1] = sumG;
            rgb[lineNum_24 + x * 3 + 2] = sumB;
        }
    }
    return imgCopy;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::Mat srcClone = src.clone();
    cv::Mat kernel;
    int percent = 30;  // [-100, 100]
    int type = 0;  // 0, 1, other
    switch (type)
    {
    case 0:
        kernel = (cv::Mat_<int>(3, 3) <<
                      0, -1, 0,
                  -1, 4, -1,
                  0, -1, 0
                  );
    case 1:
        kernel = (cv::Mat_<int>(3, 3) <<
                      -1, -1, -1,
                  -1, 8, -1,
                  -1, -1, -1
                  );
    default:
        kernel = (cv::Mat_<int>(3, 3) <<
                      0, -1, 0,
                  -1, 4, -1,
                  0, -1, 0
                  );
    }
    cv::filter2D(srcClone, srcClone, srcClone.depth(), kernel);
    dist = src + srcClone * 0.1 * percent;
    return mMatToQImage(dist);
#endif
}


// 灰度直方图
QImage ImageProcessAPI::mGrayLevelHistogram(const QImage &image)
{
#if 1
    // Qt函数实现
    QVector<int> histogram(256, 0);

    // 计算图像的灰度直方图
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            QColor color(image.pixel(x, y));
            int intensity = qGray(color.rgb());
            histogram[intensity]++;
        }
    }

    int maxValue = *std::max_element(histogram.constBegin(), histogram.constEnd());

    QImage histogramImage(256, maxValue, QImage::Format_RGB32);
    histogramImage.fill(Qt::white);

    QPainter painter(&histogramImage);
    painter.setPen(Qt::black);

    for (int i = 0; i < histogram.size(); ++i) {
        int height = histogram[i] * histogramImage.height() / maxValue;
        painter.drawLine(i, histogramImage.height(), i, histogramImage.height() - height);
    }
    histogramImage.save("C:Users/ouc/Pictures/histogramImage.png", "PNG");
    return histogramImage;
#else
    // OpeCV函数实现，暂时不想实现
#endif
}


// 直方图均衡化
QImage ImageProcessAPI::mImageQualizeHistogram(const QImage &image)
{
#if 1
    // Qt函数实现
    // 获取图像的大小
    int width = image.width();
    int height = image.height();

    // 计算像素总数
    int totalPixels = width * height;

    // 计算直方图
    int histogram[256] = {0};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            QColor pixelColor(image.pixel(x, y));
            int intensity = qRound(0.299 * pixelColor.red() + 0.587 * pixelColor.green() + 0.114 * pixelColor.blue());
            histogram[intensity]++;
        }
    }

    // 计算累积分布函数
    float cumulativeDistribution[256] = {0.0f};
    cumulativeDistribution[0] = static_cast<float>(histogram[0]) / totalPixels;
    for (int i = 1; i < 256; ++i) {
        cumulativeDistribution[i] = cumulativeDistribution[i - 1] + static_cast<float>(histogram[i]) / totalPixels;
    }

    // 对每个像素进行直方图均衡化
    QImage outputImage(image.size(), image.format());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            QColor pixelColor(image.pixel(x, y));
            int intensity = qRound(0.299 * pixelColor.red() + 0.587 * pixelColor.green() + 0.114 * pixelColor.blue());
            float newIntensity = 255.0f * cumulativeDistribution[intensity];
            outputImage.setPixel(x, y, qRgb(newIntensity, newIntensity, newIntensity));
        }
    }
    return outputImage;
#else
    // OpeCV函数实现，暂时不想实现

#endif
}


// 水平翻转
QImage ImageProcessAPI::mHorizontalFlip(const QImage &image)
{
#if 0
    // Qt函数实现
    QImage copyImage(QSize(image.width(), image.height()), QImage::Format_ARGB32);
    copyImage = image.mirrored(true, false);
    return copyImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::flip(src, dist, 1);
    return mMatToQImage(dist);
#endif
}


// 垂直翻转
QImage ImageProcessAPI::mVerticalFlip(const QImage &image)
{
#if 0
    // Qt函数实现
    QImage copyImage(QSize(image.width(), image.height()), QImage::Format_ARGB32);
    copyImage = image.mirrored(false, true);
    return copyImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::flip(src, dist, 0);
    return mMatToQImage(dist);
#endif
}


// 水平垂直翻转
QImage ImageProcessAPI::mHorizontalVerticalFlip(const QImage &image)
{
#if 0
    // Qt函数实现
    QImage copyImage(QSize(image.width(), image.height()), QImage::Format_ARGB32);
    copyImage = image.mirrored(true, true);
    return copyImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::flip(src, dist, -1);
    return mMatToQImage(dist);
#endif
}


// 色调处理
/*****************************************************************************************/


// 二值化
QImage ImageProcessAPI::mThreshold(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现
    return image;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat srcGray;
    cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    cv::Mat dist;
    cv::threshold(srcGray, dist, 127, 255, cv::THRESH_BINARY);  // 若大于thresh，则设置为maxval，否则设置为0。（常用）
    // cv::threshold(srcGray, dist, 127, 255, cv::THRESH_BINARY_INV);  // 若大于thresh，则设置为0，否则设置为maxval（反操作）
    // cv::threshold(srcGray, dist, 127, 255, cv::THRESH_TRUNC );  // 若大于thresh，则设置为thresh，否则保持不变。
    // cv::threshold(srcGray, dist, 127, 255, cv::THRESH_TOZERO );  // 若大于thresh，则保持不变，否则设置为0。
    // cv::threshold(srcGray, dist, 127, 255, cv::THRESH_TOZERO_INV );  // 若大于thresh，则设置为0，否则保持不变（反操作）
    return mMatToQImage(dist);
#endif
}


// 反二值化
QImage ImageProcessAPI::mInverseThreshold(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现
    return image;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat srcGray;
    cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    cv::Mat dist;
    cv::threshold(srcGray, dist, 127, 255, cv::THRESH_BINARY_INV);
    return mMatToQImage(dist);
#endif
}


// 自适应二值化
QImage ImageProcessAPI::mAdaptiveThreshold(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现
    return image;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat srcGray;
    cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    cv::Mat dist;
    // cv::adaptiveThreshold(srcGray, dist, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 10);  // 窗口均值阈值法
    // cv::adaptiveThreshold(srcGray, dist, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 10);
    cv::adaptiveThreshold(srcGray, dist, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 10);  // 高斯分布阈值法
    // cv::adaptiveThreshold(srcGray, dist, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 5, 10);
    return mMatToQImage(dist);
#endif
}


// 老照片
QImage ImageProcessAPI::mOldtySle(const QImage &image)
{
#if 1
    // Qt函数实现
    QImage imgCopy;
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(image).convertToFormat(QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    uint8_t *rgb = imgCopy.bits();
    if (nullptr == rgb) {
        return QImage();
    }
    int size = image.width() * image.height();
    for (int i = 0; i < size ; i++) {
        float r = 0.393 * rgb[i * 3] + 0.769 * rgb[i * 3 + 1] + 0.189 * rgb[i * 3 + 2];
        float g = 0.349 * rgb[i * 3] + 0.686 * rgb[i * 3 + 1] + 0.168 * rgb[i * 3 + 2];
        float b = 0.272 * rgb[i * 3] + 0.534 * rgb[i * 3 + 1] + 0.131 * rgb[i * 3 + 2];
        r = mPixelValueOutOfBoundProcess<float>(r, 0, 255);
        g = mPixelValueOutOfBoundProcess<float>(g, 0, 255);
        b = mPixelValueOutOfBoundProcess<float>(b, 0, 255);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g ;
        rgb[i * 3 + 2] = b  ;
    }
    return imgCopy;
#else
    // OpeCV函数实现
    // 没有必要
#endif
}


// 反色
QImage ImageProcessAPI::mInverseColor(const QImage &image)
{
#if 0
    // Qt函数实现
    QImage imgCopy;
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(image).convertToFormat(QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    uint8_t *rgb = imgCopy.bits();
    if (nullptr == rgb) {
        return QImage();
    }
    int size = image.width() * image.height();
    for (int i = 0; i < size ; i++) {
        rgb[i * 3] = 255 - rgb[i * 3] ;
        rgb[i * 3 + 1] = 255 - rgb[i * 3 + 1]  ;
        rgb[i * 3 + 2] = 255 - rgb[i * 3 + 2]  ;
    }
    return imgCopy;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist = src.clone();
    int row = src.rows;
    int col = src.cols;
    for (int i = 0; i < row; ++i)
    {
        uchar* s = src.ptr<uchar>(i);
        uchar* d = dist.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            d[j * 3] = 255 - s[j * 3];  // B通道
            d[j * 3 + 1] = 255 - s[j * 3 + 1];  // G通道
            d[j * 3 + 2] = 255 - s[j * 3 + 2];  // R通道
        }
    }
    return mMatToQImage(dist);
#endif
}


// 暖色调
QImage ImageProcessAPI::mWarmTone(const QImage &image, int offset=30)
{
#if 0
    // Qt函数实现
    QImage imgCopy;
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(image).convertToFormat(QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    uint8_t *rgb = imgCopy.bits();
    if (nullptr == rgb) {
        return QImage();
    }
    QColor frontColor;
    int size = image.width() * image.height();
    for (int i = 0; i < size ; i++) {
        int r = rgb[i * 3] + offset;
        int g = rgb[i * 3 + 1] + offset;
        int b = rgb[i * 3 + 2] - offset;

        rgb[i * 3] = r > 255 ? 255 : r;
        rgb[i * 3 + 1] = g > 255 ? 255 : g;
        rgb[i * 3 + 2] = b < 0 ? 0 : b;
    }
    return imgCopy;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist = src.clone();
    int row = src.rows;
    int col = src.cols;
    for (int i = 0; i < row; ++i)
    {
        uchar* s = src.ptr<uchar>(i);
        uchar* d = dist.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            d[j * 3] = mPixelValueOutOfBoundProcess<int>(s[j * 3] - offset, 0, 255);  // B通道
            d[j * 3 + 1] = mPixelValueOutOfBoundProcess<int>(s[j * 3 + 1] + offset, 0, 255);  // G通道
            d[j * 3 + 2] = mPixelValueOutOfBoundProcess<int>(s[j * 3 + 2] + offset, 0, 255);  // R通道
        }
    }
    return mMatToQImage(dist);
#endif
}


// 冷色调
QImage ImageProcessAPI::mColdTone(const QImage &image, int offset=30)
{
#if 0
    // Qt函数实现
    QImage imgCopy;
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(image).convertToFormat(QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    uint8_t *rgb = imgCopy.bits();
    if (nullptr == rgb) {
        return QImage();
    }
    QColor frontColor;
    int size = image.width() * image.height();

    for (int i = 0; i < size ; i++) {
        int r = rgb[i * 3] - offset;
        int g = rgb[i * 3 + 1] - offset;
        int b = rgb[i * 3 + 2] + offset;

        rgb[i * 3] = r < 0 ? 0 : r;
        rgb[i * 3 + 1] = g < 0 ? 0 : g;
        rgb[i * 3 + 2] = b > 255 ? 255 : b;
    }
    return imgCopy;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist = src.clone();
    int row = src.rows;
    int col = src.cols;
    for (int i = 0; i < row; ++i)
    {
        uchar* s = src.ptr<uchar>(i);
        uchar* d = dist.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            d[j * 3] = mPixelValueOutOfBoundProcess<int>(s[j * 3] + offset, 0, 255);  // B通道
            d[j * 3 + 1] = mPixelValueOutOfBoundProcess<int>(s[j * 3 + 1] - offset, 0, 255);  // G通道
            d[j * 3 + 2] = mPixelValueOutOfBoundProcess<int>(s[j * 3 + 2] - offset, 0, 255);  // R通道
        }
    }
    return mMatToQImage(dist);
#endif
}


// 扩展操作
/*****************************************************************************************/


// 图像融合
QImage ImageProcessAPI::mImageFusion(const QImage &image1, const QImage &image2)
{
#if 0
    // Qt函数实现，暂时不想实现
    return image;
#else
    // OpeCV函数实现
    cv::Mat src1 = mQimageToMat(image1);
    cv::Mat src2 = mQimageToMat(image2);
    if (!src1.data || !src2.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    if(src1.rows != src2.rows || src1.cols != src2.cols)
    {
        cv::resize(src2, src2, src1.size());
    }
    cv::Mat dist;
    double alpha = 0.5;
    cv::addWeighted(src1, alpha, src2, (1-alpha), 0, dist);
    // cv::addWeighted(src1, alpha, src2, 0.85, 0, dist);
    return mMatToQImage(dist);
#endif
}


// 漫画效果
QImage ImageProcessAPI::mImageCartoon(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;

    // 变量自定义 clevel阈值40-80，d阈值0-10，sigma阈值10-250，size阈值10-25 参考80, 5, 150, 20
    double clevel=80;
    int d=5;
    double sigma=150;
    int size=20;
    //（1）中值滤波
    cv::Mat m;
    cv::medianBlur(src, m, 7);
    //（2）提取轮廓
    cv::Mat c;
    clevel = cv::max(40., cv::min(80., clevel));
    cv::Canny(m, c, clevel, clevel *3);
    //（3）轮廓膨胀
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(c, c, k);
    //（4）图像反转
    c = c / 255;
    c = 1 - c;
    //（5）均值滤波
    cv::Mat cf;
    c.convertTo(cf, CV_32FC1);				// 类型转换
    cv::blur(cf, cf, cv::Size(5, 5));
    //（6）双边滤波
    cv::Mat srcb;
    d = cv::max(0, cv::min(10, d));
    sigma = cv::max(10., cv::min(250., sigma));
    cv::bilateralFilter(src, srcb, d, sigma, sigma);
    size = cv::max(10, cv::min(25, size));
    dist = srcb / size;
    dist = dist * size;
    //（7）通道合并
    cv::Mat c3;
    cv::Mat cannyChannels[] = { cf, cf, cf };
    cv::merge(cannyChannels, 3, c3);
    //（8）图像相乘
    cv::Mat temp;
    dist.convertTo(temp, CV_32FC3);		// 类型转换
    cv::multiply(temp, c3, temp);
    temp.convertTo(dist, CV_8UC3);			// 类型转换

    return mMatToQImage(dist);
#endif
}


// 傅里叶变换
QImage ImageProcessAPI::mImageFourierTransform(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }

    // 必须先转化为灰度图像
    cv::Mat srcGray;
    if (src.channels() == 3){
        cv::cvtColor(src, srcGray, cv::COLOR_RGB2GRAY);
    }
    else{
        srcGray = src.clone();
    }
    //将输入图像扩展到最佳尺寸，边界用0填充
    //离散傅里叶变换的运行速度与图像的大小有很大的关系，当图像的尺寸使2，3，5的整数倍时，计算速度最快
    //为了达到快速计算的目的，经常通过添加新的边缘像素的方法获取最佳图像尺寸
    //函数getOptimalDFTSize()用于返回最佳尺寸，copyMakeBorder()用于填充边缘像素
    cv::Mat padded;
    int opHeight = cv::getOptimalDFTSize(srcGray.rows);
    int opWidth = cv::getOptimalDFTSize(srcGray.cols);
    // qDebug() <<"原图像的行数："<<srcGray.rows;
    // qDebug() <<"原图像的列数："<<srcGray.cols;
    // qDebug() <<"为适应DFT运算调整后的行数："<<opHeight;
    // qDebug() <<"为适应DFT运算调整后的列数："<<opWidth;
    cv::copyMakeBorder(srcGray, padded, 0, opHeight - srcGray.rows, 0, opWidth - srcGray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    //为傅立叶变换的结果分配存储空间
    //将plannes数组组合成一个多通道的数组，两个同搭配，分别保存实部和虚部
    //傅里叶变换的结果使复数，这就是说对于每个图像原像素值，会有两个图像值
    //此外，频域值范围远远超过图象值范围，因此至少将频域储存在float中
    //所以我们将输入图像转换成浮点型，并且多加一个额外通道来存储复数部分
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    merge(planes, 2, complexI);
    // std::cout << complexI.size() << std::endl;
    // std::cout << planes->size() << std::endl;
    //进行离散傅立叶变换
    cv::dft(complexI, complexI);
    //将复数转化为幅值，保存在planes[0]
    split(complexI, planes);   // 将多通道分为几个单通道
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magnitudeImage = planes[0];

    //傅里叶变换的幅值达到不适合在屏幕上显示，因此我们用对数尺度来替换线性尺度
    //进行对数尺度logarithmic scale缩放
    magnitudeImage += cv::Scalar::all(1);     //所有的像素都加1
    log(magnitudeImage, magnitudeImage);      //求自然对数
    //剪切和重分布幅度图像限
    //如果有奇数行或奇数列，进行频谱裁剪
    magnitudeImage = magnitudeImage(cv::Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

    // ---- -------- 下面的是为了显示结果 ---------------
    // 一分为四，左上与右下交换，右上与左下交换
    // 重新排列傅里叶图像中的象限，使原点位于图像中心
    int cx = magnitudeImage.cols / 2;
    int cy = magnitudeImage.rows / 2;
    cv::Mat q0(magnitudeImage, cv::Rect(0, 0, cx, cy));   // ROI区域的左上
    cv::Mat q1(magnitudeImage, cv::Rect(cx, 0, cx, cy));  // ROI区域的右上
    cv::Mat q2(magnitudeImage, cv::Rect(0, cy, cx, cy));  // ROI区域的左下
    cv::Mat q3(magnitudeImage, cv::Rect(cx, cy, cx, cy)); // ROI区域的右下
    //交换象限（左上与右下进行交换）
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    //交换象限（右上与左下进行交换）
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    // 归一化
    cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);
    cv::Mat dist(magnitudeImage.size(), CV_8UC1);
    magnitudeImage.convertTo(dist, CV_8UC1, 255, 0);

    return mMatToQImage(dist);
#endif
}


// 傅里叶变换滤波
QImage ImageProcessAPI::mImageFourierTransformFilter(const QImage &image, bool flag)  // flag=false低通  flag=true高通
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }

    // 必须先转化为灰度图像
    cv::Mat srcGray;
    if (src.channels() == 3){
        cv::cvtColor(src, srcGray, cv::COLOR_RGB2GRAY);
    }
    else{
        srcGray = src.clone();
    }
    //（1）数据准备
    srcGray.convertTo(srcGray, CV_32F);			//数据格式转换
    std::vector<cv::Mat> channels;
    cv::split(srcGray, channels);  			//RGB通道分离
    cv::Mat image_B = channels[0];
    int m1 = cv::getOptimalDFTSize(image_B.rows);  		//选取最适合做fft的宽
    int n1 = cv::getOptimalDFTSize(image_B.cols);		//选取最适合做fft的高
    cv::Mat padded;		//填充
    cv::copyMakeBorder(image_B, padded, 0, m1 - image_B.rows, 0, n1 - image_B.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;

    //（2）傅里叶正变换
    //planes[0], planes[1]是实部和虚部
    cv::merge(planes, 2, complexI);  											//通道合并
    cv::dft(complexI, complexI, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);		//傅里叶正变换
    cv::split(complexI, planes);												//通道分离

    //由实部planes[0]和虚部planes[1]得到幅度谱mag和相位谱ph
    cv::Mat ph, mag, idft;
    cv::phase(planes[0], planes[1], ph);
    cv::magnitude(planes[0], planes[1], mag);

    //（3）重新排列傅里叶图像中的象限，使得原点位于图像中心
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));      //右上角图像
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));      //左下角图像
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));     //右下角图像

    //3.1、变换左上角和右下角象限
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    //3.2、变换右上角和左下角象限
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // cv::imshow("mag1", mag);
    //3.3、滤波器
    int value = flag ? 1 : 0;
    for (int i = 0; i < mag.cols;i++){
        for (int j = 0; j < mag.rows; j++){
            if (abs(i - mag.cols / 2) > mag.cols / 10 || abs(j - mag.rows / 2) > mag.rows / 10)
                mag.at<float>(j, i) = value;
        }
    }
    // cv::imshow("mag2", mag);

    //3.4、变换左上角和右下角象限
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    //3.5、变换右上角和左下角象限
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    //（4）傅里叶逆变换
    cv::polarToCart(mag, ph, planes[0], planes[1]);
    //由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
    cv::merge(planes, 2, idft);
    cv::dft(idft, idft, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    image_B = idft(cv::Rect(0, 0, srcGray.cols & -2, srcGray.rows & -2));
    image_B.copyTo(channels[0]);
    cv::Mat dist;
    cv::merge(channels, dist);
    dist.convertTo(dist, CV_8U);

    return mMatToQImage(dist);
#endif
}


// 滤波处理
/*****************************************************************************************/


// 均值滤波
QImage ImageProcessAPI::mMeanFilter(const QImage &image)
{
#if 1
    // Qt函数实现
    // 定义一个简单的滤波核，这里使用3x3的均值滤波核
    const int filterSize = 3;
    const int filter[filterSize][filterSize] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    QImage outputImage = image;
    for (int y = 1; y < image.height() - 1; ++y) {
        for (int x = 1; x < image.width() - 1; ++x) {
            int sumRed = 0, sumGreen = 0, sumBlue = 0;
            // 遍历滤波核
            for (int i = 0; i < filterSize; ++i) {
                for (int j = 0; j < filterSize; ++j) {
                    QColor color(image.pixel(x + i - 1, y + j - 1));
                    sumRed += color.red() * filter[i][j];
                    sumGreen += color.green() * filter[i][j];
                    sumBlue += color.blue() * filter[i][j];
                }
            }
            // 计算平均值
            int avgRed = sumRed / (filterSize * filterSize);
            int avgGreen = sumGreen / (filterSize * filterSize);
            int avgBlue = sumBlue / (filterSize * filterSize);
            // 将新值设置为输出图像中的像素
            QColor newColor(avgRed, avgGreen, avgBlue);
            outputImage.setPixel(x, y, newColor.rgb());
        }
    }
    return outputImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::blur(src, dist, cv::Size(3, 3));
    return mMatToQImage(dist);
#endif
}


// 中值滤波
QImage ImageProcessAPI::mMedianFilter(const QImage &image)
{
#if 1
    // Qt函数实现
    // 中值滤波核大小
    const int filterSize = 3;
    QImage outputImage = image;
    int halfSize = filterSize / 2;
    for (int y = halfSize; y < image.height() - halfSize; ++y) {
        for (int x = halfSize; x < image.width() - halfSize; ++x) {
            // 收集周围像素的颜色值
            QVector<int> redValues, greenValues, blueValues;
            for (int i = 0; i < filterSize; ++i) {
                for (int j = 0; j < filterSize; ++j) {
                    QColor color(image.pixel(x + i - halfSize, y + j - halfSize));
                    redValues.append(color.red());
                    greenValues.append(color.green());
                    blueValues.append(color.blue());
                }
            }
            // 对颜色值进行排序
            std::sort(redValues.begin(), redValues.end());
            std::sort(greenValues.begin(), greenValues.end());
            std::sort(blueValues.begin(), blueValues.end());
            // 选择排序后的中间值作为新的像素值
            int medianRed = redValues.at(redValues.size() / 2);
            int medianGreen = greenValues.at(greenValues.size() / 2);
            int medianBlue = blueValues.at(blueValues.size() / 2);
            // 更新输出图像中的像素值
            QColor newColor(medianRed, medianGreen, medianBlue);
            outputImage.setPixel(x, y, newColor.rgb());
        }
    }
    return outputImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::medianBlur(src, dist, 3);
    return mMatToQImage(dist);
#endif
}


// 方框滤波
QImage ImageProcessAPI::mBoxFilter(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::boxFilter(src, dist, src.depth(), cv::Size(3, 3));
    return mMatToQImage(dist);
#endif
}


// 生成高斯滤波核
double ImageProcessAPI::mGenerateGaussian(int x, int y, double sigma)
{
    return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
}

// 生成高斯滤波器
void ImageProcessAPI::mGenerateGaussianFilter(double kernel[][5], double sigma)
{
    // 高斯滤波核大小
    const int GSfilterSize = 5;
    double sum = 0.0;
    int halfSize = GSfilterSize / 2;

    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            kernel[x + halfSize][y + halfSize] = mGenerateGaussian(x, y, sigma);
            sum += kernel[x + halfSize][y + halfSize];
        }
    }

    for (int i = 0; i < GSfilterSize; ++i) {
        for (int j = 0; j < GSfilterSize; ++j) {
            kernel[i][j] /= sum;
        }
    }
}


// 高斯滤波
QImage ImageProcessAPI::mGaussianFilter(const QImage &image)
{
#if 1
    // Qt函数实现
    // 高斯滤波核大小
    const int GSfilterSize = 5;
    // sigma
    const int sigma = 5;

    QImage outputImage = image;
    double kernel[GSfilterSize][GSfilterSize];
    mGenerateGaussianFilter(kernel, sigma);
    int halfSize = GSfilterSize / 2;
    for (int y = halfSize; y < image.height() - halfSize; ++y) {
        for (int x = halfSize; x < image.width() - halfSize; ++x) {
            double sumRed = 0.0, sumGreen = 0.0, sumBlue = 0.0;

            for (int i = 0; i < GSfilterSize; ++i) {
                for (int j = 0; j < GSfilterSize; ++j) {
                    QColor color(image.pixel(x + i - halfSize, y + j - halfSize));
                    sumRed += color.red() * kernel[i][j];
                    sumGreen += color.green() * kernel[i][j];
                    sumBlue += color.blue() * kernel[i][j];
                }
            }
            // 更新输出图像中的像素值
            QColor newColor(qBound(0, static_cast<int>(sumRed), 255),
                            qBound(0, static_cast<int>(sumGreen), 255),
                            qBound(0, static_cast<int>(sumBlue), 255));
            outputImage.setPixel(x, y, newColor.rgb());
        }
    }
    return outputImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::GaussianBlur(src, dist, cv::Size(7, 7), 0, 0);
    return mMatToQImage(dist);
#endif
}


// 高斯函数
double ImageProcessAPI::mGaussian(double x, double sigma)
{
    return exp(-(x * x) / (2 * sigma * sigma));
}


// 计算双边滤波权重
double ImageProcessAPI::mBilateralFilterWeight(const QImage &inputImage, int x, int y, int centerX, int centerY, double sigmaS, double sigmaR)
{
    double spatialFactor = mGaussian(sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY)), sigmaS);

    QColor centerColor(inputImage.pixel(centerX, centerY));
    QColor currentColor(inputImage.pixel(x, y));
    double colorFactor = mGaussian(centerColor.red() - currentColor.red(), sigmaR) *
                         mGaussian(centerColor.green() - currentColor.green(), sigmaR) *
                         mGaussian(centerColor.blue() - currentColor.blue(), sigmaR);

    return spatialFactor * colorFactor;
}


// 双边滤波
QImage ImageProcessAPI::mBilateralFilter(const QImage &image)
{
#if 1
    // Qt函数实现
    QImage outputImage = image;

    // 双边滤波核大小
    const int filterSize = 5;
    int sigmaS = 2;  // 空间域标准差
    int sigmaR = 30;  // 灰度值域标准差

    int halfSize = filterSize / 2;

    for (int y = halfSize; y < image.height() - halfSize; ++y) {
        for (int x = halfSize; x < image.width() - halfSize; ++x) {
            double totalWeight = 0.0;
            double sumRed = 0.0, sumGreen = 0.0, sumBlue = 0.0;

            for (int i = -halfSize; i <= halfSize; ++i) {
                for (int j = -halfSize; j <= halfSize; ++j) {
                    double weight = mBilateralFilterWeight(image, x, y, x + i, y + j, sigmaS, sigmaR);
                    totalWeight += weight;

                    QColor color(image.pixel(x + i, y + j));
                    sumRed += color.red() * weight;
                    sumGreen += color.green() * weight;
                    sumBlue += color.blue() * weight;
                }
            }
            // 归一化并更新输出图像中的像素值
            sumRed /= totalWeight;
            sumGreen /= totalWeight;
            sumBlue /= totalWeight;
            QColor newColor(mPixelValueOutOfBoundProcess<int>(static_cast<int>(sumRed), 0, 255),
                            mPixelValueOutOfBoundProcess<int>(static_cast<int>(sumGreen), 0, 255),
                            mPixelValueOutOfBoundProcess<int>(static_cast<int>(sumBlue), 0, 255));
            outputImage.setPixel(x, y, newColor.rgb());
        }
    }
    return outputImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::bilateralFilter(src, dist, 5, 150, 150);
    return mMatToQImage(dist);
#endif
}


// 可分离滤波
QImage ImageProcessAPI::mSepFilter(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::Mat kx = (cv::Mat_<float>(1, 3) << 0, -1, 0);
    cv::Mat ky = (cv::Mat_<float>(1, 3) << -1, 0, -1);
    cv::sepFilter2D(src, dist, src.depth(), kx, ky);
    return mMatToQImage(dist);
#endif
}


// 边缘检测
/*****************************************************************************************/


// Sobel算子
int ImageProcessAPI::mSobelOperator(const QImage &image, int x, int y)
{
    int gx = 0, gy = 0;
    // Sobel算子
    int sobelX[3][3] = {{-1, 0, 1},
                        {-2, 0, 2},
                        {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1},
                        {0, 0, 0},
                        {1, 2, 1}};

    // 遍历Sobel算子的3x3邻域
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // 获取邻域内的像素值，超出边界的像素使用0代替
            int pixelX = mPixelValueOutOfBoundProcess<int>(x + i, 0, image.width() - 1);
            int pixelY = mPixelValueOutOfBoundProcess<int>(y + j, 0, image.height() - 1);
            QColor pixelColor(image.pixel(pixelX, pixelY));
            // 计算梯度值
            gx += sobelX[i + 1][j + 1] * pixelColor.red();
            gy += sobelY[i + 1][j + 1] * pixelColor.red();
        }
    }
    // 计算梯度的幅值
    int gradientMagnitude = qAbs(gx) + qAbs(gy);
    // 对梯度值进行归一化处理，确保在[0, 255]范围内
    gradientMagnitude = mPixelValueOutOfBoundProcess<int>(gradientMagnitude, 0, 255);
    return gradientMagnitude;
}


// sobel边缘检测
QImage ImageProcessAPI::mSobelEdgeDetection(const QImage &image)
{
#if 1
    // Qt函数实现
    QImage outputImage(image.size(), image.format());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            // 对每个像素应用Sobel算子
            int gradientMagnitude = mSobelOperator(image, x, y);
            // 将梯度值作为边缘强度，用灰度值表示
            outputImage.setPixelColor(x, y, QColor(gradientMagnitude, gradientMagnitude, gradientMagnitude));
        }
    }
    return outputImage;

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat src_Gray;
    cv::cvtColor(src, src_Gray, cv::COLOR_BGR2GRAY);  // 灰度化
    cv::Mat dist;
    // 方式1
    cv::Mat Sobel_X, Sobel_Y, Sobel_X_abs, Sobel_Y_abs;
    cv::Sobel(src_Gray, Sobel_X, src_Gray.depth(), 1, 0);  // 计算 x 轴方向
    cv::Sobel(src_Gray, Sobel_Y, src_Gray.depth(), 0, 1);  // 计算 y 轴方向
    cv::convertScaleAbs(Sobel_X, Sobel_X_abs);  // 取绝对值
    cv::convertScaleAbs(Sobel_Y, Sobel_Y_abs);
    cv::addWeighted(Sobel_X_abs, 0.5, Sobel_Y_abs, 0.5, 0, dist);  // 图像融合
    // 方式2
    // cv::Sobel(src_Gray, dist, src_Gray.depth(), 1, 1);
    return mMatToQImage(dist);
#endif
}


// scharr边缘检测
QImage ImageProcessAPI::mScharrEdgeDetection(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat src_Gray;
    cv::cvtColor(src, src_Gray, cv::COLOR_BGR2GRAY);  // 灰度化
    cv::Mat dist;
    // 方式1
    cv::Mat Sobel_X, Sobel_Y, Sobel_X_abs, Sobel_Y_abs;
    cv::Scharr(src_Gray, Sobel_X, src_Gray.depth(), 1, 0);  // 计算 x 轴方向
    cv::Scharr(src_Gray, Sobel_Y, src_Gray.depth(), 0, 1);  // 计算 y 轴方向
    cv::convertScaleAbs(Sobel_X, Sobel_X_abs);  // 取绝对值
    cv::convertScaleAbs(Sobel_Y, Sobel_Y_abs);
    cv::addWeighted(Sobel_X_abs, 0.5, Sobel_Y_abs, 0.5, 0, dist);  // 图像融合
    // 方式2报错 Error: Assertion failed (dx >= 0 && dy >= 0 && dx+dy == 1) in getScharrKernels
    // cv::Scharr(src_Gray, dist, src_Gray.depth(), 1, 1);
    return mMatToQImage(dist);
#endif
}


// canny边缘检测
QImage ImageProcessAPI::mCannyEdgeDetection(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat src_Gray;
    cv::cvtColor(src, src_Gray, cv::COLOR_BGR2GRAY);  // 灰度化
    cv::Mat dist;
    cv::Canny(src_Gray, dist, 10, 100);
    return mMatToQImage(dist);
#endif
}


// Prewitt边缘检测
QImage ImageProcessAPI::mPrewittEdgeDetection(const QImage &image)
{
#if 1
    // Qt函数实现
    // Prewitt算子
    int prewittKernelX[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };
    int prewittKernelY[3][3] = {
        {-1, -1, -1},
        { 0,  0,  0},
        { 1,  1,  1}
    };

    QImage outputImage = image;
    int width = image.width();
    int height = image.height();

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    QColor color(image.pixel(x + i - 1, y + j - 1));
                    int gray = color.red(); // 假设图像是灰度图像，直接取红色通道
                    gx += prewittKernelX[j][i] * gray;
                    gy += prewittKernelY[j][i] * gray;
                }
            }
            int magnitude = std::sqrt(gx * gx + gy * gy);
            // 应用阈值
            if (magnitude > 100) {
                outputImage.setPixel(x, y, qRgb(255, 255, 255)); // 白色表示边缘
            } else {
                outputImage.setPixel(x, y, qRgb(0, 0, 0)); // 黑色表示非边缘
            }
        }
    }
    return outputImage;

#else
    // OpeCV函数实现，暂时不想实现
#endif
}


// laplacian边缘检测
QImage ImageProcessAPI::mLaplacianEdgeDetection(const QImage &image)
{
#if 1
    // Qt函数实现
    // Laplacian算子
    const int laplacianKernel[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    QImage outputImage(image.size(), QImage::Format_RGB32);
    for (int y = 1; y < image.height() - 1; ++y)
    {
        for (int x = 1; x < image.width() - 1; ++x)
        {
            int sum = 0;
            // 应用Laplacian算子
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    QRgb pixel = image.pixel(x + j - 1, y + i - 1);
                    sum += qRed(pixel) * laplacianKernel[i][j];
                }
            }
            // 边缘值截断至0-255范围内
            // sum = qBound(0, sum, 255);
            sum = mPixelValueOutOfBoundProcess<int>(sum, 0, 255);
            outputImage.setPixel(x, y, qRgb(sum, sum, sum));
        }
    }
    return outputImage;
#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat src_Gray;
    cv::cvtColor(src, src_Gray, cv::COLOR_BGR2GRAY);  // 灰度化
    cv::Mat dist;
    cv::Laplacian(src_Gray, dist, src_Gray.depth());
    return mMatToQImage(dist);
#endif
}


// 轮廓提取
/*****************************************************************************************/


// 二值化
QImage ImageProcessAPI::mBinaryzation(const QImage &image)
{
#if 1
    // Qt函数实现
    QImage imgCopy;
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(image).convertToFormat(QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    uint8_t *rgb = imgCopy.bits();
    int newGray = 0;
    int gray = 0;
    int size = image.width() * image.height();
    for (int i = 0; i < size ; i++) {
        gray = (rgb[i * 3] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3;
        if (gray > 128)
            newGray = 255;
        else
            newGray = 0;
        rgb[i * 3] = newGray;
        rgb[i * 3 + 1] = newGray;
        rgb[i * 3 + 2] = newGray;
    }
    return imgCopy;
#else
    // OpeCV函数实现，暂时不想实现

#endif
}


// 常规轮廓提取
QImage ImageProcessAPI::mContourExtraction(const QImage &image)
{
#if 1
    // Qt函数实现
    int width = image.width();
    int height = image.height();
    int pixel[8];
    QImage binImg = mBinaryzation(image);
    QImage newImg = QImage(width, height, QImage::Format_RGB888);
    newImg.fill(Qt::white);

    uint8_t *rgb = newImg.bits();
    uint8_t *binrgb = binImg.bits();
    int nRowBytes = (width * 24 + 31) / 32 * 4;
    int  lineNum_24 = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            memset(pixel, 0, 8);
            lineNum_24 = y * nRowBytes;
            if (binrgb[lineNum_24 + x * 3] == 0) {
                rgb[lineNum_24 + x * 3] = 0;
                rgb[lineNum_24 + x * 3 + 1] = 0;
                rgb[lineNum_24 + x * 3 + 2] = 0;
                pixel[0] = binrgb[(y - 1) * nRowBytes + (x - 1) * 3];
                pixel[1] = binrgb[(y) * nRowBytes + (x - 1) * 3];
                pixel[2] = binrgb[(y + 1) * nRowBytes + (x - 1) * 3];
                pixel[3] = binrgb[(y - 1) * nRowBytes + (x) * 3];
                pixel[4] = binrgb[(y + 1) * nRowBytes + (x) * 3];
                pixel[5] = binrgb[(y - 1) * nRowBytes + (x + 1) * 3];
                pixel[6] = binrgb[(y) * nRowBytes + (x + 1) * 3];
                pixel[7] = binrgb[(y + 1) * nRowBytes + (x + 1) * 3];

                if (pixel[0] + pixel[1] + pixel[2] + pixel[3] + pixel[4] + pixel[5] + pixel[6] + pixel[7] == 0) {
                    rgb[lineNum_24 + x * 3] = 255;
                    rgb[lineNum_24 + x * 3 + 1] = 255;
                    rgb[lineNum_24 + x * 3 + 2] = 255;
                }

            }
        }
    }
    return newImg;
#else
    // OpeCV函数实现，暂时不想实现

#endif
}


// 分水岭算法-自动 需要用到的函数
cv::Vec3b RandomColor(int value)    //生成随机颜色函数
{
    value = value % 255;  //生成0~255的随机数
    cv::RNG rng;
    int aa = rng.uniform(0, value);
    int bb = rng.uniform(0, value);
    int cc = rng.uniform(0, value);
    return cv::Vec3b(aa, bb, cc);
}


// 分水岭算法-自动
QImage ImageProcessAPI::mWatershedAlgorithmAutomatic(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#elif 0
    // OpeCV函数实现 1
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    //（1）图像处理
    cv::Mat grayImage;
    cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);													//灰度化
    // cv::imshow("GRAY", grayImage);

    cv::threshold(grayImage, grayImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);					//二值化（使用大津法）
    // cv::imshow("OTSU", grayImage);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9), cv::Point(-1, -1));		//获取结构化元素
    cv::morphologyEx(grayImage, grayImage, cv::MORPH_CLOSE, kernel);									//闭运算
    // cv::imshow("CLOSE1", grayImage);

    //（2）二次处理
    cv::distanceTransform(grayImage, grayImage, cv::DIST_L2, cv::DIST_MASK_3, 5);		//距离变换
    cv::normalize(grayImage, grayImage, 0, 1, cv::NORM_MINMAX);							//由于变换后结果非常小，故需要归一化到[0-1]
    // cv::imshow("normalize", grayImage);

    grayImage.convertTo(grayImage, CV_8UC1);											//数据类型转换：8位无符号整型单通道：(0-255)
    cv::threshold(grayImage, grayImage, 0, 255, cv::THRESH_BINARY);	//（二次）二值化（使用大津法）
    // cv::imshow("threshold", grayImage);

    cv::morphologyEx(grayImage, grayImage, cv::MORPH_CLOSE, kernel);					//（二次）闭运算
    // cv::imshow("CLOSE2", grayImage);

    //（3）标记mark图像
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(grayImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(-1, -1));				//检测轮廓
    cv::Mat marks = cv::Mat::zeros(grayImage.size(), CV_32SC1);			//数据类型转换：32位有符号整型三通道（提高计算精度）
    for (size_t i = 0; i < contours.size(); i++)
    {
        //saturate_cast<uchar>(x)：
        //11、可以解决边界溢出问题。若像素值大于255，则赋值255；若像素值小于0，则赋值0。
        //22、为了区别不同区域，对每个区域进行编号：区域1、区域2、区域3...。将区域之间的分界处的值置为-1。
        cv::drawContours(marks, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i + 1)), 2);		//绘制轮廓
    }

    //（4）分水岭算法（提取分割目标）
    cv::Mat kernel0 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));		//获取结构化元素
    cv::morphologyEx(src, src, cv::MORPH_ERODE, kernel0);				//腐蚀：去掉原图中的噪声或不相关信息
    cv::watershed(src, marks);											//分水岭算法

    //（5）随机分配颜色（给每个轮廓）
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int r = cv::theRNG().uniform(0, 255);
        int g = cv::theRNG().uniform(0, 255);
        int b = cv::theRNG().uniform(0, 255);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));		//将元素添加到向量最后位置
    }

    //（6）对每一个区域进行颜色填充
    cv::Mat dist = cv::Mat::zeros(marks.size(), CV_8UC3);				//数据类型转换：8位无符号整型三通道
    int row = src.rows;
    int col = src.cols;
    int index = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            index = marks.at<int>(i, j);
            if (index > 0 && index <= contours.size())		//给每一个区域随机颜色
            {
                dist.at<cv::Vec3b>(i, j) = colors[index - 1];
            }
            else if (index == -1)		//区域之间的边界为-1，全白
            {
                dist.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }
            else						//只检测到一个轮廓，全黑
            {
                dist.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return mMatToQImage(dist);
#else
    // OpeCV函数实现 2
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }

    //（3）边缘检测
    cv::Mat rgb_image_blur, rgb_image_canny;
    cv::GaussianBlur(src, rgb_image_blur, cv::Size(5, 5), 0, 0);		//高斯滤波（去噪）
    cv::Canny(rgb_image_blur, rgb_image_canny, 10, 120, 3, false);			//边缘算子（提取边缘特征）
    // cv::imshow("blur", rgb_image_blur);
    // cv::imshow("binary", rgb_image_canny);

    //（4）轮廓检测
    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(rgb_image_canny, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point());
    cv::Mat imageContours = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat marks(src.size(), CV_32S);
    marks = cv::Scalar::all(0);
    int index = 0;
    int compCount = 0;
    for (; index >= 0; index = hierarchy[index][0], compCount++)
    {
        //对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
        drawContours(marks, contours, index, cv::Scalar::all(compCount + 1), 1, 8, hierarchy);
        drawContours(imageContours, contours, index, cv::Scalar(255), 1, 8, hierarchy);
    }
    cv::Mat marksShows;
    cv::convertScaleAbs(marks, marksShows);
    // cv::imshow("mark", marksShows);
    // cv::imshow("轮廓", imageContours);

    //（5）分水岭算法
    cv::watershed(src, marks);
    cv::Mat afterWatershed;
    cv::convertScaleAbs(marks, afterWatershed);
    // cv::imshow("watershed", afterWatershed);

    //（6）随机分配颜色（为每一个目标）
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int r = cv::theRNG().uniform(0, 255);
        int g = cv::theRNG().uniform(0, 255);
        int b = cv::theRNG().uniform(0, 255);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));		//将元素添加到向量最后位置
    }

    //（7）对每一个区域进行颜色填充
    cv::Mat dist = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < marks.rows; i++)
    {
        for (int j = 0; j < marks.cols; j++)
        {
            int index = marks.at<int>(i, j);
            if (marks.at<int>(i, j) == -1)
            {
                dist.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }
            else
            {
                dist.at<cv::Vec3b>(i, j) = RandomColor(index);
            }
        }
    }

    return mMatToQImage(dist);
#endif
}


// 超像素分割SLIC 需要用到的函数
// gamma
float gamma(float x)
{
    return x > 0.04045 ? powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92);
}


// 超像素分割SLIC 需要用到的函数
// gamma_XYZ2RGB
float gamma_XYZ2RGB(float x)
{
    return x > 0.0031308 ? (1.055f * powf(x, (1 / 2.4f)) - 0.055) : (x * 12.92);
}


// 超像素分割SLIC 需要用到的函数
// XYZ2RGB
void XYZ2RGB(float X, float Y, float Z, int* R, int* G, int* B)
{
    float RR, GG, BB;
    RR = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    GG = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    BB = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
    RR = gamma_XYZ2RGB(RR);
    GG = gamma_XYZ2RGB(GG);
    BB = gamma_XYZ2RGB(BB);
    RR = int(RR * 255.0 + 0.5);
    GG = int(GG * 255.0 + 0.5);
    BB = int(BB * 255.0 + 0.5);
    *R = RR;
    *G = GG;
    *B = BB;
}


// 超像素分割SLIC 需要用到的函数
// Lab2XYZ
void Lab2XYZ(float L, float a, float b, float* X, float* Y, float* Z)
{
    const float param_16116 = 16.0f / 116.0f;
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;

    float fX, fY, fZ;
    fY = (L + 16.0f) / 116.0;
    fX = a / 500.0f + fY;
    fZ = fY - b / 200.0f;
    if (powf(fY, 3.0) > 0.008856)
        *Y = powf(fY, 3.0);
    else
        *Y = (fY - param_16116) / 7.787f;
    if (powf(fX, 3) > 0.008856)
        *X = fX * fX * fX;
    else
        *X = (fX - param_16116) / 7.787f;
    if (powf(fZ, 3.0) > 0.008856)
        *Z = fZ * fZ * fZ;
    else
        *Z = (fZ - param_16116) / 7.787f;
    (*X) *= (Xn);
    (*Y) *= (Yn);
    (*Z) *= (Zn);
}


// 超像素分割SLIC 需要用到的函数
// RGB2XYZ
void RGB2XYZ(int R, int G, int B, float* X, float* Y, float* Z)
{
    float RR = gamma((float)R / 255.0f);
    float GG = gamma((float)G / 255.0f);
    float BB = gamma((float)B / 255.0f);
    *X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    *Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    *Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;
}


// 超像素分割SLIC 需要用到的函数
// XYZ2Lab
void XYZ2Lab(float X, float Y, float Z, float* L, float* a, float* b)
{
    const float param_13 = 1.0f / 3.0f;
    const float param_16116 = 16.0f / 116.0f;
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;

    float fX = Xn, fY = Yn, fZ = Zn;
    if (Y > 0.008856f)
        fY = pow(Y, param_13);
    else
        fY = 7.787f * Y + param_16116;
    *L = 116.0f * fY - 16.0f;
    *L = *L > 0.0f ? *L : 0.0f;
    if (X > 0.008856f)
        fX = pow(X, param_13);
    else
        fX = 7.787f * X + param_16116;
    if (Z > 0.008856)
        fZ = pow(Z, param_13);
    else
        fZ = 7.787f * Z + param_16116;
    *a = 500.0f * (fX - fY);
    *b = 200.0f * (fY - fZ);
}


// 超像素分割SLIC 需要用到的函数
// RGB2Lab
void RGB2Lab(int R, int G, int B, float* L, float* a, float* b)
{
    float X, Y, Z;
    RGB2XYZ(R, G, B, &X, &Y, &Z);
    XYZ2Lab(X, Y, Z, L, a, b);
}


// 超像素分割SLIC 需要用到的函数
// Lab2RGB
void Lab2RGB(float L, float a, float b, int* R, int* G, int* B)
{
    float X, Y, Z;
    Lab2XYZ(L, a, b, &X, &Y, &Z);
    XYZ2RGB(X, Y, Z, R, G, B);
}


// 超像素分割SLIC
QImage ImageProcessAPI::mSuperpixelSegmentation(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }

    std::vector<std::vector<std::vector<float>>> img;    //x,y,(L,a,b)
    int rows = src.rows;
    int cols = src.cols;
    int N = rows * cols;
    int K = 200;                        //K个超像素
    int M = 40;
    int S = (int)sqrt(N / K);           //以步距为S的距离划分超像素

    // cout << "rows:" << rows << " cols:" << cols << endl;
    // cout << "cluster num:" << K << endl;
    // cout << "S:" << S << endl;
    //RGB2Lab
    for (int i = 0; i < rows; i++)
    {
        std::vector<std::vector<float>> line;
        for (int j = 0; j < cols; j++)
        {
            std::vector<float> pixel;
            float L;
            float a;
            float b;
            RGB2Lab(src.at<cv::Vec3b>(i, j)[2], src.at<cv::Vec3b>(i, j)[1], src.at<cv::Vec3b>(i, j)[0], &L, &a, &b);
            pixel.push_back(L);
            pixel.push_back(a);
            pixel.push_back(b);
            line.push_back(pixel);
        }
        img.push_back(line);
    }
    qDebug() << "RGB2Lab is finished";
    // cout << "RGB2Lab is finished" << endl;
    //聚类中心，[x y l a b]
    std::vector<std::vector<float>> Cluster;
    //生成所有聚类中心
    for (int i = S / 2; i < rows; i += S)
    {
        for (int j = S / 2; j < cols; j += S)
        {
            std::vector<float> c;
            c.push_back((float)i);
            c.push_back((float)j);
            c.push_back(img[i][j][0]);
            c.push_back(img[i][j][1]);
            c.push_back(img[i][j][2]);
            Cluster.push_back(c);
        }
    }
    int cluster_num = Cluster.size();
    qDebug() << "init cluster is finished";
    // cout << "init cluster is finished" << endl;
    //获得最小梯度值作为新中心点
    for (int c = 0; c < cluster_num; c++)
    {
        int c_row = (int)Cluster[c][0];
        int c_col = (int)Cluster[c][1];
        //梯度以右侧和下侧两个像素点来计算，分别计算Lab三个的梯度来求和
        //需要保证当前点右侧和下侧是存在的点，否则就向左上移动来替代梯度值
        if (c_row + 1 >= rows)
        {
            c_row = rows - 2;
        }
        if (c_col + 1 >= cols)
        {
            c_col = cols - 2;
        }
        float c_gradient =
            img[c_row + 1][c_col][0] + img[c_row][c_col + 1][0] - 2 * img[c_row][c_col][0] +
            img[c_row + 1][c_col][1] + img[c_row][c_col + 1][1] - 2 * img[c_row][c_col][1] +
            img[c_row + 1][c_col][2] + img[c_row][c_col + 1][2] - 2 * img[c_row][c_col][2];
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int tmp_row = c_row + i;
                int tmp_col = c_col + j;
                if (tmp_row + 1 >= rows)
                {
                    tmp_row = rows - 2;
                }
                if (tmp_col + 1 >= cols)
                {
                    tmp_col = cols - 2;
                }
                float tmp_gradient =
                    img[tmp_row + 1][tmp_col][0] + img[tmp_row][tmp_col + 1][0] -
                    img[tmp_row][tmp_col][0] + img[tmp_row + 1][tmp_col][1] +
                    img[tmp_row][tmp_col + 1][1] - 2 * img[tmp_row][tmp_col][1] +
                    img[tmp_row + 1][tmp_col][2] + img[tmp_row][tmp_col + 1][2] -
                    img[tmp_row][tmp_col][2];
                if (tmp_gradient < c_gradient)
                {
                    Cluster[c][0] = (float)tmp_row;
                    Cluster[c][1] = (float)tmp_col;
                    Cluster[c][2] = img[tmp_row][tmp_col][0];
                    Cluster[c][3] = img[tmp_row][tmp_col][1];
                    Cluster[c][3] = img[tmp_row][tmp_col][2];
                    c_gradient = tmp_gradient;
                }
            }
        }
    }
    qDebug() << "move cluster is finished";
    // cout << "move cluster is finished";
    //创建一个dis的矩阵for each pixel = ∞
    std::vector<std::vector<double>> distance;
    for (int i = 0; i < rows; ++i)
    {
        std::vector<double> tmp;
        for (int j = 0; j < cols; ++j)
        {
            tmp.push_back(INT_MAX);
        }
        distance.push_back(tmp);
    }
    //创建一个dis的矩阵for each pixel = -1
    std::vector<std::vector<int>> label;
    for (int i = 0; i < rows; ++i)
    {
        std::vector<int> tmp;
        for (int j = 0; j < cols; ++j)
        {
            tmp.push_back(-1);
        }
        label.push_back(tmp);
    }
    qDebug() << "为每一个Cluster创建一个pixel集合";
    //为每一个Cluster创建一个pixel集合
    std::vector<std::vector<std::vector<int>>> pixel(Cluster.size());
    //核心代码部分，迭代计算, ****************************************************非常费时,图片太大需要时间也特别长,有可能几分钟才出结果
    for (int t = 0; t < 10; t++)
    {
        qDebug() << "iteration num: " << t+1;
        // cout << endl << "iteration num:" << t + 1 << "  ";
        //遍历所有的中心点,在2S范围内进行像素搜索
        int c_num = 0;
        for (int c = 0; c < cluster_num; c++)
        {
            if (c - c_num >= (cluster_num / 10))
            {
                // cout << "+";
                c_num = c;
            }
            int c_row = (int)Cluster[c][0];
            int c_col = (int)Cluster[c][1];
            float c_L = Cluster[c][2];
            float c_a = Cluster[c][3];
            float c_b = Cluster[c][4];
            for (int i = c_row - 2 * S; i <= c_row + 2 * S; i++)
            {
                if (i < 0 || i >= rows)
                {
                    continue;
                }
                for (int j = c_col - 2 * S; j <= c_col + 2 * S; j++)
                {
                    if (j < 0 || j >= cols)
                    {
                        continue;
                    }
                    float tmp_L = img[i][j][0];
                    float tmp_a = img[i][j][1];
                    float tmp_b = img[i][j][2];
                    double Dc = sqrt((tmp_L - c_L) * (tmp_L - c_L) + (tmp_a - c_a) * (tmp_a - c_a) +
                                     (tmp_b - c_b) * (tmp_b - c_b));
                    double Ds = sqrt((i - c_row) * (i - c_row) + (j - c_col) * (j - c_col));
                    double D = sqrt((Dc / (double)M) * (Dc / (double)M) + (Ds / (double)S) * (Ds / (double)S));
                    if (D < distance[i][j])
                    {
                        if (label[i][j] == -1)
                        {   //还没有被标记过
                            label[i][j] = c;
                            std::vector<int> point;
                            point.push_back(i);
                            point.push_back(j);
                            pixel[c].push_back(point);
                        }
                        else
                        {
                            int old_cluster = label[i][j];
                            std::vector<std::vector<int>>::iterator iter;
                            for (iter = pixel[old_cluster].begin(); iter != pixel[old_cluster].end(); iter++)
                            {
                                if ((*iter)[0] == i && (*iter)[1] == j)
                                {
                                    pixel[old_cluster].erase(iter);
                                    break;
                                }
                            }
                            label[i][j] = c;
                            std::vector<int> point;
                            point.push_back(i);
                            point.push_back(j);
                            pixel[c].push_back(point);
                        }
                        distance[i][j] = D;
                    }
                }
            }
        }
        qDebug() << "start update cluster";
        // cout << " start update cluster";
        for (int c = 0; c < Cluster.size(); c++)
        {
            int sum_i = 0;
            int sum_j = 0;
            int number = 0;
            for (int p = 0; p < pixel[c].size(); p++)
            {
                sum_i += pixel[c][p][0];
                sum_j += pixel[c][p][1];
                number++;
            }
            int tmp_i = (int)((double)sum_i / (double)number);
            int tmp_j = (int)((double)sum_j / (double)number);
            Cluster[c][0] = (float)tmp_i;
            Cluster[c][1] = (float)tmp_j;
            Cluster[c][2] = img[tmp_i][tmp_j][0];
            Cluster[c][3] = img[tmp_i][tmp_j][1];
            Cluster[c][4] = img[tmp_i][tmp_j][2];
        }
    }
    qDebug() << "导出Lab空间的矩阵";
    //导出Lab空间的矩阵
    std::vector<std::vector<std::vector<float>>> out_image = img;//x,y,(L,a,b)
    for (int c = 0; c < Cluster.size(); c++)
    {
        for (int p = 0; p < pixel[c].size(); p++)
        {
            out_image[pixel[c][p][0]][pixel[c][p][1]][0] = Cluster[c][2];
            out_image[pixel[c][p][0]][pixel[c][p][1]][1] = Cluster[c][3];
            out_image[pixel[c][p][0]][pixel[c][p][1]][2] = Cluster[c][4];
        }
        out_image[(int)Cluster[c][0]][(int)Cluster[c][1]][0] = 0;
        out_image[(int)Cluster[c][0]][(int)Cluster[c][1]][1] = 0;
        out_image[(int)Cluster[c][0]][(int)Cluster[c][1]][2] = 0;
    }
    qDebug() << "export image mat finished";
    // cout << endl << "export image mat finished" << endl;
    cv::Mat dist = src.clone();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float L = out_image[i][j][0];
            float a = out_image[i][j][1];
            float b = out_image[i][j][2];
            int R, G, B;
            Lab2RGB(L, a, b, &R, &G, &B);
            cv::Vec3b vec3b;
            vec3b[0] = B;
            vec3b[1] = G;
            vec3b[2] = R;
            dist.at<cv::Vec3b>(i, j) = vec3b;
        }
    }

    return mMatToQImage(dist);
#endif
}


// 角点检测
QImage ImageProcessAPI::mCornerHarris(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }

    // int thresh = 130;
    int thresh = 100;  // 值越小检测到的结果越多
    // int max_count = 255;
    cv::Mat img_gray;
    cv::cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);

    cv::Mat dst, norm_dst, normScaleDst;
    dst = cv::Mat::zeros(img_gray.size(), CV_32FC1);
    cv::cornerHarris(img_gray, dst, 2, 3, 0.04, cv::BORDER_DEFAULT);
    //最大最小值归一化[0, 255]
    cv::normalize(dst, norm_dst, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(norm_dst, normScaleDst);

    cv::Mat dist = src.clone();
    for (int row = 0; row < dist.rows; row++)
    {
        //定义每一行的指针
        uchar* currentRow = normScaleDst.ptr(row);
        for (int col = 0; col < dist.cols; col++)
        {
            int value = (int)*currentRow;
            if (value > thresh)
            {
                circle(dist, cv::Point(col, row), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
            }
            currentRow++;
        }
    }

    return mMatToQImage(dist);
#endif
}


// 基本数值操作
/*****************************************************************************************/


// 马赛克
QImage ImageProcessAPI::mImageMosaic(const QImage &image, int blockSize)
{
#if 1
    // Qt函数实现
    if (image.isNull() || blockSize <= 0) {
        return QImage(); // 返回空图片或处理错误
    }
    // 确保blockSize是偶数，并且不会使图像尺寸变得太小
    blockSize = (blockSize % 2 == 0) ? blockSize : blockSize + 1;
    if (image.width() < blockSize || image.height() < blockSize) {
        return image; // 如果blockSize太大，直接返回原图
    }
    // 计算新图片的尺寸
    int newWidth = image.width() / blockSize * blockSize;
    int newHeight = image.height() / blockSize * blockSize;

    QImage newImage(newWidth, newHeight, image.format());
    // 遍历每个块
    for (int y = 0; y < newHeight; y += blockSize) {
        for (int x = 0; x < newWidth; x += blockSize) {
            // 计算块的平均颜色
            QRgb averageColor = qRgb(0, 0, 0); // 初始化平均颜色为黑色
            int totalR = 0, totalG = 0, totalB = 0;
            int count = 0;
            for (int by = 0; by < blockSize && y + by < image.height(); ++by) {
                for (int bx = 0; bx < blockSize && x + bx < image.width(); ++bx) {
                    QRgb pixel = image.pixel(x + bx, y + by);
                    totalR += qRed(pixel);
                    totalG += qGreen(pixel);
                    totalB += qBlue(pixel);
                    ++count;
                }
            }
            if (count > 0) { // 确保count不是0，避免除以0
                averageColor = qRgb(totalR / count, totalG / count, totalB / count);
            }
            // 用平均颜色填充整个块
            for (int by = 0; by < blockSize && y + by < newImage.height(); ++by) {
                for (int bx = 0; bx < blockSize && x + bx < newImage.width(); ++bx) {
                    newImage.setPixel(x + bx, y + by, averageColor);
                }
            }
        }
    }
    return newImage;
#else
    // OpeCV函数实现，暂时不想实现

#endif
}


// 高斯模糊
QImage ImageProcessAPI::mImageGaussianBlur(const QImage &image, int radius)
{
#if 1
    // Qt函数实现
    QImage srcimage(image);
    if (srcimage.isNull() || radius <= 0)
        return QImage();

    QImage resultImage = srcimage;
    const int size = radius * 2 + 1;
    const int sigma = radius / 2;
    const double sigmaSq = sigma * sigma;
    QVector<double> kernel(size);

    // 构建高斯核
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i)
    {
        double value = exp(-(i * i) / (2 * sigmaSq)) / (sqrt(2 * M_PI) * sigma);
        kernel[i + radius] = value;
        sum += value;
    }

    // 归一化
    for (int i = 0; i < size; ++i)
    {
        kernel[i] /= sum;
    }

    // 水平方向模糊
    for (int y = 0; y < srcimage.height(); ++y)
    {
        for (int x = radius; x < srcimage.width() - radius; ++x)
        {
            double red = 0, green = 0, blue = 0;
            for (int i = -radius; i <= radius; ++i)
            {
                QRgb pixel = srcimage.pixel(x + i, y);
                red += qRed(pixel) * kernel[i + radius];
                green += qGreen(pixel) * kernel[i + radius];
                blue += qBlue(pixel) * kernel[i + radius];
            }
            resultImage.setPixel(x, y, qRgb(red, green, blue));
        }
    }

    // 垂直方向模糊
    for (int x = 0; x < srcimage.width(); ++x)
    {
        for (int y = radius; y < srcimage.height() - radius; ++y)
        {
            double red = 0, green = 0, blue = 0;
            for (int i = -radius; i <= radius; ++i)
            {
                QRgb pixel = resultImage.pixel(x, y + i);
                red += qRed(pixel) * kernel[i + radius];
                green += qGreen(pixel) * kernel[i + radius];
                blue += qBlue(pixel) * kernel[i + radius];
            }
            srcimage.setPixel(x, y, qRgb(red, green, blue));
        }
    }
    return srcimage;
#else
    // OpeCV函数实现，暂时不想实现

#endif
}


// 放缩 [0.1, 10.0] 最小1/10最大10
QImage ImageProcessAPI::mImageScale(const QImage &image, float scale)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    scale = mPixelValueOutOfBoundProcess<float>(scale, 0.1, 10.0);
    cv::Mat dist;
    // cv::resize(src, dist, cv::Size(0, 0), scale, scale, cv::INTER_NEAREST);  // 最近邻插值
    cv::resize(src, dist, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);  // 双线性插值（默认）
    // cv::resize(src, dist, cv::Size(0, 0), scale, scale, cv::INTER_CUBIC);  // 双三次插值
    return mMatToQImage(dist);
#endif
}


// 亮度对比度调整 // brightness(10): [0,100] contrast(0): [-100,100]
QImage ImageProcessAPI::mImageBrightnessContrastAdjust(const QImage &image, int brightness, int contrast)
{
#if 0
    // Qt函数实现
    QImage imgCopy;
    if (image.format() != QImage::Format_RGB888) {
        imgCopy = QImage(image).convertToFormat(QImage::Format_RGB888);
    } else {
        imgCopy = QImage(image);
    }
    uint8_t *rgb = imgCopy.bits();
    if (nullptr == rgb) {
        return QImage();
    }
    int r;
    int g;
    int b;
    int size = image.width() * image.height();
    for (int i = 0; i < size ; i++) {
        r = brightness * 0.1 * rgb[i * 3] - 150 + contrast;
        g = brightness * 0.1 * rgb[i * 3 + 1] - 150 + contrast;
        b = brightness * 0.1 * rgb[i * 3 + 2]  - 150 + contrast;
        r = qBound(0, r, 255);
        g = qBound(0, g, 255);
        b = qBound(0, b, 255);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return imgCopy;
#else
    // OpeCV函数实现，暂时不想实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    dist = cv::Mat::zeros(src.size(), src.type());		//新建空白模板：大小/类型与原图像一致，像素值全0。
    int height = src.rows;								//获取图像高度
    int width = src.cols;								//获取图像宽度
    float alpha = brightness / 10.0;							//亮度（0~1为暗，1~正无穷为亮）
    float beta = contrast;								//对比度

    cv::Mat template1;
    src.convertTo(template1, CV_32F);					//将CV_8UC1转换为CV32F1数据格式。
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            if (src.channels() == 3)
            {
                float b = template1.at<cv::Vec3f>(row, col)[0];		//获取通道的像素值（blue）
                float g = template1.at<cv::Vec3f>(row, col)[1];		//获取通道的像素值（green）
                float r = template1.at<cv::Vec3f>(row, col)[2];		//获取通道的像素值（red）

                //cv::saturate_cast<uchar>(vaule)：需注意，value值范围必须在0~255之间。
                dist.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha -150 + beta);		//修改通道的像素值（blue）
                dist.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha -150 + beta);		//修改通道的像素值（green）
                dist.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha -150 + beta);		//修改通道的像素值（red）
            }
            else if (src.channels() == 1)
            {
                float v = src.at<uchar>(row, col);											//获取通道的像素值（单）
                dist.at<uchar>(row, col) = cv::saturate_cast<uchar>(v * alpha + beta);		//修改通道的像素值（单）
                //saturate_cast<uchar>：主要是为了防止颜色溢出操作。如果color<0，则color等于0；如果color>255，则color等于255。
            }
        }
    }
    return mMatToQImage(dist);
#endif
}


// 饱和度调整 // saturation(0): [-100,100]
QImage ImageProcessAPI::mImageSaturationAdjust(const QImage &image, int saturation)
{
#if 0
    // Qt函数实现
    QImage imgCopy(image.width(), image.height(), QImage::Format_ARGB32);
    QColor oldColor;
    QColor newColor;
    int h,s,l;

    for(int x=0; x<imgCopy.width(); x++){
        for(int y=0; y<imgCopy.height(); y++){
            oldColor = QColor(image.pixel(x,y));

            newColor = oldColor.toHsl();
            h = newColor.hue();
            s = newColor.saturation() + saturation;
            l = newColor.lightness();

            s = qBound(0, s, 255);
            newColor.setHsl(h, s, l);
            imgCopy.setPixel(x, y, qRgb(newColor.red(), newColor.green(), newColor.blue()));
        }
    }
    return imgCopy;
#else
    // OpeCV函数实现，暂时不想实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;

    float Increment = saturation * 1.0f / 100;
    dist = src.clone();
    int row = src.rows;
    int col = src.cols;
    for (int i = 0; i < row; ++i)
    {
        uchar *t = dist.ptr<uchar>(i);
        uchar *s = src.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            uchar b = s[3 * j];
            uchar g = s[3 * j + 1];
            uchar r = s[3 * j + 2];
            float max = max3(r, g, b);
            float min = min3(r, g, b);
            float delta, value;
            float L, S, alpha;
            delta = (max - min) / 255;
            if (delta == 0)
                continue;
            value = (max + min) / 255;
            L = value / 2;
            if (L < 0.5)
                S = delta / value;
            else
                S = delta / (2 - value);
            if (Increment >= 0)
            {
                if ((Increment + S) >= 1)
                    alpha = S;
                else
                    alpha = 1 - Increment;
                alpha = 1 / alpha - 1;
                t[3 * j + 2] =static_cast<uchar>( r + (r - L * 255) * alpha);
                t[3 * j + 1] = static_cast<uchar>(g + (g - L * 255) * alpha);
                t[3 * j] = static_cast<uchar>(b + (b - L * 255) * alpha);
            }
            else
            {
                alpha = Increment;
                t[3 * j + 2] = static_cast<uchar>(L * 255 + (r - L * 255) * (1 + alpha));
                t[3 * j + 1] = static_cast<uchar>(L * 255 + (g - L * 255) * (1 + alpha));
                t[3 * j] = static_cast<uchar>(L * 255 + (b - L * 255) * (1 + alpha));
            }
        }
    }
    return mMatToQImage(dist);
#endif
}


// 透明度调整 // transparency(255): [0,255]  255是完全透明
QImage ImageProcessAPI::mImageTransparencyAdjust(const QImage &image, int transparency)
{
#if 1
    // Qt函数实现
    // qDebug()<< "transparency: " << transparency;
    QImage newImage(image.width(), image.height(), QImage::Format_ARGB32);
    QColor oldColor;
    int r, g, b;
    for (int x = 0; x < newImage.width(); x++) {
        for (int y = 0; y < newImage.height(); y++) {
            oldColor = QColor(image.pixel(x, y));
            r = oldColor.red() ;
            g = oldColor.green() ;
            b = oldColor.blue() ;
            newImage.setPixel(x, y, qRgba(r, g, b, transparency));
        }
    }
    return newImage;
#else
    // OpeCV函数实现，暂时不想实现

#endif
}


// 形态学处理
/*****************************************************************************************/


// 腐蚀
QImage ImageProcessAPI::mMorphErode(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    // 灰度化+二值化
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    // cv::erode(binary, dist, kernel);							// 腐蚀（一次）
    // cv::erode(binary, dist, kernel, cv::Point(-1, -1), 3);		// 腐蚀（迭代三次）
    cv::morphologyEx(binary, dist, cv::MORPH_ERODE, kernel, cv::Point(-1, -1), 1, 0);  // 腐蚀
    return mMatToQImage(dist);
#endif
}


// 膨胀
QImage ImageProcessAPI::mMorphDilate(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    // 灰度化+二值化
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    // cv::dilate(binary, dist, kernel);							// 膨胀（一次）
    // cv::dilate(binary, dist, kernel, cv::Point(-1, -1), 3);		// 膨胀（迭代三次）
    cv::morphologyEx(binary, dist, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1, 0);  // 膨胀
    return mMatToQImage(dist);
#endif
}


// 开运算  先腐蚀再膨胀 用于去除微小干扰点/块
QImage ImageProcessAPI::mMorphOpen(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    // 灰度化+二值化
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    cv::Mat erodeResult;
    // cv::erode(binary, erodeResult, kernel);  // 先腐蚀
    // cv::dilate(erodeResult, dist, kernel);  // 再膨胀
    cv::morphologyEx(binary, dist, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1, 0);  // 开运算
    return mMatToQImage(dist);
#endif
}


// 闭运算  先膨胀再腐蚀 用于填充闭合区域
QImage ImageProcessAPI::mMorphClose(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    // 灰度化+二值化
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    cv::Mat dilateResult;
    // cv::dilate(binary, dilateResult, kernel);  // 先膨胀
    // cv::erode(dilateResult, dist, kernel);  // 再腐蚀
    cv::morphologyEx(binary, dist, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1, 0);  // 闭运算
    return mMatToQImage(dist);
#endif
}


// 顶帽运算  原图（减去）开运算图
QImage ImageProcessAPI::mMorphTopHat(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    // 灰度化+二值化
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    // cv::Mat erodeResult, dilateResult;
    // cv::erode(binary, erodeResult, kernel);  // 先腐蚀
    // cv::dilate(erodeResult, dilateResult, kernel);  // 再膨胀
    // if(src.rows != dilateResult.rows || src.cols != dilateResult.cols)
    // {
    //     cv::resize(dilateResult, dilateResult, src.size());
    //     qDebug() << "src:" << src.rows << ":" << src.cols;
    //     qDebug() << "dilateResult:" << dilateResult.rows <<":" << dilateResult.cols;
    // }
    // dist = binary - dilateResult;  // 二值化原图减去开运算结果
    cv::morphologyEx(binary, dist, cv::MORPH_TOPHAT, kernel, cv::Point(-1, -1), 1, 0);  // 顶帽运算
    return mMatToQImage(dist);
#endif
}


// 黑帽运算  闭运算图（减去）原图
QImage ImageProcessAPI::mMorphBlockHat(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    // 灰度化+二值化
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    // cv::Mat dilateResult, erodeResult;
    // cv::dilate(binary, dilateResult, kernel);  // 先膨胀
    // cv::erode(dilateResult, erodeResult, kernel);  // 再腐蚀
    // if(src.rows != erodeResult.rows || src.cols != erodeResult.cols)
    // {
    //     cv::resize(erodeResult, erodeResult, src.size());
    //     qDebug() << "src:" << src.rows << ":" << src.cols;
    //     qDebug() << "erodeResult:" << erodeResult.rows <<":" << erodeResult.cols;
    // }
    // dist = erodeResult - binary;  // 闭运算结果减去二值化原图
    cv::morphologyEx(binary, dist, cv::MORPH_BLACKHAT, kernel, cv::Point(-1, -1), 1, 0);  // 黑帽运算
    return mMatToQImage(dist);
#endif
}


// 基本梯度运算  膨胀图（减去）腐蚀图
QImage ImageProcessAPI::mMorphGradient(const QImage &image)
{
#if 0
    // Qt函数实现，暂时不想实现

#else
    // OpeCV函数实现
    cv::Mat src = mQimageToMat(image);
    if (!src.data){
        std::cout << "Can't read cv::Mat image!" << std::endl;
        return QImage();
    }
    cv::Mat dist;
    cv::Mat gray, binary;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 生成结构元素
    // int kernel_size = 3;
    int kernel_size = 5;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
    // 调用API
    // cv::Mat erodeResult, dilateResult;
    // cv::erode(binary, erodeResult, kernel);  // 原图腐蚀
    // cv::dilate(binary, dilateResult, kernel);  // 原图膨胀
    // dist = dilateResult - erodeResult;  // 原图膨胀结果减去原图腐蚀结果
    cv::morphologyEx(binary, dist, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1, 0);  // 基本梯度运算
    return mMatToQImage(dist);
#endif
}





#if 1
    // Qt函数实现

#else
    // OpeCV函数实现，暂时不想实现

#endif
