#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;


#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#define PROCESS_IMG_SUCESS 0
#define PROCESS_IMG_FAIL 1


//傅里叶变换类
class FourierTransformate
{
public:
    FourierTransformate() { cout << "FourierTransformate is being created" << endl; }  // 构造函数
    ~FourierTransformate() { cout << "FourierTransformate is being deleted" << endl; }  // 析构函数
    int FftTransformate(cv::Mat srcImage, cv::Mat &dstImage);  //傅里叶变换类成员函数声明
};


//傅里叶变换类成员函数定义
int FourierTransformate::FftTransformate(cv::Mat srcImage, cv::Mat &dstImage)
{
	//判断图像是否加载成功
	if (srcImage.empty())
	{
		cout << "图像加载失败!" << endl;
		return 1;
	}
	//将输入图像扩展到最佳尺寸，边界用0填充
	//离散傅里叶变换的运行速度与图像的大小有很大的关系，当图像的尺寸使2，3，5的整数倍时，计算速度最快
	//为了达到快速计算的目的，经常通过添加新的边缘像素的方法获取最佳图像尺寸
	//函数getOptimalDFTSize()用于返回最佳尺寸，copyMakeBorder()用于填充边缘像素
	int m = getOptimalDFTSize(srcImage.rows);
	int n = getOptimalDFTSize(srcImage.cols);
	Mat padded;
	copyMakeBorder(srcImage, padded, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	cout << padded.size() << padded.channels() << endl;
	//为傅立叶变换的结果分配存储空间
	//将plannes数组组合成一个多通道的数组，两个同搭配，分别保存实部和虚部
	//傅里叶变换的结果使复数，这就是说对于每个图像原像素值，会有两个图像值
	//此外，频域值范围远远超过图象值范围，因此至少将频域储存在float中
	//所以我们将输入图像转换成浮点型，并且多加一个额外通道来存储复数部分
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	cout << complexI.size() << endl;
	cout << planes->size() << endl;
	//进行离散傅立叶变换
	dft(complexI, complexI);
	//将复数转化为幅值，保存在planes[0]
	split(complexI, planes);   // 将多通道分为几个单通道
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];

	//傅里叶变换的幅值达到不适合在屏幕上显示，因此我们用对数尺度来替换线性尺度
	//进行对数尺度logarithmic scale缩放
	magnitudeImage += Scalar::all(1);     //所有的像素都加1
	log(magnitudeImage, magnitudeImage);      //求自然对数
    //剪切和重分布幅度图像限
	//如果有奇数行或奇数列，进行频谱裁剪
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

	// ---- -------- 下面的是为了显示结果 ---------------
	// 一分为四，左上与右下交换，右上与左下交换
	// 重新排列傅里叶图像中的象限，使原点位于图像中心
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));   // ROI区域的左上
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));  // ROI区域的右上
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));  // ROI区域的左下
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy)); // ROI区域的右下
	//交换象限（左上与右下进行交换）
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	//交换象限（右上与左下进行交换）
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	// 归一化
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	dstImage = magnitudeImage.clone();
	//显示效果图
	/*imshow("频谱幅值", magnitudeImage);*/

	return 0;
}


//傅里叶变换测试
int test02(string img_path){
    fs::path filePath = img_path;
    string filename = filePath.stem().string();
    // cout<< filename<<endl;
    // return 0;
    //傅里叶变换类对象
    FourierTransformate ImgFft;
	// 读取源图像及判断
	cv::Mat srcImage = cv::imread(img_path);
	if (!srcImage.data)
	{
		return 1;
	}
	cv::namedWindow("原始图", 0);
	cv::imshow("原始图", srcImage);
	// 转化为灰度图像
	cv::Mat srcGray;
	if (srcImage.channels() == 3)
	{
		cv::cvtColor(srcImage, srcGray, COLOR_RGB2GRAY);
	}
	else
	{
		srcGray = srcImage.clone();
	}
    // return 0;
    // imwrite("./img/"+filename+"_gray.png", srcGray);
	// cv::namedWindow("灰度图", 0);
	// cv::imshow("灰度图", srcGray);
	//傅里叶变换	
	Mat fftImage;
	ImgFft.FftTransformate(srcGray, fftImage);
	cv::namedWindow("傅里叶变换结果图", 0);
	cv::imshow("傅里叶变换结果图", fftImage); 

	// // 创建一个空彩色图像用于存储转换结果
    // cv::Mat fftcolorImage;
    // // 将灰度图转换为彩色图
    // cv::cvtColor(fftImage, fftcolorImage, cv::COLOR_GRAY2BGR);
	// imwrite("./img/"+filename+"_ft.png", fftcolorImage);
	// imwrite("./img/"+filename+"_ft.png", fftImage * 255);
	// cv::namedWindow("FT保存前图", 0);
	// cv::imshow("FT保存前图", fftcolorImage); 
	
	// cv::Mat ft_img = cv::imread("./img/"+filename+"_ft.png");
	// if (!ft_img.data)
	// {
	// 	return 1;
	// }
	// cv::namedWindow("FT保存后图", 0);
	// cv::imshow("FT保存后图", ft_img);

	cv::waitKey(0);
	return 0;
}


int main() {
    string test_img1 = "./img/ocean_ex_00035_original.png";
    test02(test_img1);

	// string test_imgs[] = {"361.jpg", "370.jpg", "436.jpg", "1092.jpg", "2493.jpg", "5443.jpg", "6136.jpg", "6301.jpg", "ocean_ex_00035_original.png"};
	// int n = sizeof(test_imgs) / sizeof(test_imgs[0]);
	// for(int i=0; i<n; ++i){
	// 	test02("./img/" + test_imgs[i]);
	// }
    
    // cout << __cplusplus << endl;  // 201703
    // system("pause");
    return 0;
}