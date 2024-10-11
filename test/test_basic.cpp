#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;



int test_01(){

    // // Mat img01 = cv::imread("./img/ocean_ex_00035_original.png", cv::IMREAD_COLOR);
    // cv::Mat img01 = cv::imread("./img/ocean_ex_00035_original.png", cv::IMREAD_GRAYSCALE);
    // cv::imshow("水下图像", img01);
    // waitKey(5000);


    //输入图像（字符串）路径的五种方法：
	//（1）单左斜线法		string imgpath = "C:/Users/pc/Desktop/test.jpg";
	//（2）双右斜线法		string imgpath = "C:\\Users\\pc\\Desktop\\test.jpg";
	//（3）双左斜线法		string imgpath = "C://Users//pc//Desktop//test.jpg";
	//（4）以上三种混合法		string imgpath = "C:/Users//pc\\Desktop//test.jpg";
	//（5）相对路径法		string imgpath = "test.jpg";
	
	//（1）读取图像
	std::string img_path = "./img/ocean_ex_00035_original.png";	
	// cv::Mat img = cv::imread(img_path, 1);		
    cv::Mat img = cv::imread(img_path, -1);
				
	//（2）判断图像是否读取成功
	if(img.empty())											
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
    
    //（3）打印图像信息
    std::cout << "img.type(): "<<img.type() << std::endl;
    std::cout << "img.dims: "<<img.dims << std::endl;
	std::cout << "img.cols: "<<img.cols << std::endl;
	std::cout << "img.rows: " << img.rows << std::endl;
	std::cout << "img.channels(): " << img.channels() << std::endl;
    std::cout << "img.depth(): " <<img.depth() << std::endl;  // enum{CV_8U=0,CV_8S=1,CV_16U=2,CV_16S=3,CV_32S=4,CV_32F=5,CV_64F=6}
    std::cout << "img.elemSize(): " << img.elemSize() << std::endl;
    // std::cout << "img.data: " << img.data << std::endl;
	uchar* pData = img.data;
	// std::cout << "pData: " << pData << std::endl;
    std::cout << "&pData: " << &pData << std::endl;
	// std::cout << "img: " << img << std::endl;
    // std::cout << "打印矩阵：" << *pData << std::endl;
	
	//（4）保存与显示图像
	// cv::imwrite("./img/test_save.jpg", img);		//保存图像
	cv::imshow("img", img);			//显示图像
	
	cv::waitKey(0);		//等待用户任意按键后结束暂停功能

    return 0;
}



int test_02()
{
	//（1）读取图像
	std::string img_path = "./img/ocean_ex_00035_original.png";	
	cv::Mat img = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (img.empty())
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）空间颜色转换
	cv::Mat img_RGB, img_gray;
	cv::cvtColor(img, img_RGB, cv::COLOR_BGR2RGB);
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    cv::namedWindow("BGR", WINDOW_NORMAL);
    cv::namedWindow("RGB", WINDOW_NORMAL);
    cv::namedWindow("Gray", WINDOW_NORMAL);

	//（4）显示图像
	cv::imshow("BGR", img);
	cv::imshow("RGB", img_RGB);
	cv::imshow("Gray", img_gray);

	cv::waitKey(0);		//等待用户任意按键后结束暂停功能
	return 0;
}



int test_03()
{
	//（1）读取图像
	std::string img_path1 = "./img/6301.jpg";
	std::string img_path2 = "./img/436.jpg";
	cv::Mat img1 = cv::imread(img_path1, 1);				
	cv::Mat img2 = cv::imread(img_path2, 1);		
		
	//（2）判断图像是否读取成功
	if(img1.empty() || img2.empty())											
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
    
    //（3）判断两张图像的宽高是否相同
    if(img1.rows != img2.rows || img1.cols != img2.cols)
    {
    	cv::resize(img2, img2, img1.size());
	    std::cout << "img1:" << img1.rows <<":" << img1.cols << std::endl;
    	std::cout << "img2:" << img2.rows <<":" << img2.cols << std::endl;
    }
	
	//（4）计算两个数组的加权和
	cv::Mat img3;
	double alpha = 0.6;
	// cv::addWeighted(img1, alpha, img2, (1-alpha), 0, img3);
    cv::addWeighted(img1, 0.75, img2, 0.75, 0, img3);
	
	//显示图像
	cv::imshow("img1", img1);					
	cv::imshow("img2", img2);			
	cv::imshow("img3", img3);			
	
	cv::waitKey(0);		//等待用户任意按键后结束暂停功能
	return 0;
}



int test_04()
{
	//（1）读取图像
	std::string img_path = "./img/6301.jpg";	
	cv::Mat src = cv::imread(img_path, 1);			
		
	//（2）判断图像是否读取成功
	if(src.empty())											
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
 
	//（3）将图像分割成多个通道（分离后的每个通道都是灰度图；而新建其余两通道全0，则可以显示对应的通道颜色）
	std::vector<cv::Mat> rgbChannels(3);
	cv::split(src, rgbChannels);
 
	//（4）新建其余两通道全0，并显示对应的通道颜色。
	cv::Mat blank_ch;
	blank_ch = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);		//新建全0矩阵：大小与Mat相同，类型为CV_8UC1。
	//（4.1）显示红色通道
	std::vector<cv::Mat> channels_r;
	channels_r.push_back(blank_ch);		//Mat.push_back：将一个或多个元素添加到矩阵底部。其类型和列数必须与Mat矩阵中的相同。
	channels_r.push_back(blank_ch);
	channels_r.push_back(rgbChannels[2]);
	//（4.2）显示绿色通道
	std::vector<cv::Mat> channels_g;
	channels_g.push_back(blank_ch);
	channels_g.push_back(rgbChannels[1]);
	channels_g.push_back(blank_ch);
	//（4.3）显示蓝色通道
	std::vector<cv::Mat> channels_b;
	channels_b.push_back(rgbChannels[0]);
	channels_b.push_back(blank_ch);
	channels_b.push_back(blank_ch);
	
	//（5）合并三个通道
	cv::Mat r_img, g_img, b_img;		// 分离后的每个通道都是灰度图；而新建其余两通道全0，则可以显示对应的通道颜色。
	cv::merge(channels_r, r_img);		//（显示红色通道，其余两通道全部置0）
	cv::merge(channels_g, g_img);		//（显示绿色通道，其余两通道全部置0）
	cv::merge(channels_b, b_img);		//（显示蓝色通道，其余两通道全部置0）
	
	//（6）显示图像
	cv::imshow("src", src);
	cv::imshow("R_img", r_img);
	cv::imshow("G_img", g_img);
	cv::imshow("B_img", b_img);
	
	cv::imshow("R_gray", rgbChannels.at(0));
	cv::imshow("G_gray", rgbChannels.at(1));
	cv::imshow("B_gray", rgbChannels.at(2));
	
	cv::waitKey(0);
	return 0;
}


int test_05()
{
    //（1）读取图像
    std::string img_path = "./img/6301.jpg";
    cv::Mat src = cv::imread(img_path, 1);
	
	//（2）判断图像是否读取成功
    if (!src.data)
    {
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
        
    //（3）转换为灰度图
    cv::Mat srcGray;
    cv::cvtColor(src, srcGray, cv::COLOR_RGB2GRAY);
    
    //（4）阈值化处理
    cv::Mat THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV, THRESH_MASK, THRESH_OTSU, THRESH_TRIANGLE;
    double thresh = 125;
    double maxval = 255;
	cv::threshold(srcGray, THRESH_BINARY, 		thresh, maxval, 0);
	cv::threshold(srcGray, THRESH_BINARY_INV, 	thresh, maxval, 1);
	cv::threshold(srcGray, THRESH_TRUNC, 		thresh, maxval, 2);
	cv::threshold(srcGray, THRESH_TOZERO, 		thresh, maxval, 3);
	cv::threshold(srcGray, THRESH_TOZERO_INV, 	thresh, maxval, 4);
	//cv::threshold(srcGray, THRESH_MASK, 		thresh, maxval, 5);
	//cv::threshold(srcGray, THRESH_OTSU, 		thresh, maxval, 6);
	//cv::threshold(srcGray, THRESH_TRIANGLE,		thresh, maxval, 7);

	//（5）自适应阈值化处理
	cv::Mat ADAPTIVE_THRESH_MEAN_C0, ADAPTIVE_THRESH_MEAN_C1, ADAPTIVE_THRESH_GAUSSIAN_C0, ADAPTIVE_THRESH_GAUSSIAN_C1;
    int blockSize = 5;
    int constValue = 10;
    const int maxVal = 255;
    cv::adaptiveThreshold(srcGray, ADAPTIVE_THRESH_MEAN_C0, 	maxVal, cv::ADAPTIVE_THRESH_MEAN_C, 	cv::THRESH_BINARY, blockSize, constValue);
	cv::adaptiveThreshold(srcGray, ADAPTIVE_THRESH_MEAN_C1, 	maxVal, cv::ADAPTIVE_THRESH_MEAN_C, 	cv::THRESH_BINARY_INV, blockSize, constValue);
	cv::adaptiveThreshold(srcGray, ADAPTIVE_THRESH_GAUSSIAN_C0, maxVal, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, constValue);
	cv::adaptiveThreshold(srcGray, ADAPTIVE_THRESH_GAUSSIAN_C1, maxVal, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, blockSize, constValue);
	
	//（6）显示图像
	cv::imshow("srcGray", srcGray);
	
	cv::imshow("img1 THRESH_BINARY", THRESH_BINARY);
	cv::imshow("img2 THRESH_BINARY_INV", THRESH_BINARY_INV);
	cv::imshow("img3 THRESH_TRUNC", THRESH_TRUNC);
	cv::imshow("img4 THRESH_TOZERO", THRESH_TOZERO);
	cv::imshow("img5 THRESH_TOZERO_INV", THRESH_TOZERO_INV);
	//cv::imshow("img6", THRESH_MASK);
	//cv::imshow("img7", THRESH_OTSU);
	//cv::imshow("img8", THRESH_TRIANGLE);
	
	cv::imshow("img11 ADAPTIVE_THRESH_MEAN_C0", ADAPTIVE_THRESH_MEAN_C0);
	cv::imshow("img22 ADAPTIVE_THRESH_MEAN_C1", ADAPTIVE_THRESH_MEAN_C1);
	cv::imshow("img33 ADAPTIVE_THRESH_GAUSSIAN_C0", ADAPTIVE_THRESH_GAUSSIAN_C0);
	cv::imshow("img44 ADAPTIVE_THRESH_GAUSSIAN_C1", ADAPTIVE_THRESH_GAUSSIAN_C1);
	
    cv::waitKey(0);
    return 0;
}



int test_06()
{
    //（1）读取图像
    // std::string img_path = "./img/anqi.png";
    std::string img_path = "./img/436.jpg";
    cv::Mat src = cv::imread(img_path, 1);

    //（2）判断图像是否读取成功
    if (!src.data)
    {
        std::cout << "can't read image!" << std::endl;
        return -1;
    }

    //（3）滤波处理（由于滤波器的边界类型默认为自动填充=1，故滤波前后的尺寸不变。）
    cv::Mat img1, img2, img3, img4, img5, img6, img7;

    cv::blur(src, img1, cv::Size(3, 3));						//均值滤波
    cv::boxFilter(src, img2, src.depth(), cv::Size(3, 3));		//方框滤波
    cv::GaussianBlur(src, img3, cv::Size(3, 3), 0, 0);			//高斯滤波
    cv::medianBlur(src, img4, 3);								//中值滤波
    cv::bilateralFilter(src, img5, 5, 150, 150);				//双边滤波

    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, 3, 1, 1, 1, 1) / int(1 + 1 + 1 + 1 + 3 + 1 + 1 + 1 + 1);
    cv::filter2D(src, img6, src.depth(), kernel);				//自定义卷积

    cv::Mat kx = (cv::Mat_<float>(1, 3) << 0, -1, 0);
    cv::Mat ky = (cv::Mat_<float>(1, 3) << -1, 0, -1);
    cv::sepFilter2D(src, img7, src.depth(), kx, ky);			//可分离滤波

    std::cout << "高：" << src.rows << "宽" << src.cols << std::endl;
    std::cout << "高：" << img1.rows << "宽" << img1.cols << std::endl;

    //（4）显示图像
    cv::imshow("src", src);

    cv::imshow("均值", img1);
    cv::imshow("方框", img2);
    cv::imshow("高斯", img3);
    cv::imshow("中值", img4);
    cv::imshow("双边", img5);
    cv::imshow("自定义", img6);
    cv::imshow("可分离", img7);

    cv::waitKey(0);
    return 0;
}



cv::Mat Translation(cv::Mat src, int tx, int ty);
cv::Mat Rotate(cv::Mat src, int rotate_angle);
cv::Mat WARPAFFINE(cv::Mat src);

/*--------------------------------------------------
函数说明：图像平移
--------------------------------------------------
输入参数：	src			输入图像
			tx			x轴平移距离
			ty			y轴平移距离
--------------------------------------------------*/
cv::Mat Translation(cv::Mat src, int tx, int ty)
{
	cv::Mat dst;
	int height = src.cols;		//获取图像的高度
	int width = src.rows;		//获取图像的宽度
	// 使用tx和ty创建平移矩阵
	float warp_values[] = { 1.0, 0.0, tx, 0.0, 1.0, ty };
	cv::Mat translation_matrix = cv::Mat(2, 3, CV_32F, warp_values);
	// 基于平移矩阵进行仿射变换
	cv::warpAffine(src, dst, translation_matrix, src.size());

	return dst;
}

/*--------------------------------------------------
函数说明：图像旋转
--------------------------------------------------
输入参数：	src			输入图像
			angle		旋转角度
--------------------------------------------------*/
cv::Mat Rotate(cv::Mat src, int angle)
{
	cv::Mat dst;
	int wight = src.cols;
	int height = src.rows;
	//获取旋转矩阵
	cv::Mat Matrix = cv::getRotationMatrix2D(cv::Point2f(wight / 2, height / 2), angle, 1.0);
	
	//获取旋转后图像的尺寸
	double cos = abs(Matrix.at<double>(0, 0));
	double sin = abs(Matrix.at<double>(0, 1));
	int nw = cos * wight + sin * height;
	int nh = sin * wight + cos * height;
	//获取x, y方向的偏移量
	Matrix.at<double>(0, 2) += (nw / 2 - wight / 2);
	Matrix.at<double>(1, 2) += (nh / 2 - height / 2);
	//基于旋转矩阵进行仿射变换
	cv::warpAffine(src, dst, Matrix, cv::Size(nh, nw));
	
	return dst;
}


int test_07()
{
	//（1）读取图像
	std::string img_path1 = "./img/anqi.png";	
	std::string img_path2 = "./img/6301.jpg";
	cv::Mat src = cv::imread(img_path1, 1);				
	cv::Mat src2 = cv::imread(img_path2, 1);		
		
	//（2）判断图像是否读取成功
	if(src.empty() || src2.empty())											
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
    
    //（3）将（图像2）拉伸与（图像1）相同尺寸。
    if(src.rows != src2.rows || src.cols != src2.cols)
    {
       	std::cout << "src=" << src.rows <<":" << src.cols << std::endl;
    	std::cout << "src2=" << src2.rows <<":" << src2.cols << std::endl;
    	
    	cv::resize(src2, src2, src.size(), 0, 0, cv::INTER_LINEAR);
    	
    	std::cout << "src11=" << src.rows <<":" << src.cols << std::endl;
    	std::cout << "src22=" << src2.rows <<":" << src2.cols << std::endl;
    }
    
    //（4）图像缩放（等比例放大 + 等比例缩小）
    cv::Mat resize_B, resize_S;
	cv::resize(src, resize_B, cv::Size(0, 0), 1, 2, cv::INTER_LINEAR);
	cv::resize(src, resize_S, cv::Size(0, 0), 0.8, 0.8, cv::INTER_LINEAR);
   	std::cout << "resize_B=" << resize_B.rows <<":" << resize_B.cols << std::endl;
   	std::cout << "resize_S=" << resize_S.rows <<":" << resize_S.cols << std::endl;

	//（5）图像翻转
	cv::Mat src_flip;
	cv::flip(src, src_flip, 0);		//	=0：上下翻转	>0：左右翻转	<0：上下和左右同时翻转

   	//（6）图像旋转（封装函数）
	cv::Mat src_rotate;
	src_rotate = Rotate(src, 45);

   	//（7）图像平移（封装函数）
	cv::Mat src_trans;
	src_trans = Translation(src, 20, 50);		

	//（8）仿射变换
	cv::Mat src_wrap;
	cv::Point2f src_xy[3];											//三个点坐标(x,y)，其中x、y是浮点型。
	cv::Point2f dst_xy[3];		
	src_xy[0] = cv::Point2f(0, 0);									//计算输入图像的三点坐标
	src_xy[1] = cv::Point2f(src.cols - 1, 0);
	src_xy[2] = cv::Point2f(0, src.rows - 1);
	dst_xy[0] = cv::Point2f(src.cols*0.0, src.rows*0.33);			//计算输入图像变换后对应的三点坐标
	dst_xy[1] = cv::Point2f(src.cols*0.85, src.rows*0.25);
	dst_xy[2] = cv::Point2f(src.cols*0.15, src.rows*0.7);
	cv::Mat warp_mat = cv::getAffineTransform(src_xy, dst_xy);		//计算仿射变换矩阵
	cv::warpAffine(src, src_wrap, warp_mat, src.size());			//仿射变换	
	//标记坐标点
	cv::Mat src_WW(src);
	for (int i = 0; i < 4; i++)
	{
		circle(src_WW, src_xy[i], 2, cv::Scalar(0, 0, 255), 2);
		circle(src_wrap, dst_xy[i], 2, cv::Scalar(0, 0, 255), 2);
	}
	cv::imshow("src_WW", src_WW);	

	//（9）透视变换
	//11、若输入坐标超过图像尺寸，则显示异常。22、若输入坐标不对应，则显示异常
	cv::Mat src_Pers;
	cv::Point2f scrPoints[4] = { cv::Point2f(0, 0), cv::Point2f(src.cols-1, 0), cv::Point2f(0, src.rows-1), cv::Point2f(src.cols-1, src.rows-1) };
	cv::Point2f dstPoints[4] = { cv::Point2f(0, 0), cv::Point2f(100, 0), cv::Point2f(0, 100), cv::Point2f(150, 120) };
	cv::Mat Trans = cv::getPerspectiveTransform(scrPoints, dstPoints);				//计算透视变换矩阵
	cv::warpPerspective(src, src_Pers, Trans, cv::Size(src.cols, src.rows));		//透视变换
	//标记坐标点
	cv::Mat src_PP(src);
	for (int i = 0; i < 4; i++)
	{
		circle(src_PP, scrPoints[i], 2, cv::Scalar(0, 0, 255), 2);
		circle(src_Pers, dstPoints[i], 2, cv::Scalar(0, 0, 255), 2);
	}
	cv::imshow("src_PP", src_PP);	
	
	//显示图像
	cv::imshow("src", src);					
	cv::imshow("src2", src2);		
	cv::imshow("resize_B", resize_B);		
	cv::imshow("resize_S", resize_S);
	cv::imshow("src_flip", src_flip);
	cv::imshow("src_rotate", src_rotate);
	cv::imshow("src_trans", src_trans);
	cv::imshow("src_wrap", src_wrap);	
	cv::imshow("src_Pers", src_Pers);	
	
	cv::waitKey(0);		//等待用户任意按键后结束暂停功能
	return 0;
}



int test_08()
{
	//（1）读取图像
	std::string img_path = "./img/anqi.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）灰度化+二值化
	cv::Mat gray, binary;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);


	//（4）形态学变化
	cv::Mat img_h, img_v;
	int x_size = binary.cols / 30;			//像素的水平长度=width/30
	int y_size = binary.rows / 30;			//像素的垂直长度=height/30
	cv::Mat h_line = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(x_size, 1), cv::Point(-1, -1));
	cv::Mat v_line = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, y_size), cv::Point(-1, -1));
	cv::morphologyEx(binary, img_h, cv::MORPH_OPEN, h_line , cv::Point(-1, -1), 1, 0);			//开运算（提取水平线）
	cv::morphologyEx(binary, img_v, cv::MORPH_CLOSE, v_line, cv::Point(-1, -1), 1, 0);			//闭运算（提取垂直线）

	//（4）显示图像
	cv::imshow("binary", binary);
	cv::imshow("img_h", img_h);
	cv::imshow("img_v", img_v);

	cv::waitKey(0);
	return 0;
}



int test_09()
{
	//（1）读取图像
	std::string img_path = "./img/anqi.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）灰度化+二值化
	cv::Mat gray, binary;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);


	//（4）形态学变化
	int kernel_size = 5;
	//getStructuringElement：返回指定大小和形状的结构化元素以进行形态学运算。
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));

	cv::Mat img1, img2, img3, img4;
	cv::erode(binary, img1, kernel);							//腐蚀
	cv::erode(binary, img2, kernel, cv::Point(-1, -1), 3);		//腐蚀（迭代三次）
	cv::dilate(binary, img3, kernel);							//膨胀
	cv::dilate(binary, img4, kernel, cv::Point(-1, -1), 3);		//膨胀（迭代三次）

	cv::Mat img11, img22, img33, img44, img55, img66, img77, img88;
	cv::morphologyEx(binary, img11, cv::MORPH_ERODE, kernel, cv::Point(-1, -1), 1, 0);			//腐蚀
	cv::morphologyEx(binary, img22, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1, 0);			//膨胀
	cv::morphologyEx(binary, img33, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1, 0);			//开运算
	cv::morphologyEx(binary, img44, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1, 0);			//闭运算

	cv::morphologyEx(binary, img55, cv::MORPH_GRADIENT, kernel, cv::Point(-1, -1), 1, 0);		//基本梯度
	cv::morphologyEx(binary, img66, cv::MORPH_TOPHAT, kernel, cv::Point(-1, -1), 1, 0);			//顶帽
	cv::morphologyEx(binary, img77, cv::MORPH_BLACKHAT, kernel, cv::Point(-1, -1), 1, 0);		//黑帽
	cv::morphologyEx(binary, img88, cv::MORPH_HITMISS, kernel, cv::Point(-1, -1), 1, 0);		//击中击不中

	//（4）显示图像
	cv::imshow("src", src);

	cv::imshow("腐蚀1", img1);
	cv::imshow("腐蚀3", img2);
	cv::imshow("膨胀1", img3);
	cv::imshow("膨胀3", img4);

	cv::imshow("腐蚀", img11);
	cv::imshow("膨胀", img22);
	cv::imshow("开运算", img33);
	cv::imshow("闭运算", img44);
	cv::imshow("基本梯度", img55);
	cv::imshow("顶帽", img66);
	cv::imshow("黑帽", img77);
	cv::imshow("击中击不中", img88);

	cv::waitKey(0);
	return 0;
}



int test_10()
{
	//（1）读取图像
	std::string img_path = "./img/anqi.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）图像金字塔
	cv::Mat img_up, img_down;
	cv::pyrUp(src, img_up, cv::Size(src.cols * 2, src.rows * 2));			//上采样
	cv::pyrDown(src, img_down, cv::Size(src.cols / 2, src.rows / 2));		//降采样
	
	//（4）显示图像
	cv::imshow("src", src);
	cv::imshow("img_up", img_up);
	cv::imshow("img_down", img_down);

	cv::waitKey(0);
	return 0;
}



int test_11()
{
	//（1）读取图像
	std::string img_path = "./img/anqi.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）图像预处理
	cv::Mat src_Gray, src_Gaus;
	cv::GaussianBlur(src, src_Gaus, cv::Size(3, 3), 0, 0);		//高斯滤波
	cv::cvtColor(src, src_Gray, cv::COLOR_BGR2GRAY);			//灰度化

	//（4）边缘检测
	cv::Mat Sobel_X, Sobel_Y, Sobel_X_abs, Sobel_Y_abs, Sobel_XY, Sobel_XY1;
	cv::Sobel(src_Gray, Sobel_X, src_Gray.depth(), 1, 0);					//计算 x 轴方向
	cv::Sobel(src_Gray, Sobel_Y, src_Gray.depth(), 0, 1);					//计算 y 轴方向
	cv::convertScaleAbs(Sobel_X, Sobel_X_abs);								//取绝对值
	cv::convertScaleAbs(Sobel_Y, Sobel_Y_abs);								//取绝对值
	cv::addWeighted(Sobel_X_abs, 0.5, Sobel_Y_abs, 0.5, 0, Sobel_XY);		//图像融合
	cv::Sobel(src_Gray, Sobel_XY1, src_Gray.depth(), 1, 1);					//同时计算 x和y 轴方向
	
	cv::Mat Scharr_X, Scharr_Y, Scharr_X_abs, Scharr_Y_abs, Scharr_XY, Scharr_XY1; 
	cv::Scharr(src_Gray, Scharr_X, src_Gray.depth(), 1, 0);					//计算 x 轴方向
	cv::Scharr(src_Gray, Scharr_Y, src_Gray.depth(), 0, 1);					//计算 y 轴方向
	cv::convertScaleAbs(Scharr_X, Scharr_X_abs);							//取绝对值
	cv::convertScaleAbs(Scharr_Y, Scharr_Y_abs);							//取绝对值
	cv::addWeighted(Scharr_X_abs, 0.5, Scharr_Y_abs, 0.5, 0, Scharr_XY);	//图像融合
	//cv::Scharr(src_Gray, Scharr_XY1, src_Gray.depth(), 1, 1);				//同时计算 x和y 轴方向

	cv::Mat src_Laplacian, src_Canny;
	cv::Laplacian(src_Gray, src_Laplacian, src_Gray.depth());		
	cv::Canny(src_Gray, src_Canny, 10, 100);

	//（5）显示图像
	cv::imshow("src", src);
	//cv::imshow("Sobel_X", Sobel_X);
	//cv::imshow("Sobel_Y", Sobel_Y);	
	//cv::imshow("Sobel_X_abs", Sobel_X_abs);
	//cv::imshow("Sobel_Y_abs", Sobel_Y_abs);	
	cv::imshow("Sobel_XY", Sobel_XY);
	//cv::imshow("Sobel_XY1", Sobel_XY1);
	
	//cv::imshow("Scharr_X", Scharr_X);
	//cv::imshow("Scharr_Y", Scharr_Y);
	//cv::imshow("Scharr_X_abs", Scharr_X_abs);
	//cv::imshow("Scharr_Y_abs", Scharr_Y_abs);	
	cv::imshow("Scharr_XY", Scharr_XY);
	//cv::imshow("Scharr_XY1", Scharr_XY1);
	
	cv::imshow("src_Laplacian", src_Laplacian);
	cv::imshow("src_Canny", src_Canny);
	
	cv::waitKey(0);
	return 0;
}



int test_12()
{
	//（1）读取图像
	std::string img_path = "./img/anqi.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）图像预处理
	cv::Mat src_Gray, src_binary;
	cv::cvtColor(src, src_Gray, cv::COLOR_BGR2GRAY);							//灰度化
	cv::threshold(src_Gray, src_binary, 125, 255, cv::THRESH_BINARY);			//二值化

	//（4.1）轮廓检测
	cv::Mat src_binary1 = src_binary.clone();			//复制矩阵
	std::vector<std::vector<cv::Point>> contours;			
	std::vector<cv::Vec4i> hierarchy;					//hierarchy = cv::noArray()
	cv::findContours(src_binary1, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	
	//（4.2）在全黑画板上绘制轮廓
	cv::Mat src_binary2 = src_binary1.clone();			//复制矩阵
	src_binary2 = cv::Scalar::all(0);					//返回所有元素都为标量0的矩阵。
	cv::drawContours(src_binary2, contours, -1, cv::Scalar(255, 255, 2555));

	//（5）用近似曲线拟合轮廓的边界（全黑画板）
	cv::Mat src_binary3 = src_binary1.clone();			//复制矩阵
	src_binary3 = cv::Scalar::all(0);					//返回所有元素都为标量0的矩阵。
	std::vector<std::vector<cv::Point>> contours_poly(contours.size());					//用于存放曲线点集
	for (int i = 0; i < contours.size(); i++)
	{
		//int epsilon = cv::arcLength(cv::Mat(contours[i]), true);						//5.1计算周长
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 15, true);				//5.2近似曲线坐标
		cv::drawContours(src_binary3, contours_poly, i, cv::Scalar(255, 255, 255));  	//5.3绘制曲线
	}
	
	//（6）矩形画出轮廓的边界（全黑画板）
	cv::Mat src_binary4 = src_binary1.clone();			//复制矩阵
	src_binary4 = cv::Scalar::all(0);					//返回所有元素都为标量0的矩阵。
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Rect rect = cv::boundingRect(cv::Mat(contours[i]));					//6.1矩形坐标
		cv::rectangle(src_binary4, rect, cv::Scalar(255, 255, 255));  			//6.2绘制矩形
	}

	//（7）圆形画出轮廓的边界（全黑画板）
	cv::Mat src_binary5 = src_binary1.clone();			//复制矩阵
	src_binary5 = cv::Scalar::all(0);					//返回所有元素都为标量0的矩阵。
	cv::Point2f center;
	float radius = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		cv::minEnclosingCircle(cv::Mat(contours[i]), center, radius);			//7.1圆形坐标
		cv::circle(src_binary5, center, radius, cv::Scalar(255, 255, 255));  	//7.2绘制圆形
	}

	//（8）显示图像（由于是单通道，故颜色三通道必须都有值）
	cv::imshow("src", src);
	cv::imshow("dst", src_binary1);
	cv::imshow("cont", src_binary2);
	cv::imshow("appr", src_binary3);
	cv::imshow("rect", src_binary4);
	cv::imshow("circ", src_binary5);
	
	cv::waitKey(0);
	return 0;
}



int test_13()
{
	//绘制多种图形
	cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3);		//值全0矩阵。
	std::cout << img.cols << img.rows << std::endl;
	cv::Scalar color(0, 255, 255);		//指定颜色（RGB）
	
	//（1）绘制线
	cv::Point p1(50, 200);				//起点(y,x)
	cv::Point p2(150, 150);				//起点1(y,x)
	cv::Point p3(250, 200);				//起点2(y,x)
	cv::line(img, p1, p2, color);
	cv::line(img, p2, p3, color);
	
	//（2）绘制矩形
	cv::Point PT1(10, 10);				//左上角坐标(y,x)
	cv::Point PT2(100, 100);			//右下角坐标(y,x)
	cv::rectangle(img, PT1, PT2, color); 
	
	//（3）绘制圆形
	cv::Point P_Y(150, 300);			//中心坐标(y,x)
	int radius = 20;					//半径
	cv::circle(img, P_Y, radius, color); 

	//（4）绘制椭圆
	cv::Point P_TY(150, 300);			//中心坐标(y,x)
	cv::Size radius_TY(50, 100);		//x,y的半径
	int angle = 90;
	int angle_start = 0;
	int angle_end = 360;
	cv::ellipse(img, P_TY, radius_TY, angle, angle_start, angle_end, color); 

	//（5）填充自定义的四边形
	cv::Point pts[1][5];		//左上角、右上角、右下角、左下角、左上角。
	pts[0][0] = cv::Point(200, 10);
	pts[0][1] = cv::Point(300, 10);
	pts[0][2] = cv::Point(300, 100);
	pts[0][3] = cv::Point(200, 100);
	pts[0][4] = cv::Point(200, 10);
	const cv::Point* ppts[] = { pts[0] };
	int npt[] = { 5 };
	cv::fillPoly(img, ppts, npt, 1, color);

	//（6）添加文字
	cv::Point Putt(60, 220);
	cv::putText(img, "Hi, Pearson!", Putt, cv::FONT_HERSHEY_COMPLEX, 1.0, color);	

	//（7）循环线图
	cv::Mat img11 = img.clone();			//复制矩阵
	img11 = cv::Scalar::all(0);				//返回所有元素都为标量0的矩阵。
	
	cv::RNG rng(-1);
	cv::Point pt1;
	cv::Point pt2;
	for (int i = 0; i < 10; i++)
	{
		pt1.x = rng.uniform(0, img11.cols);
		pt2.x = rng.uniform(0, img11.cols);
		pt1.y = rng.uniform(0, img11.rows);
		pt2.y = rng.uniform(0, img11.rows);
		//cv::Scalar color0 = (rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));				//每次生成数固定
		cv::Scalar color0 = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));		//每次生成数随机
		cv::waitKey(50);
		cv::line(img11, pt1, pt2, color0);
		cv::imshow("img11", img11);
	}

	//（3）显示图像
	cv::imshow("img", img);
	cv::waitKey(0);
	return 0;
}



int test_14()
{
	//（1）读取图像
	// std::string img_path1 = "./img/anqi.png";
    std::string img_path1 = "./img/anqi_2.png";
    // std::string img_path1 = "./img/anqi_3.png";
    // std::string img_path1 = "./img/anqi_4.png";
    // std::string img_path1 = "./img/anqi_5.png";
	std::string img_path2 = "./img/anqi_face.png";
	cv::Mat src = cv::imread(img_path1, 1);
	cv::Mat temp = cv::imread(img_path2, 1);	//截取于原图的一部分

	//（2）判断图像是否读取成功
	if (!src.data || !temp.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）模板匹配
	cv::Mat src_result;
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;

	cv::Mat src_S = src.clone();
	cv::matchTemplate(src_S, temp, src_result, cv::TM_SQDIFF);
	cv::minMaxLoc(src_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(src_S, cv::Point(minLoc.x, minLoc.y), cv::Point((minLoc.x + temp.cols), (minLoc.y + temp.rows)), cv::Scalar(0, 0, 255), 2);

	cv::Mat src_SN = src.clone();
	cv::matchTemplate(src_SN, temp, src_result, cv::TM_SQDIFF_NORMED);
	cv::minMaxLoc(src_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(src_SN, cv::Point(minLoc.x, minLoc.y), cv::Point((minLoc.x + temp.cols), (minLoc.y + temp.rows)), cv::Scalar(0, 0, 255), 2);

	cv::Mat src_C = src.clone();
	cv::matchTemplate(src_C, temp, src_result, cv::TM_CCORR);
	cv::minMaxLoc(src_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(src_C, cv::Point(minLoc.x, minLoc.y), cv::Point((minLoc.x + temp.cols), (minLoc.y + temp.rows)), cv::Scalar(0, 0, 255), 2);
	
	cv::Mat src_CN = src.clone();
	cv::matchTemplate(src_CN, temp, src_result, cv::TM_CCORR_NORMED);
	cv::minMaxLoc(src_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(src_CN, cv::Point(minLoc.x, minLoc.y), cv::Point((minLoc.x + temp.cols), (minLoc.y + temp.rows)), cv::Scalar(0, 0, 255), 2);

	cv::Mat src_CF = src.clone();
	cv::matchTemplate(src_CF, temp, src_result, cv::TM_CCOEFF);
	cv::minMaxLoc(src_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(src_CF, cv::Point(minLoc.x, minLoc.y), cv::Point((minLoc.x + temp.cols), (minLoc.y + temp.rows)), cv::Scalar(0, 0, 255), 2);
	
	cv::Mat src_CFN = src.clone();
	cv::matchTemplate(src_CFN, temp, src_result, cv::TM_CCOEFF_NORMED);
	cv::minMaxLoc(src_result, &minVal, &maxVal, &minLoc, &maxLoc);
	cv::rectangle(src_CFN, cv::Point(minLoc.x, minLoc.y), cv::Point((minLoc.x + temp.cols), (minLoc.y + temp.rows)), cv::Scalar(0, 0, 255), 2);

	//（8）显示图像（由于是单通道，故颜色三通道必须都有值）
	cv::imshow("src_S", src_S);
	// cv::imshow("src_SN", src_SN);
	// cv::imshow("src_C", src_C);
	// cv::imshow("src_CN", src_CN);
	// cv::imshow("src_CF", src_CF);
	// cv::imshow("src_CFN", src_CFN);
	
	cv::waitKey(0);
	return 0;
}



int test_15()
{
	//（1）读取图像
	// std::string img_path = "./img/anqi.png";
    std::string img_path = "./img/ocean_ex_00035_original.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（4）直方图
	//4.1、通道分割
	std::vector<cv::Mat> bgr;
	cv::split(src, bgr);
	//4.2、计算直方图
	cv::Mat b_hist, g_hist, r_hist;
	int numbins = 256;						//灰度级个数（柱子数量）。一般256。
	float range[] = { 0, 256 };				//像素值范围。一般[0, 255]。
	const float* histRange = { range };
	cv::calcHist(&bgr[0], 1, 0, cv::Mat(), b_hist, 1, &numbins, &histRange);
	cv::calcHist(&bgr[1], 1, 0, cv::Mat(), g_hist, 1, &numbins, &histRange);
	cv::calcHist(&bgr[2], 1, 0, cv::Mat(), r_hist, 1, &numbins, &histRange);
	//4.3、新建空白直方图
	int hist_width = 512;
	int hist_height = 300;
	cv::Mat hist_Image(hist_height, hist_width, CV_8UC3, cv::Scalar(20, 20, 20));
	//4.4、标准化：将图像直方图的高度与输出直方图的高度保持一致
	cv::normalize(b_hist, b_hist, 0, hist_height, cv::NORM_MINMAX);
	cv::normalize(g_hist, g_hist, 0, hist_height, cv::NORM_MINMAX);
	cv::normalize(r_hist, r_hist, 0, hist_height, cv::NORM_MINMAX);
	//4.5、线图
	int binStep = cvRound((float)hist_width / (float)numbins);
	for (int i = 1; i < numbins; i++)
	{
		//11、将宽度除以数组大小，进行标准化。
		//22、统计像素值在[0, 255]中的数量。
		//33、绘制曲线。x范围[i-1, i]；y是对应xi中的像素值。
		cv::line(hist_Image, cv::Point(binStep * (i - 1), hist_height - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(binStep * (i), hist_height - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0));
		cv::line(hist_Image, cv::Point(binStep * (i - 1), hist_height - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(binStep * (i), hist_height - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0));
		cv::line(hist_Image, cv::Point(binStep * (i - 1), hist_height - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(binStep * (i), hist_height - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255));
	}

	//（5）直方图均衡化
	double cpH1 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_CORREL);
	double cpH2 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_CHISQR);
	double cpH3 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_INTERSECT);
	double cpH4 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_BHATTACHARYYA);
	double cpH5 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_HELLINGER);
	double cpH6 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_CHISQR_ALT);
	double cpH7 = cv::compareHist(b_hist, g_hist, cv::HISTCMP_KL_DIV);
	std::cout << cpH1 << std::endl;
	std::cout << cpH2 << std::endl;
	std::cout << cpH3 << std::endl;
	std::cout << cpH4 << std::endl;
	std::cout << cpH5 << std::endl;
	std::cout << cpH6 << std::endl;
	std::cout << cpH7 << std::endl;

	//（6）直方图均衡化
	cv::Mat image_EH0, image_EH1, image_EH2;
	cv::equalizeHist(bgr[0], image_EH0);
	cv::equalizeHist(bgr[1], image_EH1);
	cv::equalizeHist(bgr[2], image_EH2);

	//（7）自适应直方图均衡化
	cv::Mat img_clahe0, img_clahe1, img_clahe2;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();		//实例化CLAHE算法
	clahe->setClipLimit(4);								//在CLAHE对象上，设置对比度限制阈值
	clahe->setTilesGridSize(cv::Size(8, 8));			//在CLAHE对象上，设置均匀切分的区域
	clahe->apply(bgr[0], img_clahe0);			//在CLAHE对象上，调用.apply()方法（对B通道）来应用直方图均衡化
	clahe->apply(bgr[1], img_clahe1);			//在CLAHE对象上，调用.apply()方法（对G通道）来应用直方图均衡化
	clahe->apply(bgr[2], img_clahe2);			//在CLAHE对象上，调用.apply()方法（对R通道）来应用直方图均衡化

	//（8）显示图像（由于是单通道，故颜色三通道必须都有值）
	cv::imshow("hist", hist_Image);

	cv::imshow("EH0", image_EH0);
	cv::imshow("EH1", image_EH1);
	cv::imshow("EH2", image_EH2);

	cv::imshow("clahe0", img_clahe0);
	cv::imshow("clahe1", img_clahe1);
	cv::imshow("clahe2", img_clahe2);

	cv::waitKey(0);
	return 0;
}




void FourierTransform(cv::Mat& image, int value)
{
	//（1）数据准备
	image.convertTo(image, CV_32F);			//数据格式转换
	std::vector<cv::Mat> channels;
	cv::split(image, channels);  			//RGB通道分离
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
	
	cv::imshow("mag", mag);
	//3.3、滤波器
	for (int i = 0; i < mag.cols;i++){
		for (int j = 0; j < mag.rows; j++){
			if (abs(i - mag.cols / 2) > mag.cols / 10 || abs(j - mag.rows / 2) > mag.rows / 10)
				mag.at<float>(j, i) = value;
		}
	} 
	cv::imshow("mag2", mag);
	
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
	image_B = idft(cv::Rect(0, 0, image.cols & -2, image.rows & -2));
	image_B.copyTo(channels[0]);
	cv::merge(channels, image);
	image.convertTo(image, CV_8U);
}
 
 
int test_16()
{
	//（1）读取图像
	// std::string img_path = "./img/anqi_5.png";
    std::string img_path = "./img/ocean_ex_00035_original.png";
	cv::Mat img = cv::imread(img_path, 0);	//读取灰度图
	cv::imshow("src", img);
	
	//（2）判断图像是否读取成功
	if (!img.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
	//cvtColor(img, img, COLOR_BGR2GRAY);
	
	//（3）傅里叶变换
	int value = 1;		//0低通滤波器、1高通滤波器
    // int value = 0;		//0低通滤波器、1高通滤波器
	FourierTransform(img, value);
	cv::imshow("Filter_img", img);
	cv::waitKey();
	return 0;
}




int thresh = 130;
int max_count = 255;
cv::Mat img, img_gray;
const char* output_title = "Result";

void Harris_Demo(int, void *) 
{
	cv::Mat dst, norm_dst, normScaleDst;
	dst = cv::Mat::zeros(img_gray.size(), CV_32FC1);
	cv::cornerHarris(img_gray, dst, 2, 3, 0.04, cv::BORDER_DEFAULT);
	//最大最小值归一化[0, 255]
	cv::normalize(dst, norm_dst, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs(norm_dst, normScaleDst);

	cv::Mat resultImg = img.clone();
	for (int row = 0; row < resultImg.rows; row++) 
	{
		//定义每一行的指针
		uchar* currentRow = normScaleDst.ptr(row);
		for (int col = 0; col < resultImg.cols; col++) 
		{
			int value = (int)*currentRow;
			if (value > thresh) 
			{
				circle(resultImg, cv::Point(col, row), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			currentRow++;
		}
	}

	cv::imshow(output_title, resultImg);
}


int test_17() 
{
	//（1）读取图像
	// std::string img_path = "./img/objdetect.png";
    // std::string img_path = "./img/objdetect_2.png";
    std::string img_path = "./img/objdetect_3.png";
	img = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (img.empty())
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}
	
	//（3）角点检测
	cv::namedWindow(output_title, cv::WINDOW_AUTOSIZE);
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	cv::createTrackbar("Threshold", output_title, &thresh, max_count, Harris_Demo);
	Harris_Demo(0, 0);

	cv::waitKey(0);
	return 0;
}




cv::Mat WaterSegment(cv::Mat src);

/*--------------------------------------------------
函数说明：分水岭算法
--------------------------------------------------
输入参数：	src			输入图像
返回值：		dst			输出图像
--------------------------------------------------*/
cv::Mat WaterSegment(cv::Mat src)
{
	//（1）图像处理
	cv::Mat grayImage;
	cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);													//灰度化
	cv::imshow("GRAY", grayImage);

	cv::threshold(grayImage, grayImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);					//二值化（使用大津法）
	cv::imshow("OTSU", grayImage);

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9), cv::Point(-1, -1));		//获取结构化元素
	cv::morphologyEx(grayImage, grayImage, cv::MORPH_CLOSE, kernel);									//闭运算
	cv::imshow("CLOSE1", grayImage);

	//（2）二次处理
	cv::distanceTransform(grayImage, grayImage, cv::DIST_L2, cv::DIST_MASK_3, 5);		//距离变换
	cv::normalize(grayImage, grayImage, 0, 1, cv::NORM_MINMAX);							//由于变换后结果非常小，故需要归一化到[0-1]
	cv::imshow("normalize", grayImage);

	grayImage.convertTo(grayImage, CV_8UC1);											//数据类型转换：8位无符号整型单通道：(0-255)
	cv::threshold(grayImage, grayImage, 0, 255, cv::THRESH_BINARY);	//（二次）二值化（使用大津法）
	cv::imshow("threshold", grayImage);

	cv::morphologyEx(grayImage, grayImage, cv::MORPH_CLOSE, kernel);					//（二次）闭运算
	cv::imshow("CLOSE2", grayImage);

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
	cv::Mat dst = cv::Mat::zeros(marks.size(), CV_8UC3);				//数据类型转换：8位无符号整型三通道
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
				dst.at<cv::Vec3b>(i, j) = colors[index - 1];
			}
			else if (index == -1)		//区域之间的边界为-1，全白
			{
				dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
			else						//只检测到一个轮廓，全黑 
			{
				dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
		}
	}

	return dst;
}


int test_18()
{
	//（1）读取图像
	// std::string img_path = "./img/objdetect_2.png";
    std::string img_path = "./img/objdetect_3.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (src.empty())
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）分水岭算法
	cv::Mat dst = WaterSegment(src);
	cv::imshow("dst", dst);
	cv::waitKey(0);		//等待用户任意按键后结束暂停功能
	return 0;
}




#define WINDOW_NAME1 "【原图窗口】"        
#define WINDOW_NAME2 "【分水岭图】" 

Mat g_maskImage, g_srcImage;
Point prevPt(-1, -1);

static void ShowHelpText();												//窗口帮助文档
static void on_Mouse(int event, int x, int y, int flags, void*);		//获取鼠标绘制区域

static void on_Mouse(int event, int x, int y, int flags, void*)
{
	if (x < 0 || x >= g_srcImage.cols || y < 0 || y >= g_srcImage.rows)
		return;

	if (event == cv::EVENT_LBUTTONUP || !(flags & cv::EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == cv::EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);

	else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(g_maskImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(g_srcImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow(WINDOW_NAME1, g_srcImage);
	}
}

static void ShowHelpText()
{
	printf("\n----------------------------------------------------------------------------\n");
	printf("\t（1）使用鼠标在原图中绘制mask区域，\n\n\t（2）通过键盘控制算法。"
		"\n\n\t按键操作说明: \n\n"
		"\t\t键盘按键【1】		：运行分水岭分割算法\n"
		"\t\t键盘按键【2】		：恢复原图\n"
		"\t\t键盘按键【ESC】		：退出程序\n\n\n");
}

int test_19()
{
	//（1）窗口帮助文档
	system("color 6F");
	ShowHelpText();		
	
	//（2）读取图像
	g_srcImage = imread("./img/objdetect_3.png", 1);
	imshow(WINDOW_NAME1, g_srcImage);
	
	//（3）图像处理
	Mat srcImage, grayImage;
	g_srcImage.copyTo(srcImage);
	cvtColor(g_srcImage, g_maskImage, COLOR_BGR2GRAY);
	cvtColor(g_maskImage, grayImage, COLOR_GRAY2BGR);
	g_maskImage = Scalar::all(0);
	
	//（4）获取鼠标绘制区域
	setMouseCallback(WINDOW_NAME1, on_Mouse, 0);		//设置鼠标事件的回调函数
	
	while (1)		//可以多次运行
	{
		//给定多个按键，控制不同功能。
		int c = waitKey(0);								//等待用户任意按键后结束暂停功能
		if ((char)c == 27)								//Esc：退出程序
			break;
		else if ((char)c == '2')						//2：恢复原始图像
		{
			g_maskImage = Scalar::all(0);
			srcImage.copyTo(g_srcImage);
			imshow(WINDOW_NAME1, g_srcImage);
		}
		else if ((char)c == '1' || (char)c == ' ')		//1 or 空格：运行分水岭分割算法
		{
			//（5）检测轮廓
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(g_maskImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
			if (contours.empty())
				continue;
			
			//（6）绘制轮廓
			Mat maskImage(g_maskImage.size(), CV_32S);
			maskImage = Scalar::all(0);
			for (int index = 0; index >= 0; index = hierarchy[index][0], compCount++)
				drawContours(maskImage, contours, index, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
			if (compCount == 0)
				continue;
			
			//（7）分水岭 + 计算耗时
			double dTime = (double)getTickCount();
			watershed(srcImage, maskImage);
			dTime = (double)getTickCount() - dTime;
			printf("\t处理时间 = %gms\n", dTime * 1000. / getTickFrequency());

			//（8）随机颜色
			vector<Vec3b> colorTab;
			for (i = 0; i < compCount; i++)
			{
				int b = theRNG().uniform(0, 255);
				int g = theRNG().uniform(0, 255);
				int r = theRNG().uniform(0, 255);
				colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			
			//（9）对每一个区域进行颜色填充
			Mat watershedImage(maskImage.size(), CV_8UC3);
			for (i = 0; i < maskImage.rows; i++)
				for (j = 0; j < maskImage.cols; j++)
				{
					int index = maskImage.at<int>(i, j);
					if (index == -1)
						watershedImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
					else if (index <= 0 || index > compCount)
						watershedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					else
						watershedImage.at<Vec3b>(i, j) = colorTab[index - 1];
				}

			//（10）通道融合
			watershedImage = watershedImage * 0.5 + grayImage * 0.5;		
			imshow(WINDOW_NAME2, watershedImage);
		}
	}

	return 0;
}




cv::Vec3b RandomColor(int value)    //生成随机颜色函数
{
	value = value % 255;  //生成0~255的随机数
	cv::RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return cv::Vec3b(aa, bb, cc);
}


int test_20()
{
	//（1）读取图像
	// std::string img_path = "./img/objdetect_3.png";
    std::string img_path = "./img/anqi.png";
	cv::Mat rgb_image = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (rgb_image.empty())
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（3）边缘检测
	cv::Mat rgb_image_blur, rgb_image_canny;
	cv::GaussianBlur(rgb_image, rgb_image_blur, cv::Size(5, 5), 0, 0);		//高斯滤波（去噪）
	cv::Canny(rgb_image_blur, rgb_image_canny, 10, 120, 3, false);			//边缘算子（提取边缘特征）
	cv::imshow("blur", rgb_image_blur);
	cv::imshow("binary", rgb_image_canny);

	//（4）轮廓检测
	std::vector<std::vector<cv::Point>>contours;
	std::vector<cv::Vec4i>hierarchy;
	cv::findContours(rgb_image_canny, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	cv::Mat imageContours = cv::Mat::zeros(rgb_image.size(), CV_8UC1);
	cv::Mat marks(rgb_image.size(), CV_32S);
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
	cv::imshow("mark", marksShows);
	cv::imshow("轮廓", imageContours);

	//（5）分水岭算法
	cv::watershed(rgb_image, marks);
	cv::Mat afterWatershed;
	cv::convertScaleAbs(marks, afterWatershed);
	cv::imshow("watershed", afterWatershed);

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
	cv::Mat PerspectiveImage = cv::Mat::zeros(rgb_image.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++)
	{
		for (int j = 0; j < marks.cols; j++)
		{
			int index = marks.at<int>(i, j);
			if (marks.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<cv::Vec3b>(i, j) = RandomColor(index);
			}
		}
	}


	cv::imshow("ColorFill", PerspectiveImage);

	cv::waitKey(0);
	return 0;
}







const float param_13 = 1.0f / 3.0f;
const float param_16116 = 16.0f / 116.0f;
const float Xn = 0.950456f;
const float Yn = 1.0f;
const float Zn = 1.088754f;

float gamma(float x)
{
    return x > 0.04045 ? powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92);
}

float gamma_XYZ2RGB(float x)
{
    return x > 0.0031308 ? (1.055f * powf(x, (1 / 2.4f)) - 0.055) : (x * 12.92);
}

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

void Lab2XYZ(float L, float a, float b, float* X, float* Y, float* Z)
{
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

void RGB2XYZ(int R, int G, int B, float* X, float* Y, float* Z)
{
    float RR = gamma((float)R / 255.0f);
    float GG = gamma((float)G / 255.0f);
    float BB = gamma((float)B / 255.0f);
    *X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    *Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    *Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;
}

void XYZ2Lab(float X, float Y, float Z, float* L, float* a, float* b)
{
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

void RGB2Lab(int R, int G, int B, float* L, float* a, float* b)
{
    float X, Y, Z;
    RGB2XYZ(R, G, B, &X, &Y, &Z);
    XYZ2Lab(X, Y, Z, L, a, b);
}

void Lab2RGB(float L, float a, float b, int* R, int* G, int* B)
{
    float X, Y, Z;
    Lab2XYZ(L, a, b, &X, &Y, &Z);
    XYZ2RGB(X, Y, Z, R, G, B);
}


int test_21()
{
    // Mat src = imread("./img/objdetect_3.png");
    // Mat src = imread("./img/ocean_ex_00035_original.png");
    Mat src = imread("./img/anqi_face.png");
    if (src.empty())
    {
        cout << "read error" << endl;
        return 0;
    }
    imshow("src", src);

    vector<vector<vector<float>>> image;    //x,y,(L,a,b)
    int rows = src.rows;
    int cols = src.cols;
    int N = rows * cols;
    int K = 200;                        //K个超像素
    int M = 40;
    int S = (int)sqrt(N / K);           //以步距为S的距离划分超像素

    cout << "rows:" << rows << " cols:" << cols << endl;
    cout << "cluster num:" << K << endl;
    cout << "S:" << S << endl;
    //RGB2Lab
    for (int i = 0; i < rows; i++)
    {
        vector<vector<float>> line;
        for (int j = 0; j < cols; j++)
        {
            vector<float> pixel;
            float L;
            float a;
            float b;
            RGB2Lab(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i, j)[0], &L, &a,
                &b);
            pixel.push_back(L);
            pixel.push_back(a);
            pixel.push_back(b);
            line.push_back(pixel);
        }
        image.push_back(line);
    }
    cout << "RGB2Lab is finished" << endl;
    //聚类中心，[x y l a b]
    vector<vector<float>> Cluster;
    //生成所有聚类中心
    for (int i = S / 2; i < rows; i += S)
    {
        for (int j = S / 2; j < cols; j += S)
        {
            vector<float> c;
            c.push_back((float)i);
            c.push_back((float)j);
            c.push_back(image[i][j][0]);
            c.push_back(image[i][j][1]);
            c.push_back(image[i][j][2]);
            Cluster.push_back(c);
        }
    }
    int cluster_num = Cluster.size();
    cout << "init cluster is finished" << endl;
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
            image[c_row + 1][c_col][0] + image[c_row][c_col + 1][0] - 2 * image[c_row][c_col][0] +
            image[c_row + 1][c_col][1] + image[c_row][c_col + 1][1] - 2 * image[c_row][c_col][1] +
            image[c_row + 1][c_col][2] + image[c_row][c_col + 1][2] - 2 * image[c_row][c_col][2];
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
                    image[tmp_row + 1][tmp_col][0] + image[tmp_row][tmp_col + 1][0] -
                    image[tmp_row][tmp_col][0] + image[tmp_row + 1][tmp_col][1] +
                    image[tmp_row][tmp_col + 1][1] - 2 * image[tmp_row][tmp_col][1] +
                    image[tmp_row + 1][tmp_col][2] + image[tmp_row][tmp_col + 1][2] -
                    image[tmp_row][tmp_col][2];
                if (tmp_gradient < c_gradient)
                {
                    Cluster[c][0] = (float)tmp_row;
                    Cluster[c][1] = (float)tmp_col;
                    Cluster[c][2] = image[tmp_row][tmp_col][0];
                    Cluster[c][3] = image[tmp_row][tmp_col][1];
                    Cluster[c][3] = image[tmp_row][tmp_col][2];
                    c_gradient = tmp_gradient;
                }
            }
        }
    }
    cout << "move cluster is finished";
    //创建一个dis的矩阵for each pixel = ∞
    vector<vector<double>> distance;
    for (int i = 0; i < rows; ++i)
    {
        vector<double> tmp;
        for (int j = 0; j < cols; ++j)
        {
            tmp.push_back(INT_MAX);
        }
        distance.push_back(tmp);
    }
    //创建一个dis的矩阵for each pixel = -1
    vector<vector<int>> label;
    for (int i = 0; i < rows; ++i)
    {
        vector<int> tmp;
        for (int j = 0; j < cols; ++j)
        {
            tmp.push_back(-1);
        }
        label.push_back(tmp);
    }
    //为每一个Cluster创建一个pixel集合
    vector<vector<vector<int>>> pixel(Cluster.size());
    //核心代码部分，迭代计算
    for (int t = 0; t < 10; t++)
    {
        cout << endl << "iteration num:" << t + 1 << "  ";
        //遍历所有的中心点,在2S范围内进行像素搜索
        int c_num = 0;
        for (int c = 0; c < cluster_num; c++)
        {
            if (c - c_num >= (cluster_num / 10))
            {
                cout << "+";
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
                    float tmp_L = image[i][j][0];
                    float tmp_a = image[i][j][1];
                    float tmp_b = image[i][j][2];
                    double Dc = sqrt((tmp_L - c_L) * (tmp_L - c_L) + (tmp_a - c_a) * (tmp_a - c_a) +
                        (tmp_b - c_b) * (tmp_b - c_b));
                    double Ds = sqrt((i - c_row) * (i - c_row) + (j - c_col) * (j - c_col));
                    double D = sqrt((Dc / (double)M) * (Dc / (double)M) + (Ds / (double)S) * (Ds / (double)S));
                    if (D < distance[i][j])
                    {
                        if (label[i][j] == -1)
                        {   //还没有被标记过
                            label[i][j] = c;
                            vector<int> point;
                            point.push_back(i);
                            point.push_back(j);
                            pixel[c].push_back(point);
                        }
                        else
                        {
                            int old_cluster = label[i][j];
                            vector<vector<int>>::iterator iter;
                            for (iter = pixel[old_cluster].begin(); iter != pixel[old_cluster].end(); iter++)
                            {
                                if ((*iter)[0] == i && (*iter)[1] == j)
                                {
                                    pixel[old_cluster].erase(iter);
                                    break;
                                }
                            }
                            label[i][j] = c;
                            vector<int> point;
                            point.push_back(i);
                            point.push_back(j);
                            pixel[c].push_back(point);
                        }
                        distance[i][j] = D;
                    }
                }
            }
        }
        cout << " start update cluster";
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
            Cluster[c][2] = image[tmp_i][tmp_j][0];
            Cluster[c][3] = image[tmp_i][tmp_j][1];
            Cluster[c][4] = image[tmp_i][tmp_j][2];
        }
    }
    //导出Lab空间的矩阵
    vector<vector<vector<float>>> out_image = image;//x,y,(L,a,b)
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
    cout << endl << "export image mat finished" << endl;
    Mat dst = src.clone();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float L = out_image[i][j][0];
            float a = out_image[i][j][1];
            float b = out_image[i][j][2];
            int R, G, B;
            Lab2RGB(L, a, b, &R, &G, &B);
            Vec3b vec3b;
            vec3b[0] = B;
            vec3b[1] = G;
            vec3b[2] = R;
            dst.at<Vec3b>(i, j) = vec3b;
        }
    }
    imshow("dst", dst);
    waitKey(0);  //暂停，保持图像显示，等待按键结束
    return 0;
}



int test_22()
{
    cv::Mat src = cv::imread("./img/objdetect.png");
    if(src.empty())
    {
        printf("could not load image..\n");
        return -1;
    }
    
    cv::Mat canny, dst;
    cv::Canny(src, canny, 150, 200); 				//canny算子
    cv::cvtColor(canny, dst, cv::COLOR_GRAY2BGR); 	//灰度图转换为彩色图
    
    std::vector<cv::Vec4f> plines;
    cv::HoughLinesP(canny, plines, 1, CV_PI / 180.0, 5, 0, 10);
    cv::Scalar color = cv::Scalar(0, 0, 255);
    for(size_t i = 0; i < plines.size(); i++)
    {
        cv::Vec4f hline = plines[i];
        cv::line(dst, cv::Point(hline[0], hline[1]), cv::Point(hline[2], hline[3]), color, 3, cv::LINE_AA);
    }
    
    cv::imshow("input", src);
	cv::imshow("edge", canny);
    cv::imshow("output", dst);
    cv::waitKey(0);
    return 0;
}


int test_23()
{
    cv::Mat src = cv::imread("./img/circle.png");
    if(src.empty())
    {
        printf("could not load image..\n");
        return -1;
    }
    
    cv::Mat temp, gray, dst;
    cv::medianBlur(src, temp, 3); 					// 中值滤波
    cv::cvtColor(temp, gray, cv::COLOR_BGR2GRAY); 	// 灰度化
 
    // 霍夫圆检测
    std::vector<cv::Vec3f> pcircles;
    cv::HoughCircles(gray, pcircles, cv::HOUGH_GRADIENT, 1, 20, 100, 100, 1, 100);
 
    src.copyTo(dst);
    for(size_t i = 0; i < pcircles.size(); i++)
    {
        cv::Vec3f cc = pcircles[i]; 		// [x, y, r]
        cv::circle(dst, cv::Point(cc[0], cc[1]), cc[2], cv::Scalar(0, 0, 255), 2, cv::LINE_AA); 	// 可视化圆弧 
        cv::circle(dst, cv::Point(cc[0], cc[1]), 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA); 		// 可视化圆心
    }
    
	cv::imshow("input", src);
    cv::imshow("output", dst);
 	cv::waitKey(0);
    return 0;
}









#define max2(a,b) (a>b?a:b)
#define max3(a,b,c) (a>b?max2(a,c):max2(b,c))
#define min2(a,b) (a<b?a:b)
#define min3(a,b,c) (a<b?min2(a,c):min2(b,c))

//函数申明
cv::Mat Brightness(cv::Mat src, float brightness, int contrast);				//亮度+对比度。
cv::Mat Saturation(cv::Mat src, int saturation);								//饱和度
cv::Mat HighLight(cv::Mat src, int highlight);									//高光
cv::Mat ColorTemperature(cv::Mat src, int warm);								//暖色调
cv::Mat Shadow(cv::Mat src, int shadow);										//阴影

cv::Mat Sharpen(cv::Mat input, int percent, int type);							//图像锐化
cv::Mat Grainy(cv::Mat src, int level);											//颗粒感
cv::Mat Cartoon(cv::Mat src, double clevel, int d, double sigma, int size);		//漫画效果
cv::Mat WhiteBalcane_PRA(cv::Mat src);											//白平衡-完美反射算法（效果偏白）
cv::Mat WhiteBalcane_Gray(cv::Mat src);											//白平衡-灰度世界算法（效果偏蓝）
cv::Mat Relief(cv::Mat src);													//浮雕
cv::Mat Eclosion(cv::Mat src, cv::Point center, float level);					//羽化



//--------------------------------------------------------------------------------
//调整对比度与亮度
cv::Mat Brightness(cv::Mat src, float brightness, int contrast)
{
	cv::Mat dst;
    dst = cv::Mat::zeros(src.size(), src.type());		//新建空白模板：大小/类型与原图像一致，像素值全0。
    int height = src.rows;								//获取图像高度
    int width = src.cols;								//获取图像宽度
    float alpha = brightness;							//亮度（0~1为暗，1~正无穷为亮）
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
                dst.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha + beta);		//修改通道的像素值（blue）
                dst.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha + beta);		//修改通道的像素值（green）
                dst.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha + beta);		//修改通道的像素值（red）
            }
            else if (src.channels() == 1)
            {
                float v = src.at<uchar>(row, col);											//获取通道的像素值（单）
                dst.at<uchar>(row, col) = cv::saturate_cast<uchar>(v * alpha + beta);		//修改通道的像素值（单）
				//saturate_cast<uchar>：主要是为了防止颜色溢出操作。如果color<0，则color等于0；如果color>255，则color等于255。
            }
        }
    }
    return dst;
}

//--------------------------------------------------------------------------------
// 饱和度
cv::Mat Saturation(cv::Mat src, int saturation)
{
	float Increment = saturation * 1.0f / 100;
	cv::Mat temp = src.clone();
	int row = src.rows;
	int col = src.cols;
	for (int i = 0; i < row; ++i)
	{
		uchar *t = temp.ptr<uchar>(i);
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
	return temp;
}

//--------------------------------------------------------------------------------
// 高光
cv::Mat HighLight(cv::Mat src, int highlight)
{
	// 生成灰度图
	cv::Mat gray = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat f = src.clone();
	f.convertTo(f, CV_32FC3);
	std::vector<cv::Mat> pics;
	split(f, pics);
	gray = 0.299f*pics[2] + 0.587*pics[2] + 0.114*pics[0];
	gray = gray / 255.f;
 
	// 确定高光区
	cv::Mat thresh = cv::Mat::zeros(gray.size(), gray.type());
	thresh = gray.mul(gray);
	// 取平均值作为阈值
	cv::Scalar t = mean(thresh);
	cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
	mask.setTo(255, thresh >= t[0]);
 
	// 参数设置
	int max = 4;
	float bright = highlight / 100.0f / max;
	float mid = 1.0f + max * bright;
 
	// 边缘平滑过渡
	cv::Mat midrate = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat brightrate = cv::Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < src.rows; ++i)
	{
		uchar *m = mask.ptr<uchar>(i);
		float *th = thresh.ptr<float>(i);
		float *mi = midrate.ptr<float>(i);
		float *br = brightrate.ptr<float>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			if (m[j] == 255)
			{
				mi[j] = mid;
				br[j] = bright;
			}
			else {
				mi[j] = (mid - 1.0f) / t[0] * th[j] + 1.0f;
				br[j] = (1.0f / t[0] * th[j])*bright;
			}
		}
	}
 
	// 高光提亮，获取结果图
	cv::Mat result = cv::Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
	{
		float *mi = midrate.ptr<float>(i);
		float *br = brightrate.ptr<float>(i);
		uchar *in = src.ptr<uchar>(i);
		uchar *r = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				float temp = pow(float(in[3 * j + k]) / 255.f, 1.0f / mi[j])*(1.0 / (1 - br[j]));
				if (temp > 1.0f)
					temp = 1.0f;
				if (temp < 0.0f)
					temp = 0.0f;
				uchar utemp = uchar(255*temp);
				r[3 * j + k] = utemp;
			}
 
		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// 暖色调
cv::Mat ColorTemperature(cv::Mat src, int warm)
{
	cv::Mat result = src.clone();
	int row = src.rows;
	int col = src.cols;
	int level = warm/2;
	for (int i = 0; i < row; ++i)
	{
		uchar* a = src.ptr<uchar>(i);
		uchar* r = result.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			int R,G,B;
			// R通道
			R = a[j * 3 + 2];
			R = R + level;
			if (R > 255) 
			{
				r[j * 3 + 2] = 255;
			}
			else if (R < 0) 
			{
				r[j * 3 + 2] = 0;
			}
			else 
			{
				r[j * 3 + 2] = R;
			}
			// G通道
			G = a[j * 3 + 1];
			G = G + level;
			if (G > 255) 
			{
				r[j * 3 + 1] = 255;
			}
			else if (G < 0) 
			{
				r[j * 3 + 1] = 0;
			}
			else 
			{
				r[j * 3 + 1] = G;
			}
			// B通道
			B = a[j * 3];
			B = B - level;
			if (B > 255) 
			{
				r[j * 3] = 255;
			}
			else if (B < 0) 
			{
				r[j * 3] = 0;
			}
			else {
				r[j * 3] = B;
			}
		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// 阴影
cv::Mat Shadow(cv::Mat src, int shadow)
{
	// 生成灰度图
	cv::Mat gray = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat f = src.clone();
	f.convertTo(f, CV_32FC3);
	std::vector<cv::Mat> pics;
	split(f, pics);
	gray = 0.299f*pics[2] + 0.587*pics[2] + 0.114*pics[0];
	gray = gray / 255.f;
 
	// 确定阴影区
	cv::Mat thresh = cv::Mat::zeros(gray.size(), gray.type());	
	thresh = (1.0f - gray).mul(1.0f - gray);
	// 取平均值作为阈值
	cv::Scalar t = mean(thresh);
	cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
	mask.setTo(255, thresh >= t[0]);
 
	// 参数设置
	int max = 4;
	float bright = shadow / 100.0f / max;
	float mid = 1.0f + max * bright;
 
	// 边缘平滑过渡
	cv::Mat midrate = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat brightrate = cv::Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < src.rows; ++i)
	{
		uchar *m = mask.ptr<uchar>(i);
		float *th = thresh.ptr<float>(i);
		float *mi = midrate.ptr<float>(i);
		float *br = brightrate.ptr<float>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			if (m[j] == 255)
			{
				mi[j] = mid;
				br[j] = bright;
			}
			else 
			{
				mi[j] = (mid - 1.0f) / t[0] * th[j]+ 1.0f;   
				br[j] = (1.0f / t[0] * th[j])*bright;               
			}
		}
	}
 
	// 阴影提亮，获取结果图
	cv::Mat result = cv::Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
	{
		float *mi = midrate.ptr<float>(i);
		float *br = brightrate.ptr<float>(i);
		uchar *in = src.ptr<uchar>(i);
		uchar *r = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				float temp = pow(float(in[3 * j + k]) / 255.f, 1.0f / mi[j])*(1.0 / (1 - br[j]));
				if (temp > 1.0f)
					temp = 1.0f;
				if (temp < 0.0f)
					temp = 0.0f;
				uchar utemp = uchar(255*temp);
				r[3 * j + k] = utemp;
			}
 
		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// 漫画效果
cv::Mat Cartoon(cv::Mat src, double clevel, int d, double sigma, int size)
{
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
	cv::Mat temp = srcb / size;
	temp = temp * size;
	//（7）通道合并
	cv::Mat c3;
	cv::Mat cannyChannels[] = { cf, cf, cf };
	cv::merge(cannyChannels, 3, c3);
	//（8）图像相乘
	cv::Mat tempf;
	temp.convertTo(tempf, CV_32FC3);		// 类型转换
	cv::multiply(tempf, c3, tempf);			
	tempf.convertTo(temp, CV_8UC3);			// 类型转换
	return temp;
}

//--------------------------------------------------------------------------------
// 白平衡-灰度世界
cv::Mat WhiteBalcane_Gray(cv::Mat src)
{
	//（1）3通道处理
	cv::Mat result = src.clone();
	if (src.channels() != 3)
	{
		std::cout << "The number of image channels is not 3." << std::endl;
		return result;
	}
 	//（2）通道分离
	std::vector<cv::Mat> Channel;
	cv::split(src, Channel);
 	//（3）计算通道灰度值均值
	double Bm = cv::mean(Channel[0])[0];
	double Gm = cv::mean(Channel[1])[0];
	double Rm = cv::mean(Channel[2])[0];
	double Km = (Bm + Gm + Rm) / 3;
 	//（4）通道灰度值调整
	Channel[0] *= Km / Bm;
	Channel[1] *= Km / Gm;
	Channel[2] *= Km / Rm;
	//（5）通道合并
	cv::merge(Channel, result);
 	return result;
}

//--------------------------------------------------------------------------------
// 白平衡-完美反射
cv::Mat WhiteBalcane_PRA(cv::Mat src)
{
	//（1）3通道处理
	cv::Mat result = src.clone();
	if (src.channels() != 3)
	{
		std::cout << "The number of image channels is not 3." << std::endl;
		return result;
	}
 	//（2）通道分离
	std::vector<cv::Mat> Channel;
	cv::split(src, Channel);
 	//（3）计算单通道最大值
	int row = src.rows;
	int col = src.cols;
	int RGBSum[766] = { 0 };
	uchar maxR, maxG, maxB;
 	for (int i = 0; i < row; ++i)
	{
		uchar *b = Channel[0].ptr<uchar>(i);
		uchar *g = Channel[1].ptr<uchar>(i);
		uchar *r = Channel[2].ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			int sum = b[j] + g[j] + r[j];
			RGBSum[sum]++;
			maxB = cv::max(maxB, b[j]);
			maxG = cv::max(maxG, g[j]);
			maxR = cv::max(maxR, r[j]);
		}
	}
 	//（4）计算最亮区间下限T
	int T = 0;
	int num = 0;
	int K = static_cast<int>(row * col * 0.1);
	for (int i = 765; i >= 0; --i)
	{
		num += RGBSum[i];
		if (num > K)
		{
			T = i;
			break;
		}
	}
	//（5）计算单通道亮区平均值
	double Bm = 0.0, Gm = 0.0, Rm = 0.0;
	int count = 0;
	for (int i = 0; i < row; ++i)
	{
		uchar *b = Channel[0].ptr<uchar>(i);
		uchar *g = Channel[1].ptr<uchar>(i);
		uchar *r = Channel[2].ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			int sum = b[j] + g[j] + r[j];
			if (sum > T)
			{
				Bm += b[j];
				Gm += g[j];
				Rm += r[j];
				count++;
			}
		}
	}
	Bm /= count;
	Gm /= count;
	Rm /= count;
 	//（6）通道调整
	Channel[0] *= maxB / Bm;
	Channel[1] *= maxG / Gm;
	Channel[2] *= maxR / Rm;
 	//（7）通道合并
	cv::merge(Channel, result);
 	return result;
}

//--------------------------------------------------------------------------------
// 浮雕
cv::Mat Relief(cv::Mat src)
{
	CV_Assert(src.channels() == 3);
	int row = src.rows;
	int col = src.cols;
	cv::Mat temp = src.clone();
	for (int i = 1; i < row-1; ++i)
	{
		uchar *s1 = src.ptr<uchar>(i - 1);
		uchar *s2 = src.ptr<uchar>(i + 1);
		uchar *t = temp.ptr<uchar>(i);
		for (int j = 1; j < col-1; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				int RGB = s1[3 * (j - 1) + k] - s2[3 * (j + 1) + k] + 128;
				if (RGB < 0)RGB = 0;
				if (RGB > 255)RGB = 255;
				t[3*j+k] =(uchar)RGB;
			}
		}
	}
	return temp;
}

//--------------------------------------------------------------------------------
// 羽化
cv::Mat Eclosion(cv::Mat src, cv::Point center, float level)
{
	if (level > 0.9)
		level = 0.9f;
	float diff = (1-level) * (src.rows / 2 * src.rows / 2 + src.cols / 2 * src.cols / 2);
	cv::Mat result = src.clone();
	for (int i = 0; i < result.rows; ++i)
	{
		for (int j = 0; j < result.cols; ++j)
		{
			float dx = float(center.x - j);
			float dy = float(center.y - i);
			float ra = dx * dx + dy * dy;
			float m = ((ra-diff) / diff * 255)>0? ((ra - diff) / diff * 255):0;
			int b = result.at<cv::Vec3b>(i, j)[0];
			int g = result.at<cv::Vec3b>(i, j)[1];
			int r = result.at<cv::Vec3b>(i, j)[2];
			b = (int)(b+ m);
			g = (int)(g + m);
			r = (int)(r + m);
			result.at<cv::Vec3b>(i, j)[0] = (b > 255 ? 255 : (b < 0 ? 0 : b));
			result.at<cv::Vec3b>(i, j)[1] = (g > 255 ? 255 : (g < 0 ? 0 : g));
			result.at<cv::Vec3b>(i, j)[2] = (r > 255 ? 255 : (r < 0 ? 0 : r));
		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// 锐化
cv::Mat Sharpen(cv::Mat input, int percent, int type)
{
	cv::Mat result;
	cv::Mat s = input.clone();
	cv::Mat kernel;
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
	cv::filter2D(s, s, s.depth(), kernel);
	result = input + s * 0.01 * percent;
	return result;
}

//--------------------------------------------------------------------------------
// 颗粒感
cv::Mat Grainy(cv::Mat src, int level)
{
	int row = src.rows;
	int col = src.cols;
	if (level > 100)
		level = 100;
	if (level < 0)
		level = 0;
	cv::Mat result = src.clone();
	for (int i = 0; i < row; ++i)
	{
		uchar *t = result.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				int temp = t[3 * j + k];
				temp += ((rand() % (2 * level)) - level);
				if (temp < 0)temp = 0;
				if (temp > 255)temp = 255;
				t[3 * j + k] = temp;
			}
		}
	}
	return result;
}



int test_24()
{
	//（1）读取图像
	std::string img_path = "./img/objdetect_3.png";
	cv::Mat src = cv::imread(img_path, 1);

	//（2）判断图像是否读取成功
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	float brightness_value = 3;		//[0, 10]			亮度。暗~亮：[0, 1] ~ [1, 10]
	int contrast_value = 30;		//[-100, 100]		对比度。
	int saturation_value = 30;		//[-100, 100]		饱和度。
	int highlight_value = 30;		//[-100, 100]		高光。
	int warm_value = 30;			//[-100, 100]		暖色调。
	int shadow_value = 30;			//[-100, 100]		阴影。
	int sharpen_value = 30;			//[-100, 100]		锐化。[-1000000, 1000000]
	int grainy_value = 30;			//[0, 100]			颗粒感。

	int eclosion_flag = 1;			//[0, 1]			羽化。
	int cartoon_flag = 1;			//[0, 1]			漫画效果。clevel阈值40-80，d阈值0-10，sigma阈值10-250，size阈值10-25
	int reflect_flag = 1;			//[0, 1]			白平衡-完美反射。
	int world_flag = 1;				//[0, 1]			白平衡-灰度世界。
	int relief_flag = 1;			//[0, 1]			浮雕。

	cv::imshow("src", src);
	cv::Mat dst = src.clone();
	if (brightness_value != 1){
		dst = Brightness(dst, brightness_value, 0);
		cv::imshow("Brightness", dst);
	}
	if (contrast_value != 0){
		dst = Brightness(dst, 1, contrast_value);
		cv::imshow("Contrast", dst);
	}
	if (saturation_value != 0){
		dst = Saturation(dst, saturation_value);
		cv::imshow("Saturation", dst);
	}
	if (highlight_value != 0){
		dst = HighLight(dst, highlight_value);
		cv::imshow("HighLight", dst);
	}
	if (warm_value != 0){
		dst = ColorTemperature(dst, warm_value);
		cv::imshow("ColorTemperature", dst);
	}
	if (shadow_value != 0){
		dst = Shadow(dst, shadow_value);
		cv::imshow("Shadow", dst);
	}
	if (sharpen_value != 0){
		dst = Sharpen(dst, sharpen_value, 0);
		cv::imshow("Sharpen", dst);
	}
	if (grainy_value != 0){
		dst = Grainy(dst, grainy_value);
		cv::imshow("Grainy", dst);
	}


	cv::Mat dst2 = src.clone();
	if (cartoon_flag != 0){
		dst2 = Cartoon(dst2, 80, 5, 150, 20);		//clevel阈值40-80，d阈值0-10，sigma阈值10-250，size阈值10-25说
		cv::imshow("Cartoon", dst2);
	}
	if (reflect_flag != 0){
		dst2 = WhiteBalcane_PRA(dst2);
		cv::imshow("WhiteBalcane_PRA", dst2);
	}
	if (world_flag != 0){
		dst2 = WhiteBalcane_Gray(dst2);
		cv::imshow("WhiteBalcane_Gray", dst2);
	}
	if (relief_flag != 0){
		dst2 = Relief(dst2);
		cv::imshow("Relief", dst2);
	}
	if (eclosion_flag != 0){
		dst2 = Eclosion(dst2, cv::Point(src.cols / 2, src.rows / 2), 0.8f);
		cv::imshow("Eclosion", dst2);
	}

	//（4）显示图像
	// cv::imshow("src", src);
	// cv::imshow("锐化", dst);
	cv::waitKey(0);		//等待用户任意按键后结束暂停功能
	return 0;
}



int test_25() 
{
	//（1）构建一张单通道8位400 x 400的图像，
	const int r = 100;
	cv::Mat src = cv::Mat::zeros(r * 4, r * 4, CV_8UC1);
	
	//（2）绘制自定义六边形（通过line绘制六次）
	std::vector<cv::Point2f> vert(6);
	vert[0] = cv::Point(3 * r / 2, static_cast<int>(1.34 * r));
	vert[1] = cv::Point(1 * r, 2 * r);
	vert[2] = cv::Point(3 * r / 2, static_cast<int>(2.866 * r));
	vert[3] = cv::Point(5 * r / 2, static_cast<int>(2.866 * r));
	vert[4] = cv::Point(3 * r, 2 * r);
	vert[5] = cv::Point(5 * r / 2, static_cast<int>(1.34 * r));
	for (int ii = 0; ii < 6; ii++)
	{
		cv::line(src, vert[ii], vert[(ii+1) % 6], cv::Scalar(255), 3, 8, 0);
	}
	
	//（3）轮廓检测
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierachy;
	cv::Mat src_contours = src.clone();
	cv::findContours(src_contours, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	//（4）对图像中所有像素点进行【点多边形测试】：测试像素点是在多边形内部、边界或外部上
	cv::Mat raw_dist = cv::Mat::zeros(src_contours.size(), CV_32FC1);
	for (int row = 0; row < raw_dist.rows; row++)
	{
		for (int col = 0; col < raw_dist.cols; col++)
		{
			//输入参数：输入轮廓，测试点，是否返回距离值。（False：1表示在内部，0表示在边界上，-1表示在外部）（True表示返回实际距离值）
			double dist = cv::pointPolygonTest(contours[0], cv::Point2f(static_cast<float>(col), static_cast<float>(row)), true);
			raw_dist.at<float>(row, col) = static_cast<float>(dist);
		}
	}
	
	//（5）按内部、边界、外部三个区域分开，并且内/外部依据距离远近动态赋值，形成渐变色（也可以自定义为固定值）
	double minValue, maxValue;
	cv::minMaxLoc(raw_dist, &minValue, &maxValue, 0, 0, cv::Mat());		//计算最大值和最小值
	cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < dst.rows; row++)
	{
		for (int col = 0; col < dst.cols; col++)
		{
			float dist = raw_dist.at<float>(row, col);
			if (dist > 0)
				dst.at<cv::Vec3b>(row, col)[0] = (uchar)(abs(1.0 - (dist / maxValue)) * 255);		//内部
			else if (dist < 0)
				dst.at<cv::Vec3b>(row, col)[2] = (uchar)(abs(1.0 - (dist / minValue)) * 255);		//外部
			else
			{
				dst.at<cv::Vec3b>(row, col)[0] = (uchar)(abs(255 - dist));		//边界
				dst.at<cv::Vec3b>(row, col)[1] = (uchar)(abs(255 - dist));		//边界
				dst.at<cv::Vec3b>(row, col)[2] = (uchar)(abs(255 - dist));		//边界
			}
		}
	}
	
	//（6）显示图像
	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	return 0;
}



int test_26()
{
	//（1）输入图像
	cv::Mat src = cv::imread("./img/anqi_4.png");
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（2）图像处理
	cv::Mat src_gray_, src_blur, src_canny;
	cv::cvtColor(src, src_gray_, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(src_gray_, src_blur, cv::Size(3, 3), 0, 0);
	cv::Canny(src_blur, src_canny, 0, 160);			//该参数极大影响最终效果
	//cv::Canny(blur_src, canny_src, 80, 160);		//该参数极大影响最终效果


	//（3）轮廓检测
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierachy;
	cv::findContours(src_canny, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//（4）计算每个轮廓的矩。矩中心点center=(x0, y0)。【其中：x0=m10/m00，y0=m01/m00】
	std::vector<cv::Moments> contours_moments(contours.size());
	std::vector<cv::Point2f> ccs(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		//输入参数：输入图像，是否返回二值化图像
		contours_moments[i] = cv::moments(contours[i]);		//计算矩
		ccs[i] = cv::Point(static_cast<float>(contours_moments[i].m10 / contours_moments[i].m00), static_cast<float>(contours_moments[i].m01 / contours_moments[i].m00));
	}

	//（5）绘制轮廓和圆（打印轮廓面积 + 弧长）
	cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);		//空矩阵
	//cv::Mat dst = src.clone();
	cv::RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 10) 		//轮廓筛选（过滤较小的轮廓）
			continue;
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));		//生成随机数
		cv::drawContours(dst, contours, i, color, 2, 8, hierachy, 0, cv::Point(0, 0));
		cv::circle(dst, ccs[i], 2, color, 2, 8);
		std::cout << "当前为第[i]个轮廓：" << i << "【轮廓中心点】x=" << ccs[i].x << ", y=" << ccs[i].y << "【轮廓面积contourArea】" << cv::contourArea(contours[i]) << "【轮廓弧长arcLength】" << cv::arcLength(contours[i], true) << std::endl;
	}

	//（6）显示图像
	cv::imshow("src", src);
	cv::imshow("gray", src_gray_);
	cv::imshow("blur", src_blur);
	cv::imshow("canny", src_canny);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	return 0;
}



int test_27()
{
	//（1）输入图像
	cv::Mat src = cv::imread("./img/anqi_4.png");
	if (!src.data)
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（2）图像处理
	cv::Mat src_gray, src_blur, src_bin;
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	cv::blur(src_gray, src_blur, cv::Size(3, 3));
	cv::threshold(src_blur, src_bin, 100, 255, cv::THRESH_BINARY);		//该参数极大影响最终效果

	//（3）轮廓检测
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierachy;
	cv::findContours(src_bin, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//（4）计算凸包
	std::vector<std::vector<cv::Point>> convexs(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		//输入参数：轮廓点，凸包，方向（默认False=逆时针），是否返回点个数（默认True）
		cv::convexHull(contours[i], convexs[i], false, true);
	}

	//（5）绘制凸包
	cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);		//空矩阵
	//cv::Mat dst = src.clone();							//复制原图
	std::vector<cv::Vec4i> empty(0);
	cv::RNG rng(12345);
	for (size_t k = 0; k < contours.size(); k++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::drawContours(dst, contours, k, color, 2, cv::LINE_8, hierachy, 0, cv::Point(0, 0));
		cv::drawContours(dst, convexs, k, color, 2, cv::LINE_8, empty, 0, cv::Point(0, 0));
	}

	//（6）显示图像
	cv::imshow("src", src);
	cv::imshow("src_gray", src_gray);
	cv::imshow("src_blur", src_blur);
	cv::imshow("src_bin", src_bin);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	return 0;
}



int test_28()
{
	while (true)
	{
		//（1）输入图像
		cv::Mat src = cv::imread("./img/anqi_4.png");
		if (!src.data)
		{
			std::cout << "can't read image!" << std::endl;
			return -1;
		}

		//（2）像素重映射的四种类型（自定义）
		int c = cv::waitKey(500);		//2.1、等待键盘事件
		if ((char)c == 27) 				//2.2、退出键：Esc
			break;
		int index = c % 4;				//2.3、根据输入值进行四种类型判断：[0, 1, 2, 3]
		
		cv::Mat map_x, map_y;
		map_x.create(src.size(), CV_32FC1);			//x映射表
		map_y.create(src.size(), CV_32FC1);			//y映射表
		for (int row = 0; row < src.rows; row++) 
		{
			for (int col = 0; col < src.cols; col++) 
			{
				switch (index) 
				{
				case 0:					//2.2.1、缩小一半
					if (col > (src.cols * 0.25) && col <= (src.cols*0.75) && row > (src.rows*0.25) && row <= (src.rows*0.75)) 
					{
						map_x.at<float>(row, col) = 2 * (col - (src.cols*0.25));
						map_y.at<float>(row, col) = 2 * (row - (src.rows*0.25));
					}
					else 
					{
						map_x.at<float>(row, col) = 0;
						map_y.at<float>(row, col) = 0;
					}
					break;
				case 1:					//2.2.2、沿着Y方向翻转
					map_x.at<float>(row, col) = (src.cols - col - 1);
					map_y.at<float>(row, col) = row;
					break;
				case 2:					//2.2.3、沿着X方向翻转
					map_x.at<float>(row, col) = col;
					map_y.at<float>(row, col) = (src.rows - row - 1);
					break;
				case 3:					//2.2.4、沿着XY方向同时翻转
					map_x.at<float>(row, col) = (src.cols - col - 1);
					map_y.at<float>(row, col) = (src.rows - row - 1);
					break;
				}
			}
		}

		//（3）像素重映射：将输入图像的所有像素根据指定规则进行映射，并形成新图像。
		cv::Mat dst;
		cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 255, 255));

		//（4）显示图像
		cv::imshow("src", src);
		cv::imshow("dst", dst);
	}

	return 0;
}




int test_29() 
{
	//（1）输入图像
	cv::Mat src = cv::imread("./img/anqi_4.png");
	if (src.empty())
	{
		std::cout << "can't read image!" << std::endl;
		return -1;
	}

	//（2）图像处理
	cv::Mat src_hsv, src_hue;
	cv::cvtColor(src, src_hsv, cv::COLOR_BGR2HSV);			//格式转换
	src_hue.create(src_hsv.size(), src_hsv.depth());		//新建矩阵
	int nchannels[] = {0, 0};
	cv::mixChannels(&src_hsv, 1, &src_hue, 1, nchannels, 1);		//将制定通道从输入阵列复制到输出阵列的指定通道

	//（3）计算直方图和归一化
	cv::Mat h_hist;
	int bins = 12;
	float range[] = {0, 180};
	const float* histRanges = {range};
	cv::calcHist(&src_hue, 1, 0, cv::Mat(), h_hist, 1, &bins, &histRanges);
	cv::normalize(h_hist, h_hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

	//（4）计算直方图的反向投影
	cv::Mat Back_Project_Image;
	cv::calcBackProject(&src_hue, 1, 0, h_hist, Back_Project_Image, &histRanges, 1);
	
	//（5）计算反向投影的直方图
	int hist_h = 400;
	int hist_w = 400;
	cv::Mat Hist_Image(hist_w, hist_h, CV_8UC3, cv::Scalar(0, 0, 0));
	int bin_w = hist_w / bins;
	for(int ii = 1; ii < bins; ii++)
	{
		cv::rectangle(Hist_Image, cv::Point((ii - 1) * bin_w, (hist_h - cvRound(h_hist.at<float>(ii - 1) * (400 / 255)))), cv::Point(ii * bin_w, hist_h), cv::Scalar(0, 0, 255), -1);
	}
	
	//（6）显示图像
	cv::imshow("src", src);								//原图
	cv::imshow("BackProj", Back_Project_Image);			//反向投影
	cv::imshow("Histogram", Hist_Image);				//反向投影的直方图
	cv::waitKey(0);
	return 0;
}





int main(){

    // test_01();
    // test_02();
    // test_03();
    // test_04();
    // test_05();
    // test_06();
    // test_07();
    // test_08();
    // test_09();
    // test_10();
    // test_11();
    // test_12();
    // test_13();
    // test_14();
    // test_15();
    // test_16();
    // test_17();
    // test_18();
    // test_19();
    // test_20();
    // test_21();
    // test_22();
    // test_23();
	// test_24();
	// test_25();
	// test_26();
	// test_27();
	// test_28();
	test_29();


    return 0;
}