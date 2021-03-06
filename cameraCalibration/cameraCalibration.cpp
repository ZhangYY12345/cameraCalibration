// cameraCalibration.cpp: 定义控制台应用程序的入口点。
//

#include "methods/methods.h"
#include "methods/method_StereoMatching.h"
#include "methods/patchmatch.h"
#include "methods/parametersStereo.h"
#include "vtkAutoInit.h"
#include <pcl/io/pcd_io.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/surface/impl/organized_fast_mesh.hpp>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/features/board.h>
#include "omp.h"
#include "disp_method/methods_disp.h"
#include <future>

using namespace cv;

VTK_MODULE_INIT(vtkRenderingOpenGL);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

int main()
{
	////gif testing
	//{
	//	cv::Mat src = imread("./test/toy.bmp");
	//	cv::Mat dst;
	//	dst = addGaussianNoise(src);
	//	imwrite("./test/toy_Gaussian.png", dst);
	//	test_gifs(dst);
	//}


	//////std::string pclPath_pre = "D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/4.1/pre/filteredL_roi_res.pcd";
	//////pclFilter_my2(pclPath_pre, "filtered_");

	//////std::string pclPath_post = "D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/4.1/pre+200ml/filteredL_roi_res.pcd";
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
	//pcl::io::loadPCDFile("res.pcd", *cloud_);

	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
	//	new pcl::visualization::PCLVisualizer("Cloud Viewer"));
	////viewer->setBackgroundColor(255, 255, 255);
	//viewer->addPointCloud<pcl::PointXYZRGB>(cloud_, "cloud");
	//viewer->setPointCloudRenderingProperties(
	//	pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
	//	1, "cloud");

	//while (!viewer->wasStopped())
	//{
	//	viewer->spinOnce(100);
	//	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	//}

	//std::string xmlFilePath = "unditortStereoCalib.xml";

	//Mat stereoPair_rectified_left = imread("D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\rectify\\pcl-test\\10L_rectify.jpg");
	//Mat stereoPair_rectified_right = imread("D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\rectify\\pcl-test\\10R_rectify.jpg");
	////Mat disparityMap = imread("D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\rectify\\pcl-test\\10_ADAPTIVE_WEIGHT_GUIDED_FILTER_2disparity.jpg", IMREAD_GRAYSCALE);
	//Size imgSize = Size(640, 360);

	//Mat left_ = Mat::zeros(360, 640, CV_8UC3);
	//stereoPair_rectified_left.copyTo(left_(Rect(0, 1, 640, 358)));
	//imwrite("left_rectify_.jpg", left_);
	//Mat right_ = Mat::zeros(360, 640, CV_8UC3);
	//stereoPair_rectified_right.copyTo(right_(Rect(0, 1, 640, 358)));
	//imwrite("right_rectify_.jpg", right_);

	//Mat sobelL_x, sobelL_y, sobelR_x, sobelR_y;
	//Sobel(stereoPair_rectified_left, sobelL_x, CV_8U, 1, 0, 3, 1, 0, BORDER_REFLECT);
	//Sobel(stereoPair_rectified_left, sobelL_y, CV_8U, 0, 1, 3, 1, 0, BORDER_REFLECT);

	//Sobel(stereoPair_rectified_right, sobelR_x, CV_8U, 1, 0, 3, 1, 0, BORDER_REFLECT);
	//Sobel(stereoPair_rectified_right, sobelR_y, CV_8U, 0, 1, 3, 1, 0, BORDER_REFLECT);

	//stereoPair_rectified_left = stereoPair_rectified_left + sobelL_x + sobelL_y;
	//stereoPair_rectified_right = stereoPair_rectified_right + sobelR_x + sobelR_y;

	//Mat stereoPair_rectified_left = imread("D:/studying/stereo vision/research code/data/rectifyTest/20190409/3/rectifiedLeft.jpg");
	//Mat stereoPair_rectified_right = imread("D:/studying/stereo vision/research code/data/rectifyTest/20190409/3/rectifiedRight.jpg");

	// load image
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/data20200107/rectifyL.jpg");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/data20200107/rectifyR.jpg");
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/20200107/testImgL/2560_1440/rectifyL/23_pattern0_rectifyL.jpg");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/20200107/testImgR/2560_1440/rectifyR/23_pattern0_rectifyR.jpg");
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/20200107/testImgL/2560_2_1440_2/rectifyL/14_pattern0_rectifyL.jpg");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/20200107/testImgR/2560_2_1440_2/rectifyR/14_pattern0_rectifyR.jpg");
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/20200107/testImgL/2560_1440/rectifyL/14_pattern0_rectifyL.jpg");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/20200107/testImgR/2560_1440/rectifyR/14_pattern0_rectifyR.jpg");
	cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/20190924/0924/LL/rectify20200111/2L_rectifyL.jpg");
	cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/20190924/0924/RR/rectify20200111/2R_rectifyR.jpg");

	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Flowerpots/view1.png");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Flowerpots/view5.png");
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Rocks2/view1.png");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Rocks2/view5.png");
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Monopoly/view1.png");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Monopoly/view5.png");
	//cv::Mat imgL = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Cloth3/view1.png");
	//cv::Mat imgR = imread("D:/studying/stereo vision/research code/data/middlebury/ALL-2views-2006/Cloth3/view5.png");


	// Image loading check
	if (!check_image(imgL, "Image left")
		|| !check_image(imgR, "Image right"))
		return 1;

	// Image sizes check
	if (!check_dimensions(imgL, imgR))
		return 1;

	//equalHisImg(imgL, imgL);
	//equalHisImg(imgR, imgR);

	Size imgSize = Size(640, 360)*2;
	if (imgL.size() != imgSize)
	{
		resize(imgL, imgL, imgSize);
		resize(imgR, imgR, imgSize);
	}

	////////////////////////////////// Patch Match
	////////////////////////////////// processing images
	////////////////////////////////const float alpha = 0.9f;
	////////////////////////////////const float gamma = 10.0f;
	////////////////////////////////const float tau_c = 10.0f;
	////////////////////////////////const float tau_g = 2.0f;
	//
	////////////////////////////////pm::PatchMatch patch_match(alpha, gamma, tau_c, tau_g);
	////////////////////////////////patch_match.set(stereoPair_rectified_left, stereoPair_rectified_right);
	////////////////////////////////patch_match.process(3);
	////////////////////////////////patch_match.postProcess();
	//
	////////////////////////////////cv::Mat1f disp1 = patch_match.getLeftDisparityMap();
	////////////////////////////////cv::Mat1f disp2 = patch_match.getRightDisparityMap();
	//
	////////////////////////////////cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);
	////////////////////////////////cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);
	//
	////////////////////////////////try
	////////////////////////////////{
	////////////////////////////////	cv::imwrite("PatchMatch_left_disparity.png", disp1);
	////////////////////////////////	cv::imwrite("PatchMatch_right_disparity.png", disp2);
	////////////////////////////////}
	////////////////////////////////catch (std::exception &e)
	////////////////////////////////{
	////////////////////////////////	std::cerr << "Disparity save error.\n" << e.what();
	////////////////////////////////	return 1;
	////////////////////////////////}

	//compute disparity image
	cv::Mat dispImgL, dispImgR;
	std::future<void> ft1 = std::async(std::launch::async, [&]
	{
		StereoMatchParam param;
		param.imgLeft = imgL.clone();
		param.imgRight = imgR.clone();
		param.imgLeft_C = imgL.clone();
		param.imgRight_C = imgR.clone();
		param.isDispLeft = true;
		param.minDisparity = 50;
		param.maxDisparity = 800;
		param.winSize =35;
		dispImgL = asw_gifs(param, 0.01, OURS_GIF3, 15, 0.01, 0.5);//EGIF, WGIF, SKWGIF
	});
	std::future<void> ft2 = std::async(std::launch::async, [&]
	{
		StereoMatchParam param;
		param.imgLeft = imgL.clone();
		param.imgRight = imgR.clone();
		param.imgLeft_C = imgL.clone();
		param.imgRight_C = imgR.clone();
		param.isDispLeft = false;
		param.minDisparity = 50;
		param.maxDisparity = 800;
		param.winSize = 35;
		dispImgR = asw_gifs(param, 0.01, OURS_GIF3, 15, 0.01, 0.5);//EGIF, WGIF, SKWGIF
	});
	ft1.wait();
	ft2.wait();

	cv::Mat dispNormL, dispNormR;
	normalize(dispImgL, dispNormL, 0, 255, NORM_MINMAX);
	normalize(dispImgR, dispNormR, 0, 255, NORM_MINMAX);
	dispNormL.convertTo(dispNormL, CV_8UC1);
	dispNormR.convertTo(dispNormR, CV_8UC1);
	cv::imwrite("dispL.jpg", dispNormL);
	cv::imwrite("dispR.jpg", dispNormR);

	StereoMatchParam param;
	param.imgLeft = imgL.clone();
	param.imgRight = imgR.clone();
	param.minDisparity = 50;
	param.maxDisparity = 800;
	param.winSize = 35;

	cv::Mat dispL_filtered, dispR_filtered;
	postProcess_(param, dispImgL, dispImgR, dispL_filtered, dispR_filtered);

	cv::Mat L_filtered_norm, R_filtered_norm;
	normalize(dispL_filtered, L_filtered_norm, 0, 255, NORM_MINMAX);
	L_filtered_norm.convertTo(L_filtered_norm, CV_8UC1);
	normalize(dispR_filtered, R_filtered_norm, 0, 255, NORM_MINMAX);
	R_filtered_norm.convertTo(R_filtered_norm, CV_8UC1);
	imwrite("L_filtered.jpg", L_filtered_norm);
	imwrite("R_filtered.jpg", R_filtered_norm);

	//boxFilter(dispImgL, dispImgL, dispImgL.depth(), Size(15, 15));
	//boxFilter(dispImgR, dispImgR, dispImgR.depth(), Size(15, 15));

	//coal
	Rect roi_(cv::Point(372, 101), cv::Point(559, 408));
	Rect roi_2(cv::Point(368, 126), cv::Point(720, 387));
	Rect roi_3(cv::Point(758, 130), cv::Point(336, 497));

	cv::Mat img1 = dispL_filtered(roi_).clone();
	cv::Mat img2 = dispL_filtered(roi_2).clone();
	cv::Mat img3 = dispL_filtered(roi_3).clone();
	cv::Mat img4 = dispR_filtered(roi_).clone();
	cv::Mat img5 = dispR_filtered(roi_2).clone();
	cv::Mat img6 = dispR_filtered(roi_3).clone();

	cv::Mat img_norm;
	normalize(img1, img_norm, 0, 255, NORM_MINMAX);
	img_norm.convertTo(img_norm, CV_8UC1);
	imwrite("roi1_L.jpg", img_norm);
	normalize(img2, img_norm, 0, 255, NORM_MINMAX);
	img_norm.convertTo(img_norm, CV_8UC1);
	imwrite("roi2_L.jpg", img_norm);
	normalize(img3, img_norm, 0, 255, NORM_MINMAX);
	img_norm.convertTo(img_norm, CV_8UC1);
	imwrite("roi3_L.jpg", img_norm);
	normalize(img4, img_norm, 0, 255, NORM_MINMAX);
	img_norm.convertTo(img_norm, CV_8UC1);
	imwrite("roi1_R.jpg", img_norm);
	normalize(img5, img_norm, 0, 255, NORM_MINMAX);
	img_norm.convertTo(img_norm, CV_8UC1);
	imwrite("roi2_R.jpg", img_norm);
	normalize(img6, img_norm, 0, 255, NORM_MINMAX);
	img_norm.convertTo(img_norm, CV_8UC1);
	imwrite("roi3_R.jpg", img_norm);


	//
	Rect ROIL;
	Rect ROIR;
	{
		//ROIL = cv::Rect(cv::Point(279, 88), cv::Point(457, 359));//for data20200107/rectifyL.jpg (3_patterns0)
		//ROIR = cv::Rect(cv::Point(175, 93), cv::Point(353, 359));
	}
	{
		//ROIL = cv::Rect(cv::Point(276, 19), cv::Point(486, 315));//for 2560_1440/rectifyR/23_pattern0
		//ROIR = cv::Rect(cv::Point(168, 21), cv::Point(388, 313));
	}
	{
		//ROIL = cv::Rect(cv::Point(380, 146), cv::Point(1192, 719));//for 2560_2_1440_2/rectifyR/3_pattern0
		//ROIR = cv::Rect(cv::Point(54, 156), cv::Point(853, 719));
	}
	{
		//ROIL = cv::Rect(cv::Point(190, 69), cv::Point(598, 359));//for 2560_1440/rectifyR/3_pattern0
		//ROIR = cv::Rect(cv::Point(28, 76), cv::Point(427, 359));
	}
	{
		//ROIL = cv::Rect(cv::Point(454, 229), cv::Point(1025, 660));//for 2560_2_1440_2/rectifyL/14_pattern0
		//ROIR = cv::Rect(cv::Point(201, 237), cv::Point(785, 600));
	}
	{
		//ROIL = cv::Rect(cv::Point(268, 100), cv::Point(1279, 719));//for 2560_1440/rectifyL/14_pattern0
		//ROIR = cv::Rect(cv::Point(0, 113), cv::Point(922, 719));
	}
	{
		//ROIL = cv::Rect(cv::Point(0, 0), cv::Point(828, 475));//for 2560_2_1440_2/rectifyR/1L//20190924
		//ROIR = cv::Rect(cv::Point(0, 0), cv::Point(823, 480));
	}
	{
		//ROIL = cv::Rect(cv::Point(0, 0), cv::Point(865, 485));//for 2560_2_1440_2/rectifyR/2L//20190924
		//ROIR = cv::Rect(cv::Point(0, 0), cv::Point(856, 485));
	}
	{
		//ROIL = cv::Rect(cv::Point(202, 0), cv::Point(939, 502));//for 2560_2_1440_2/rectifyR/3L//20190924
		//ROIR = cv::Rect(cv::Point(186, 0), cv::Point(932, 508));
	}

	cv::Mat dispL_roi, dispL_filtered_roi;
	dispL_roi = cv::Mat::zeros(imgL.size(), CV_32FC1);
	dispL_filtered_roi = cv::Mat::zeros(imgL.size(), CV_32FC1);
	//
	dispImgL(ROIL).copyTo(dispL_roi(ROIL));
	dispL_filtered(ROIL).copyTo(dispL_filtered_roi(ROIL));

	cv::Mat dispR_roi, dispR_filtered_roi;
	dispR_roi = cv::Mat::zeros(imgR.size(), CV_32FC1);
	dispR_filtered_roi = cv::Mat::zeros(imgR.size(), CV_32FC1);
	//
	dispImgR(ROIR).copyTo(dispR_roi(ROIR));
	dispR_filtered(ROIR).copyTo(dispR_filtered_roi(ROIR));

	cv::Mat roi_norm_filtered_L, roi_norm_filtered_R;
	normalize(dispL_filtered_roi, roi_norm_filtered_L, 0, 255, NORM_MINMAX);
	normalize(dispR_filtered_roi, roi_norm_filtered_R, 0, 255, NORM_MINMAX);
	roi_norm_filtered_L.convertTo(roi_norm_filtered_L, CV_8UC1);
	roi_norm_filtered_R.convertTo(roi_norm_filtered_R, CV_8UC1);
	cv::imwrite("dispL_filtered_roi.jpg", roi_norm_filtered_L);
	cv::imwrite("dispR_filtered_roi.jpg", roi_norm_filtered_R);

	cv::Mat roi_norm_L, roi_norm_R;
	normalize(dispL_roi, roi_norm_L, 0, 255, NORM_MINMAX);
	normalize(dispR_roi, roi_norm_R, 0, 255, NORM_MINMAX);
	roi_norm_L.convertTo(roi_norm_L, CV_8UC1);
	roi_norm_R.convertTo(roi_norm_R, CV_8UC1);
	cv::imwrite("dispL_roi.jpg", roi_norm_L);
	cv::imwrite("dispR_roi.jpg", roi_norm_R);



	//point cloud show
	std::string sysParams_path = "D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/data20200107/stereoRes.xml";
	//std::string sysParams_path = "D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/temp_for_industry/fisheyeStereoCalib/20191017-1-2/stereoRes.xml";
	showPointCloudVisual_my2(imgL, dispL_filtered, sysParams_path, "filteredL_", true);
	showPointCloudVisual_my2(imgL, dispImgL, sysParams_path, "L_", true);
	showPointCloudVisual_my2(imgR, dispR_filtered, sysParams_path, "filteredR_", false);
	showPointCloudVisual_my2(imgR, dispImgR, sysParams_path, "R_", false);

	showPointCloudVisual_my3(imgL, dispL_filtered_roi, sysParams_path, "filteredL_roi_", true);
	showPointCloudVisual_my3(imgL, dispL_roi, sysParams_path, "L_roi_", true);
	showPointCloudVisual_my3(imgR, dispR_filtered_roi, sysParams_path, "filteredR_roi_", false);
	showPointCloudVisual_my3(imgR, dispR_roi, sysParams_path, "R_roi_", false);

	waitKey(0);
	return 0;
}

//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//using namespace cv;
//
//void main()
//{
//	std::string url_left = "rtsp://admin:yanfa1304@192.168.43.6:554/80";
//	std::string url_right = "rtsp://admin:yanfa1304@192.168.43.178:554/80";
//
//	VideoCapture captureL(url_left);//如果是笔记本，0打开的是自带的摄像头，1 打开外接的相机
//	VideoCapture captureR(url_right);
//	//double rate = capture.get(CV_CAP_PROP_FPS);//视频的帧率
//
//	std::string videoPathL = ".\\videos\\VideoTest_6_20190627L.avi";
//	double rateL = 25.0;
//	Size videoSizeL((int)captureL.get(CV_CAP_PROP_FRAME_WIDTH), (int)captureL.get(CV_CAP_PROP_FRAME_HEIGHT));
//	VideoWriter writerL(videoPathL, CV_FOURCC('M', 'J', 'P', 'G'), rateL, videoSizeL);
//
//	std::string videoPathR = ".\\videos\\VideoTest_6_20190627R.avi";
//	double rateR = 25.0;
//	Size videoSizeR((int)captureR.get(CV_CAP_PROP_FRAME_WIDTH), (int)captureR.get(CV_CAP_PROP_FRAME_HEIGHT));
//	VideoWriter writerR(videoPathR, CV_FOURCC('M', 'J', 'P', 'G'), rateR, videoSizeR);
//
//	Mat frameL, frameR;
//
//	while (captureL.isOpened() && captureR.isOpened())
//	{
//		captureL >> frameL;
//		if(frameL.empty())
//		{
//			continue;
//		}
//		writerL << frameL;
//		imshow("videoL", frameL);
//
//		captureL >> frameR;
//		if (frameR.empty())
//		{
//			continue;
//		}
//		writerR << frameR;
//		imshow("videoR", frameR);
//
//		if (waitKey(20) == 27)//27是键盘摁下esc时，计算机接收到的ascii码值
//		{
//			break;
//		}
//	}
//}

//void print_my(int b)
//{
//	//omp_set_num_threads(4);
//#pragma omp parallel for
//	for(int i = 0; i < b; i++)
//	{
//		printf("total %d, i = %d, I am Thread %d\n", b, i, omp_get_thread_num());
//	}
//}
//
//int main()
//{
//	int a[8] = { 10, 11, 8, 9 , 7 , 3, 2, 4};
//
//	omp_set_num_threads(4);
//#pragma omp parallel for
//	for (int i = 0; i < 8; i++)
//	{
//		//print_my(a[i]);
//		for (int j = 0; j < a[i]; j++)
//		{
//			printf("total %d, j = %d, I am Thread %d\n", a[i], j, omp_get_thread_num());
//		}
//	}
//
//	waitKey(0);
//	return 0;
//}