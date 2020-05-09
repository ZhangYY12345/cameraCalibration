#include <pcl/visualization/cloud_viewer.h>
#include "methods.h"
#include "method_StereoMatching.h"
#include "method_pcl_filters.h"
#include "method_pcl_keypoint.h"
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
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>

using namespace cv;

int user_data;

/**
 * \brief open the target camera and capture images for calibration
 *			\notice that the location of the camera is supposed to be fixed during the capturing
 */
void myCameraCalibration(std::string cameraParaPath)
{
	VideoCapture cap;
	cap.open(0);
	if(!cap.isOpened())
	{
		std::cout << "Camera open failed!" << std::endl;
		return;
	}
	cap.set(CAP_PROP_FOURCC, 'GPJM');
	Size imgSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));

	Size patternSize(5, 7);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;


	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec;		//store the detected inner corners of each image
	Mat img;
	Mat imgGrey;
	while(cornerPtsVec.size() < 20)
	{
		cap.read(img);
		cvtColor(img, imgGrey, COLOR_BGR2GRAY);

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(imgGrey, patternSize, cornerPts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cornerSubPix(imgGrey, cornerPts, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			cornerPtsVec.push_back(cornerPts);
			drawChessboardCorners(imgGrey, patternSize, cornerPts, patternFound);
			cornerPtsVec.push_back(img);
		}
	}
	cap.release();


	//camera calibration
	Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler
	
	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	Mat cameraMatrixInnerPara = Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	Mat cameraMatrixDistPara = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT;									//matrix T of each image
	std::vector<Mat> vectorMatR;									//matrix R of each image

	for (std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec.begin(); itor != cornerPtsVec.end(); itor++)
	{
		std::vector<Point3f> tempPts;
		for (int i = 0; i < patternSize.height; i++)
		{
			for (int j = 0; j < patternSize.width; j++)
			{
				Point3f realPt;
				realPt.x = i * squareSize.width;
				realPt.y = j * squareSize.height;
				realPt.z = 0;
				tempPts.push_back(realPt);
			}
		}
		objPts3d.push_back(tempPts);
	}

	calibrateCamera(objPts3d, cornerPtsVec, imgSize, cameraMatrixInnerPara, cameraMatrixDistPara, vectorMatR, vectorMatT, 0);


	//evaluate the result of the camera calibration,calculate the error of calibration in each image
	double totalErr = 0.0;
	double err = 0.0;
	std::vector<Point2f> imgPts_2d;		//store the rechecked points' coordination
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		std::vector<Point3f> tempPts = objPts3d[i];  //the actual coordination of point in 3d corrdinate system
		projectPoints(tempPts, vectorMatR[i], vectorMatT[i], cameraMatrixInnerPara, cameraMatrixDistPara, imgPts_2d);

		//calculate the error
		std::vector<Point2f> tempImagePoint = cornerPtsVec[i]; //the detected corner coordination in the image
		Mat tempImgPt = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat recheckImgPt = Mat(1, imgPts_2d.size(), CV_32FC2);

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			recheckImgPt.at<Vec2f>(0, j) = Vec2f(imgPts_2d[j].x, imgPts_2d[j].y);
			tempImgPt.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(recheckImgPt, tempImgPt, NORM_L2);
		totalErr += err / gridPatternNum;
	}

	std::cout << "总平均误差：" << totalErr / cornerPtsVec.size() << std::endl;


	//output the calibration result
	std::cout << "相机内参数矩阵：\n" << cameraMatrixInnerPara << std::endl;
	std::cout << "相机畸变参数[k1,k2,k3,p1,p2]:\n" << cameraMatrixDistPara << std::endl;
	Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		Rodrigues(vectorMatR[i], rotationMatrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：\n" << rotationMatrix << std::endl << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移矩阵：\n" << vectorMatT[i] << std::endl << std::endl << std::endl;
	}

	//store the calibration result to the .xml file
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "CameraInnerPara" << cameraMatrixInnerPara;
	fn << "CameraDistPara" << cameraMatrixDistPara;
	fn.release();
}

/**
 * \brief using pre-captured image to calibrate the camera which is used to capture these image
 *			\notice that the location of the camera is supposed to be fixed during the pre-capturing
 * \param imgFilePath :the path of the folder to store the images
 */
void myCameraCalibration(std::string imgFilePath, std::string cameraParaPath)
{
	//load all the images in the folder
	String filePath = imgFilePath + "/*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	Size patternSize(9, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = 54;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec;		//store the detected inner corners of each image
	for(int i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i], IMREAD_GRAYSCALE);
		if(i == 0)
		{
			imgSize.width = img.rows;
			imgSize.height = img.cols;
		}

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(img, patternSize, cornerPts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
																						   + CALIB_CB_FAST_CHECK);
		if(patternFound)
		{
			cornerSubPix(img, cornerPts, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			cornerPtsVec.push_back(cornerPts);
			drawChessboardCorners(img, patternSize, cornerPts, patternFound);
		}
	}


	//camera calibration
	Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler
	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	Mat cameraMatrixInnerPara = Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	Mat cameraMatrixDistPara = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT;									//matrix T of each image
	std::vector<Mat> vectorMatR;									//matrix R of each image
	
	for(std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec.begin(); itor != cornerPtsVec.end(); itor++)
	{
		std::vector<Point3f> tempPts;
		for(int i = 0; i < patternSize.height; i++)
		{
			for(int j = 0; j < patternSize.width; j++)
			{
				Point3f realPt;
				realPt.x = i * squareSize.width;
				realPt.y = j * squareSize.height;
				realPt.z = 0;
				tempPts.push_back(realPt);
			}
		}
		objPts3d.push_back(tempPts);
	}

	calibrateCamera(objPts3d, cornerPtsVec, imgSize, cameraMatrixInnerPara, cameraMatrixDistPara, vectorMatR, vectorMatT, 0);


	//evaluate the result of the camera calibration,calculate the error of calibration in each image
	double totalErr = 0.0;
	double err = 0.0;
	std::vector<Point2f> imgPts_2d;		//store the rechecked points' coordination
	for(int i = 0; i < cornerPtsVec.size(); i++)
	{
		std::vector<Point3f> tempPts = objPts3d[i];  //the actual coordination of point in 3d corrdinate system
		projectPoints(tempPts, vectorMatR[i], vectorMatT[i], cameraMatrixInnerPara, cameraMatrixDistPara, imgPts_2d);
		
		//calculate the error
		std::vector<Point2f> tempImagePoint = cornerPtsVec[i]; //the detected corner coordination in the image
		Mat tempImgPt = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat recheckImgPt = Mat(1, imgPts_2d.size(), CV_32FC2);

		for(int j = 0; j < tempImagePoint.size(); j++)
		{
			recheckImgPt.at<Vec2f>(0, j) = Vec2f(imgPts_2d[j].x, imgPts_2d[j].y);
			tempImgPt.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(recheckImgPt, tempImgPt, NORM_L2);
		totalErr += err / gridPatternNum;
	}

	std::cout << "总平均误差：" << totalErr / cornerPtsVec.size() << std::endl;


	//output the calibration result
	std::cout << "相机内参数矩阵：\n" << cameraMatrixInnerPara << std::endl;
	std::cout << "相机畸变参数[k1,k2,k3,p1,p2]:\n" << cameraMatrixDistPara << std::endl;
	Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	for(int i = 0; i < cornerPtsVec.size(); i++)
	{
		Rodrigues(vectorMatR[i], rotationMatrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：\n" << rotationMatrix << std::endl << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移矩阵：\n" << vectorMatT[i] << std::endl << std::endl << std::endl;
	}

	//store the calibration result to the .xml file
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "CameraInnerPara" << cameraMatrixInnerPara;
	fn << "CameraDistPara" << cameraMatrixDistPara;
	fn.release();
}

/**
 * \brief using pre-calibrated camera inner parameters and distort parameters to undistort the images captured by the camera
 * \param cameraParaPath :the .xml file path of pre-calculated parameters of camera
 */
void myCameraUndistort(std::string cameraParaPath)
{
	Mat cameraMatrixInnerPara;
	Mat cameraMatrixDistPara;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["CameraInnerPara"] >> cameraMatrixInnerPara;
	fn["CameraDistPara"] >> cameraMatrixDistPara;
	fn.release();


	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cout << "Camera open failed!" << std::endl;
		return;
	}
	cap.set(CAP_PROP_FOURCC, 'GPJM');

	while(1)
	{
		Mat img;
		Mat undistortImg;
		cap.read(img);
		imshow("originView", img);
		undistort(img, undistortImg, cameraMatrixInnerPara, cameraMatrixDistPara);
		imshow("undistortView", undistortImg);
		waitKey(10);
	}
	cap.release();
}

/**
 * \brief using pre-calibrated camera inner parameters and distort parameters to undistort the images captured by the camera
 * \param imgFilePath :the file path of images to be undistored,which is captured by the particular camera
 * \param cameraParaPath :the .xml file path of pre-calculated parameters of camera
 */
void myCameraUndistort(std::string imgFilePath, std::string cameraParaPath)
{
	Mat cameraMatrixInnerPara;
	Mat cameraMatrixDistPara;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["CameraInnerPara"] >> cameraMatrixInnerPara;
	fn["CameraDistPara"] >> cameraMatrixDistPara;
	fn.release();

	String filePath = imgFilePath + "/*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	Mat undistortImg;
	std::vector<Mat> undistortImgs;
	for(int i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i]);
		undistort(img, undistortImg, cameraMatrixInnerPara, cameraMatrixDistPara);
		imwrite(imgFilePath + "/" + std::to_string(i) + ".jpg", undistortImg);
		undistortImgs.push_back(undistortImg);
	}
}

/**
 * \brief two camera calibration for stereo vision by openning camera and capturing the chess board images
 * \param cameraParaPath 
 */
void twoCamerasCalibration(std::string cameraParaPath)
{
	//openning the two cameras:left camera, right camera
	VideoCapture cap_left, cap_right;
	cap_left.open(0);
	if (!cap_left.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_left.set(CAP_PROP_FOURCC, 'GPJM');
	cap_left.set(CAP_PROP_FRAME_HEIGHT, 480);		//rows
	cap_left.set(CAP_PROP_FRAME_WIDTH, 640);			//cols

	cap_right.open(1);
	if (!cap_right.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_right.set(CAP_PROP_FOURCC, 'GPJM');
	cap_right.set(CAP_PROP_FRAME_HEIGHT, cap_left.get(CAP_PROP_FRAME_HEIGHT));
	cap_right.set(CAP_PROP_FRAME_WIDTH, cap_left.get(CAP_PROP_FRAME_WIDTH));

	Size imgSize(cap_left.get(CAP_PROP_FRAME_WIDTH), cap_left.get(CAP_PROP_FRAME_HEIGHT));


	Size patternSize(9, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;


	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	Mat img_left, img_right;
	Mat imgGrey_left, imgGrey_right;
	while (cornerPtsVec_left.size() < 20)
	{
		cap_left.read(img_left);
		cvtColor(img_left, imgGrey_left, COLOR_BGR2GRAY);
		cap_right.read(img_right);
		cvtColor(img_right, imgGrey_right, COLOR_BGR2GRAY);

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(imgGrey_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(imgGrey_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left;
		std::vector<Point2f> cornerPts_right;

		bool patternFound_left = findChessboardCorners(imgGrey_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			+ CALIB_CB_NORMALIZE_IMAGE);
		bool patternFound_right = findChessboardCorners(imgGrey_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			+ CALIB_CB_NORMALIZE_IMAGE);

		if (patternFound_left && patternFound_right)
		{
			//for (int k = 0; k < cornerPts_left.size(); k++)
			//{
			//	cornerPts_left[k].x /= 2.0;
			//	cornerPts_left[k].y /= 2.0;
			//}

			//for (int k = 0; k < cornerPts_right.size(); k++)
			//{
			//	cornerPts_right[k].x /= 2.0;
			//	cornerPts_right[k].y /= 2.0;
			//}

			cornerSubPix(imgGrey_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(imgGrey_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(imgGrey_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(imgGrey_right, patternSize, cornerPts_right, patternFound_right);
		}
	}
	cap_left.release();
	cap_right.release();

	//two cameras calibration
	Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	for (std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec_left.begin(); itor != cornerPtsVec_left.end(); itor++)
	{
		std::vector<Point3f> tempPts;
		for (int i = 0; i < patternSize.height; i++)
		{
			for (int j = 0; j < patternSize.width; j++)
			{
				Point3f realPt;
				realPt.x = i * squareSize.width;
				realPt.y = j * squareSize.height;
				realPt.z = 0;
				tempPts.push_back(realPt);
			}
		}
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat(3, 3, CV_32FC1, Scalar::all(0));		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image
	std::vector<Mat> vectorMatR_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize, 
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	Mat cameraMatrixInnerPara_right = Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize, 
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F
	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right, 
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, 
		imgSize, matrixR, matrixT, matrixE, matrixF, CALIB_FIX_INTRINSIC | CALIB_SAME_FOCAL_LENGTH,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0));
	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "Essential_Matrix" << matrixE;
	fn << "Fundamental_Matrix" << matrixF;
	fn.release();

	double err = 0;
	int npoints = 0;
	std::vector<Vec3f> lines[2];
	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		int npt = (int)cornerPtsVec_left[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(cornerPtsVec_left[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrixInnerPara_left, cameraMatrixDistPara_left, Mat(), cameraMatrixInnerPara_left);
		computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

		imgpt[1] = Mat(cornerPtsVec_right[i]);
		undistortPoints(imgpt[1], imgpt[1], cameraMatrixInnerPara_right, cameraMatrixDistPara_right, Mat(), cameraMatrixInnerPara_right);
		computeCorrespondEpilines(imgpt[1], 1 + 1, matrixF, lines[1]);

		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(cornerPtsVec_left[i][j].x*lines[1][j][0] +
				cornerPtsVec_left[i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(cornerPtsVec_right[i][j].x*lines[0][j][0] +
					cornerPtsVec_right[i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	std::cout << "average epipolar err = " << err / npoints << std::endl;
}

/**
 * \brief two camera calibration for stereo vision with pre-captured images
 * \param imgFilePath :the path of image pairs captured by left and right cameras, for stereo camera calibration
 * \param cameraParaPath :the path of files storing the camera parameters
 */
void twoCamerasCalibration(std::string imgFilePath, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
	//load all the images in the folder
	String filePath = imgFilePath + "/cal_left_*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	Size patternSize(8, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat img_left = imread(fileNames[i], IMREAD_GRAYSCALE );
		Mat img_right = imread(fileNames[i].substr(0, fileNames[i].length() - 11) + "right_"
			+ fileNames[i].substr(fileNames[i].length() - 6, 6), IMREAD_GRAYSCALE );
		
		if(img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.rows;
			imgSize.height = img_left.cols;
		}
		
		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(img_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(img_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE);
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE);
		if (patternFound_left && patternFound_right)
		{
			//for(int k = 0; k < cornerPts_left.size(); k++)
			//{
			//	cornerPts_left[k].x /= 2.0;
			//	cornerPts_left[k].y /= 2.0;
			//}

			//for (int k = 0; k < cornerPts_right.size(); k++)
			//{
			//	cornerPts_right[k].x /= 2.0;
			//	cornerPts_right[k].y /= 2.0;
			//}

			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(20.2222, 38.2857);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	for (std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec_left.begin(); itor != cornerPtsVec_left.end(); itor++)
	{
		std::vector<Point3f> tempPts;
		for (int i = 0; i < patternSize.height; i++)
		{
			for (int j = 0; j < patternSize.width; j++)
			{
				Point3f realPt;
				realPt.x = i * squareSize.width;
				realPt.y = j * squareSize.height;
				realPt.z = 0;
				tempPts.push_back(realPt);
			}
		}
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image
	std::vector<Mat> vectorMatR_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST 
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F
	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, CALIB_FIX_INTRINSIC | CALIB_SAME_FOCAL_LENGTH,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "EssentialMat" << matrixE;
	fn << "FundamentalMat" << matrixF;
	fn.release();

	double err = 0;
	int npoints = 0;
	std::vector<Vec3f> lines[2];
	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		int npt = (int)cornerPtsVec_left[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(cornerPtsVec_left[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrixInnerPara_left, cameraMatrixDistPara_left, Mat(), cameraMatrixInnerPara_left);
		computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

		imgpt[1] = Mat(cornerPtsVec_right[i]);
		undistortPoints(imgpt[1], imgpt[1], cameraMatrixInnerPara_right, cameraMatrixDistPara_right, Mat(), cameraMatrixInnerPara_right);
		computeCorrespondEpilines(imgpt[1], 1 + 1, matrixF, lines[1]);

		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(cornerPtsVec_left[i][j].x*lines[1][j][0] +
				cornerPtsVec_left[i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(cornerPtsVec_right[i][j].x*lines[0][j][0] +
					cornerPtsVec_right[i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	std::cout << "average epipolar err = " << err / npoints << std::endl;

}

void twoCamerasCalibration(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
//load all the images in the folder
	String filePathL = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNamesL;
	glob(filePathL, fileNamesL, false);
	Size patternSize(9, 6);		//表示棋盘行和列的内角数（行，列）	//9:the number of inner corners in each row of the chess board
																					//6:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNamesL.size(); i++)
	{
		Mat img_left = imread(fileNamesL[i], IMREAD_GRAYSCALE );
		Mat img_right = imread(fileNamesL[i].substr(0, fileNamesL[i].length() - 5) + "R.jpg"
			, IMREAD_GRAYSCALE );

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols;
			imgSize.height = img_left.rows;
		}

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(img_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(img_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);//函数定位棋盘图案的内部棋盘角，若找到所有角，并且它们按特定顺序放置（逐行，每行从左到右），则函数返回非零值，否则返回0
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	std::vector<Point3f> tempPts;
	//for (int i = patternSize.height - 1; i >= 0; i--)
	//{
	//	for (int j = 0; j < patternSize.width; j++)
	//	{
	//		Point3f realPt;
	//		realPt.x = j * squareSize.width;
	//		realPt.y = i * squareSize.height;
	//		realPt.z = 0;
	//		tempPts.push_back(realPt);
	//	}
	//}
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}

	for(int i = 0; i < cornerPtsVec_right.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image
	std::vector<Mat> vectorMatR_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		CALIB_FIX_PRINCIPAL_POINT
		| CALIB_RATIONAL_MODEL);// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right, 
		CALIB_FIX_PRINCIPAL_POINT 
		| CALIB_RATIONAL_MODEL );//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F
	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, CALIB_FIX_INTRINSIC,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 1e-6));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "EssentialMat" << matrixE;
	fn << "FundamentalMat" << matrixF;
	fn.release();

	//double err = 0;
	//int npoints = 0;
	//std::vector<Vec3f> lines[2];
	//for (int i = 0; i < cornerPtsVec_left.size(); i++)
	//{
	//	int npt = (int)cornerPtsVec_left[i].size();
	//	Mat imgpt[2];
	//	imgpt[0] = Mat(cornerPtsVec_left[i]);
	//	undistortPoints(imgpt[0], imgpt[0], cameraMatrixInnerPara_left, cameraMatrixDistPara_left, Mat(), cameraMatrixInnerPara_left);
	//	computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

	//	imgpt[1] = Mat(cornerPtsVec_right[i]);
	//	undistortPoints(imgpt[1], imgpt[1], cameraMatrixInnerPara_right, cameraMatrixDistPara_right, Mat(), cameraMatrixInnerPara_right);
	//	computeCorrespondEpilines(imgpt[1], 1 + 1, matrixF, lines[1]);

	//	for (int j = 0; j < npt; j++)
	//	{
	//		double errij = fabs(cornerPtsVec_left[i][j].x*lines[1][j][0] +
	//			cornerPtsVec_left[i][j].y*lines[1][j][1] + lines[1][j][2]) +
	//			fabs(cornerPtsVec_right[i][j].x*lines[0][j][0] +
	//				cornerPtsVec_right[i][j].y*lines[0][j][1] + lines[0][j][2]);
	//		err += errij;
	//	}
	//	npoints += npt;
	//}
	//std::cout << "average epipolar err = " << err / npoints << std::endl;
}

cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r)
{
	CV_Assert(l.type() == r.type() && l.size() == r.size());
	cv::Mat merged(l.rows, l.cols * 2, l.type());
	cv::Mat lpart = merged.colRange(0, l.cols);
	cv::Mat rpart = merged.colRange(l.cols, merged.cols);
	l.copyTo(lpart);
	r.copyTo(rpart);

	for (int i = 0; i < l.rows; i += 20)
		cv::line(merged, cv::Point(0, i), cv::Point(merged.cols, i), cv::Scalar(0, 255, 0));

	return merged;
}

void stereoFisheyeCamCalib(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
//load all the images in the folder
	String filePathL = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNamesL;
	glob(filePathL, fileNamesL, false);
	Size patternSize(9, 6);		//表示棋盘行和列的内角数（行，列）	//9:the number of inner corners in each row of the chess board
																					//6:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2d>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNamesL.size(); i++)
	{
		Mat img_left = imread(fileNamesL[i], IMREAD_GRAYSCALE );
		Mat img_right = imread(fileNamesL[i].substr(0, fileNamesL[i].length() - 5) + "R.jpg"
			, IMREAD_GRAYSCALE );

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols;
			imgSize.height = img_left.rows;
		}

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(img_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(img_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);//函数定位棋盘图案的内部棋盘角，若找到所有角，并且它们按特定顺序放置（逐行，每行从左到右），则函数返回非零值，否则返回0
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			//获取角点更精细的检测结果
			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_left.push_back(VecPointF2D(cornerPts_left));
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_right.push_back(VecPointF2D(cornerPts_right));
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3d>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	std::vector<Point3d> tempPts;
	//for (int i = patternSize.height - 1; i >= 0; i--)
	//{
	//	for (int j = 0; j < patternSize.width; j++)
	//	{
	//		Point3f realPt;
	//		realPt.x = j * squareSize.width;
	//		realPt.y = i * squareSize.height;
	//		realPt.z = 0;
	//		tempPts.push_back(realPt);
	//	}
	//}
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3d realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}

	for (int i = 0; i < cornerPtsVec_right.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image:translation
	std::vector<Mat> vectorMatR_left;										//matrix R of each image:rotation
	double rmsLeft = fisheye::calibrate(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW, 
		cv::TermCriteria(3, 20, 1e-6));// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = fisheye::calibrate(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 20, 1e-6));//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F

	double rms = fisheye::stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT, 
		fisheye::CALIB_USE_INTRINSIC_GUESS | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-12));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "EssentialMat" << matrixE;
	fn << "FundamentalMat" << matrixF;
	fn.release();

}

void stereoFisheyCamCalibRecti(std::string imgFilePathL, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
	//load all the images in the folder
	String filePathL = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNamesL;
	glob(filePathL, fileNamesL, false);
	Size patternSize(9, 6);		//表示棋盘行和列的内角数（行，列）	//9:the number of inner corners in each row of the chess board
																					//6:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2d>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNamesL.size(); i++)
	{
		Mat img_left = imread(fileNamesL[i], IMREAD_GRAYSCALE );
		Mat img_right = imread(fileNamesL[i].substr(0, fileNamesL[i].length() - 5) + "R.jpg"
			, IMREAD_GRAYSCALE );

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols;
			imgSize.height = img_left.rows;
		}

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);//函数定位棋盘图案的内部棋盘角，若找到所有角，并且它们按特定顺序放置（逐行，每行从左到右），则函数返回非零值，否则返回0
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			//获取角点更精细的检测结果
			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_left.push_back(VecPointF2D(cornerPts_left));
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_right.push_back(VecPointF2D(cornerPts_right));
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3d> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	std::vector<Point3d> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3d realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}

	for (int i = 0; i < cornerPtsVec_right.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image:translation
	std::vector<Mat> vectorMatR_left;										//matrix R of each image:rotation
	double rmsLeft = fisheye::calibrate(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 100, 1e-6));// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = fisheye::calibrate(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 100, 1e-6));//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	//Mat monoMapL1, monoMapL2, monoMapR1, monoMapR2;
	////Precompute maps for cv::remap()
	//fisheye::initUndistortRectifyMap(cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
	//	noArray(), cameraMatrixInnerPara_left,
	//	imgSize, CV_32F, monoMapL1, monoMapL2);

	//fisheye::initUndistortRectifyMap(cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
	//	noArray(), cameraMatrixInnerPara_right,
	//	imgSize, CV_32F, monoMapR1, monoMapR2);


	//for (int i = 0; i < cornerPtsVec_left.size(); i++)
	//{
	//	fisheye::undistortPoints(cornerPtsVec_left[i], cornerPtsVec_left[i],
	//		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
	//		noArray(), cameraMatrixInnerPara_left);

	//	fisheye::undistortPoints(cornerPtsVec_right[i], cornerPtsVec_right[i],
	//		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
	//		noArray(), cameraMatrixInnerPara_right);
	//}

	Mat K1, D1, K2, D2;
	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat zeroDistortion = Mat::zeros(cameraMatrixDistPara_right.size(), cameraMatrixDistPara_right.type());

	int flag = 0;
	flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flag |= cv::fisheye::CALIB_CHECK_COND;
	flag |= cv::fisheye::CALIB_FIX_SKEW;
	double rms = fisheye::stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, zeroDistortion,
		cameraMatrixInnerPara_right, zeroDistortion,
		//K1, D1, K2, D2,
		imgSize, matrixR, matrixT,
		flag);

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << K1;
	fn << "Left_CameraDistPara" << D1;
	fn << "Right_CameraInnerPara" << K2;
	fn << "Right_CameraDistPara" << D2;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn.release();

}

/**
 * \brief using pre-calibrated parameters to rectify the images captured by left and right cameras in a real-time manner
 * \param cameraParaPath :the .xml file path of the pre-calculated camera calibration parameters
 */
void stereoCameraUndistort(std::string cameraParaPath)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();


	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0.0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;  //检验校正是否正确
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3); //高度一样，宽度双倍
	}
	else
	{
		sf = 300. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3); //高度双倍，宽度一样
	}

	destroyAllWindows();

	VideoCapture cap_left, cap_right;
	cap_left.open(0);
	if (!cap_left.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_left.set(CAP_PROP_FOURCC, 'GPJM');
	cap_left.set(CAP_PROP_FRAME_HEIGHT, imgSize.height);		//rows
	cap_left.set(CAP_PROP_FRAME_WIDTH, imgSize.width);			//cols

	cap_right.open(1);
	if (!cap_right.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_right.set(CAP_PROP_FOURCC, 'GPJM');
	cap_right.set(CAP_PROP_FRAME_HEIGHT, cap_left.get(CAP_PROP_FRAME_HEIGHT));
	cap_right.set(CAP_PROP_FRAME_WIDTH, cap_left.get(CAP_PROP_FRAME_WIDTH));



	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setMinDisparity(0);
	//sbm->setNumDisparities(64);
	sbm->setTextureThreshold(10);
	sbm->setDisp12MaxDiff(-1);
	sbm->setPreFilterCap(31);
	sbm->setUniquenessRatio(25);
	sbm->setSpeckleRange(32);
	sbm->setSpeckleWindowSize(100);


	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

	Mat frame_left, frame_right;
	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;
	while (true)
	{
		cap_left >> frame_left;
		cap_right >> frame_right;

		if (frame_left.empty() || frame_right.empty())
			continue;

		remap(frame_left, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h)); //浅拷贝
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA); //INTER_AREA：插值方法
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(frame_right, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h)); //浅拷贝
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;		//all pixels in the roi are valid in both left and right view maps

		imgLeft = canvasPart1(vroi).clone();  //rectified images with only valid pixels
		imgRight = canvasPart2(vroi).clone();

		//draw the valid rectangle in two views respectively
		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8); 
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		//draw lines for verification quality evaluation
		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 32)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 32)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		//the input image for stereo matching is supposed to be 8-bit single channel image
		cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);

		//-- And create the image in which we will save our disparities
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S); //for BM
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S); //for SGBM
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl; 
			return;
		}

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		Mat disparityShow;
		imgDisparity8U.copyTo(disparityShow, Mask);




		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		Mat  sgbmDisparityShow;
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		imshow("bmDisparity", disparityShow);
		imshow("sgbmDisparity", sgbmDisparityShow);
		imshow("rectified", canvas);
		char c = (char)waitKey(1);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

/**
 * \brief using pre-calibrated parameters to rectify the images pre-captured by left and right cameras
 * \param imgFilePath 
 * \param cameraParaPath 
 */
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();

	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right;
	Rect validRoi[2];

	Mat R_L, R_R, P_L, P_R, Q;
	fisheye::stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, R_L, R_R, P_L, P_R, Q,
		CALIB_ZERO_DISPARITY, imgSize, 0.0, 1.1);

	//// OpenCV can handle left-right
	//// or up-down camera arrangements
	//bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat lmapx, lmapy, rmapx, rmapy;
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	fisheye::initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left,
		R_L, P_L,
		imgSize*2, CV_32F, lmapx, lmapy);

	fisheye::initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right,
		R_R, P_R,
		imgSize*2, CV_32F, rmapx, rmapy);

	//Mat canvas;
	//double sf;
	//int w, h;
	//if (!isVerticalStereo)
	//{
	//	sf = 600. / MAX(imgSize.width, imgSize.height);
	//	w = cvRound(imgSize.width*sf);
	//	h = cvRound(imgSize.height*sf);
	//	canvas.create(h, w * 2, CV_8UC3);
	//}
	//else
	//{
	//	sf = 300. / MAX(imgSize.width, imgSize.height);
	//	w = cvRound(imgSize.width*sf);
	//	h = cvRound(imgSize.height*sf);
	//	canvas.create(h * 2, w, CV_8UC3);
	//}

	//destroyAllWindows();


	//Stereo matching
	int ndisparities = 16 * 15;   /**< Range of disparity */
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
	//Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	//sbm->setMinDisparity(0);			//确定匹配搜索从哪里开始，默认为0
	////sbm->setNumDisparities(64);		//在该数值确定的视差范围内进行搜索，视差窗口
	//								//即，最大视差值与最小视差值之差，大小必须为16的整数倍
	//sbm->setTextureThreshold(10);		//保证有足够的纹理以克服噪声
	//sbm->setDisp12MaxDiff(-1);			//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异，默认为-1
	//sbm->setPreFilterCap(31);			//
	//sbm->setUniquenessRatio(25);		//使用匹配功能模式
	//sbm->setSpeckleRange(32);			//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	//sbm->setSpeckleWindowSize(100);		//检查视差连通区域变化度的窗口大小，值为0时取消speckle检查


	//Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
	//	10 * 7 * 7,
	//	40 * 7 * 7,
	//	1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

	Mat frame_left, frame_right;
	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;

	String filePath = imgFilePath + "\\*L.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	for(int i = 0; i < fileNames.size(); i++)
	{
		frame_left = imread(fileNames[i]);
		frame_right = imread(fileNames[i].substr(0, fileNames[i].length() - 5) + "R.jpg");

		if (frame_left.rows != frame_right.rows 
			&& frame_left.cols != frame_right.cols 
			&& frame_left.rows != imgSize.height 
			&& frame_left.cols != imgSize.width)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (frame_left.empty() || frame_right.empty())
			continue;

		//cv::rectangle(frame_left, cv::Rect(255, 0, 829, frame_left.rows - 1), cv::Scalar(0, 0, 255));
		//cv::rectangle(frame_right, cv::Rect(255, 0, 829, frame_left.rows - 1), cv::Scalar(0, 0, 255));
		//cv::rectangle(frame_right, cv::Rect(255 - ndisparities, 0, 829 + ndisparities, frame_left.rows - 1), cv::Scalar(0, 0, 255));

		Mat lundist, rundist;
		cv::remap(frame_left, lundist, lmapx, lmapy, INTER_LINEAR);
		cv::remap(frame_right, rundist, rmapx, rmapy, cv::INTER_LINEAR);

		cv::Mat rectification = mergeRectification(lundist, rundist);

		//imgLeft = canvasPart1(vroi).clone();
		//imgRight = canvasPart2(vroi).clone();

		//rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		//rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		//if (!isVerticalStereo)
		//	for (int j = 0; j < canvas.rows; j += 32)
		//		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		//else
		//	for (int j = 0; j < canvas.cols; j += 32)
		//		line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		//Mat imgLeftBGR = imgLeft.clone();
		//Mat imgRightBGR = imgRight.clone();

		//cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		//cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


		////-- And create the image in which we will save our disparities
		//Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		//Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		//Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		//Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		//if (imgLeft.empty() || imgRight.empty())
		//{
		//	std::cout << " --(!) Error reading images " << std::endl;
		//	return;
		//}

		//sbm->compute(imgLeft, imgRight, imgDisparity16S);

		//imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		//cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		//applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		//Mat disparityShow;
		//imgDisparity8U.copyTo(disparityShow, Mask);


		//sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		//sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		//cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		//applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		//Mat  sgbmDisparityShow;
		//sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		//imshow("bmDisparity", disparityShow);
		//imshow("sgbmDisparity", sgbmDisparityShow);
		//imshow("rectified", canvas);

		//showPointCloud(imgLeftBGR, imgDisparity16S, cameraParaPath);

		char c = (char)waitKey(0);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();


	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	destroyAllWindows();


	//Stereo matching
	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setMinDisparity(0);			//确定匹配搜索从哪里开始，默认为0
										//sbm->setNumDisparities(64);		//在该数值确定的视差范围内进行搜索，视差窗口
										//即，最大视差值与最小视差值之差，大小必须为16的整数倍
	sbm->setTextureThreshold(10);		//保证有足够的纹理以克服噪声
	sbm->setDisp12MaxDiff(-1);			//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异，默认为-1
	sbm->setPreFilterCap(31);			//
	sbm->setUniquenessRatio(25);		//使用匹配功能模式
	sbm->setSpeckleRange(32);			//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	sbm->setSpeckleWindowSize(100);		//检查视差连通区域变化度的窗口大小，值为0时取消speckle检查


	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

	Mat frame_left, frame_right;
	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;

	String filePath = imgFilePath + "/left*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	for (int i = 0; i < fileNames.size(); i++)
	{
		frame_left = imread(fileNames[i]);
		frame_right = imread(fileNames[i].substr(0, fileNames[i].length() - 10) + "right"
			+ fileNames[i].substr(fileNames[i].length() - 6, 6));

		if (frame_left.rows != frame_right.rows && frame_left.cols != frame_right.cols && frame_left.rows != imgSize.height && frame_left.cols != imgSize.width)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (frame_left.empty() || frame_right.empty())
			continue;

		remap(frame_left, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(frame_right, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;

		imgLeft = canvasPart1(vroi).clone();
		imgRight = canvasPart2(vroi).clone();

		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 32)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 32)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		Mat imgLeftBGR = imgLeft.clone();
		Mat imgRightBGR = imgRight.clone();

		cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


		//-- And create the image in which we will save our disparities
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return;
		}

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		Mat disparityShow;
		imgDisparity8U.copyTo(disparityShow, Mask);


		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		Mat  sgbmDisparityShow;
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		imshow("bmDisparity", disparityShow);
		imshow("sgbmDisparity", sgbmDisparityShow);
		imshow("rectified", canvas);

		showPointCloud(imgLeftBGR, imgDisparity16S, cameraParaPath);

		char c = (char)waitKey(0);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

void getRectifiedImages(cv::Mat imgLeft, cv::Mat imgRight, std::string cameraParaPath, cv::Mat& rectifiedLeft,
	cv::Mat& rectifiedRight)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();


	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	Mat rimg, cimg;
	Mat Mask;

		if (imgLeft.rows != imgRight.rows && imgLeft.cols != imgRight.cols && imgLeft.rows != imgSize.height && imgLeft.cols != imgSize.width)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (imgLeft.empty() || imgRight.empty())
			return;

		remap(imgLeft, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(imgRight, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;

		rectifiedLeft = canvasPart1(vroi).clone();
		rectifiedRight = canvasPart2(vroi).clone();

		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 32)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 32)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
}

/**
 * \brief to compute the disparity map with two rectified images from left and right cameras respectively
 *		  \notice:the higher disparity means the point in the world coordination is closer to the camera optical center
 * \param srcLeft :rectified image from left camera(reference image)
 * \param srcRight :rectified image from right camera(target image)
 * \param disparityMap :computed disparity image
 * \param algorithmType :the algorithm used to compute the disparity map
 */
void stereoMatching(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap, StereoMatchingAlgorithms algorithmType)
{
	switch (algorithmType)
	{
	case BM:
		getDisparity_BM(srcLeft, srcRight, disparityMap);
		break;
	case SGBM:
		getDisparity_SGBM(srcLeft, srcRight, disparityMap);
		break;
	case ADAPTIVE_WEIGHT:
		disparityMap = stereomatch_1::computeAdaptiveWeight(srcLeft, srcRight,DISPARITY_LEFT, 15, 0, 64);
		break;
	case ADAPTIVE_WEIGHT_8DIRECT:
		disparityMap = stereomatch_1::computeAdaptiveWeight_direct8(srcLeft, srcRight, DISPARITY_LEFT, 35, 0, 64);
		break;
	case ADAPTIVE_WEIGHT_GEODESIC:
		disparityMap = stereomatch_1::computeAdaptiveWeight_geodesic(srcLeft, srcRight, DISPARITY_LEFT, 15, 0, 64);
		break;
	case ADAPTIVE_WEIGHT_BILATERAL_GRID:
		disparityMap = stereomatch_1::computeAdaptiveWeight_bilateralGrid(srcLeft, srcRight, DISPARITY_LEFT, 10, 10, 20, 64);
		break;
	case ADAPTIVE_WEIGHT_BLO1:
		disparityMap = stereomatch_1::computeAdaptiveWeight_BLO1(srcLeft, srcRight, DISPARITY_LEFT, 0.015, 35, 0, 64);
		break;
	case ADAPTIVE_WEIGHT_GUIDED_FILTER:
		disparityMap = stereomatch_1::computeAdaptiveWeight_GuidedF(srcLeft, srcRight, DISPARITY_LEFT, 0.01, 15, 10, 150);
		break;
	case ADAPTIVE_WEIGHT_GUIDED_FILTER_2:
		disparityMap = stereomatch_1::computeAdaptiveWeight_GuidedF_2(srcLeft, srcRight, DISPARITY_LEFT, 0.01, 13, 10, 150);
		break;
	case ADAPTIVE_WEIGHT_GUIDED_FILTER_3:
		disparityMap = stereomatch_1::computeAdaptiveWeight_GuidedF_3(srcLeft, srcRight, DISPARITY_LEFT, 1e-6, 15, 0, 64);
		break;
	case ADAPTIVE_WEIGHT_MEDIAN:
		disparityMap = stereomatch_1::computeAdaptiveWeight_WeightedMedian(srcLeft, srcRight, DISPARITY_LEFT, 15, 10, 10, 0, 64);
	}
}

/**
 * \brief compute the disparity map from a pair of images captured by left and right cameras,which are rectified
 *			\notice:the left and right image should be the same type and size,
 *					and the type of the images should be 8-bit single-channel image.
 *			BM algorithm
 *			the higher disparity means the point in the world coordination is closer to the camera optical center
 * \param srcLeft :rectified left image(reference)
 * \param srcRight :rectified right image(target)
 * \param disparityMap :output disparity map
 */
void getDisparity_BM(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap)
{
	int ndisparities = 16 * 9;   /**< Range of disparity */
	int SADWindowSize = 35; /**< Size of the block window. Must be odd */
	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setMinDisparity(186);			//确定匹配搜索从哪里开始，默认为0
										//sbm->setNumDisparities(64);		//在该数值确定的视差范围内进行搜索，视差窗口
										//即，最大视差值与最小视差值之差，大小必须为16的整数倍
	sbm->setTextureThreshold(10);		//保证有足够的纹理以克服噪声
	sbm->setDisp12MaxDiff(-1);			//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异，默认为-1
	sbm->setPreFilterCap(31);			//
	sbm->setUniquenessRatio(25);		//使用匹配功能模式
	sbm->setSpeckleRange(32);			//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	sbm->setSpeckleWindowSize(0);		//检查视差连通区域变化度的窗口大小，值为0时取消speckle检查

	Mat imgDisparity16S = Mat(srcLeft.rows, srcLeft.cols, CV_16S);
	Mat imgDisparity8U = Mat(srcLeft.rows, srcLeft.cols, CV_8UC1);
	Mat Mask;

	if (srcLeft.empty() || srcRight.empty())
	{
		std::cout << " --(!) Error reading images " << std::endl;
		return;
	}

	if(srcLeft.type() != CV_8UC1)
	{
		cvtColor(srcLeft, srcLeft, COLOR_BGR2GRAY);
		srcLeft.convertTo(srcLeft, CV_8UC1);
	}

	if (srcRight.type() != CV_8UC1)
	{
		cvtColor(srcRight, srcRight, COLOR_BGR2GRAY);
		srcRight.convertTo(srcRight, CV_8UC1);
	}

	sbm->compute(srcLeft, srcRight, imgDisparity16S);

	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
	//cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
	//cvtGreyToBGR(imgDisparity8U, imgDisparity8U);
	//Mat disparityShow;
	//imgDisparity8U.copyTo(disparityShow, Mask);

	disparityMap = imgDisparity16S.clone();
}

/**
 * \brief compute the disparity map from a pair of images captured by left and right cameras,which are rectified
 *			\notice:the left and right image should be the same type and size,
 *					and the type of the images should be 8-bit single-channel image.
 *			SGBM algorithm
 *			the higher disparity means the point in the world coordination is closer to the camera optical center
 * \param srcLeft :rectified left image(reference)
 * \param srcRight :rectified right image(target)
 * \param disparityMap :output disparity map
 */
void getDisparity_SGBM(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap)
{
	int sgbmWinSize = 7;
	int cn = 3;
	int minDisparity = 0;
	int numDisparity = 48;
	//Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16*3, 7,
	//	8 * 3 * 7 * 7,
	//	32 * 3 * 7 * 7,
	//	1, 10, 5, 5, 2, StereoSGBM::MODE_SGBM_3WAY);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(minDisparity, numDisparity, sgbmWinSize);
	//sgbm->setPreFilterCap(10);
	//sgbm->setBlockSize(sgbmWinSize);
	//sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
	//sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
	//sgbm->setMinDisparity(minDisparity);
	//sgbm->setNumDisparities(numDisparity);
	//sgbm->setUniquenessRatio(10);
	//sgbm->setSpeckleWindowSize(175);
	//sgbm->setSpeckleRange(32);
	//sgbm->setDisp12MaxDiff(200);					//left-right consistency check
	//sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

	if (srcLeft.empty() || srcRight.empty())
	{
		std::cout << " --(!) Error reading images " << std::endl;
		return;
	}

	sgbm->compute(srcLeft, srcRight, disparityMap);

	disparityMap = disparityMap / 16;
	//sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
	//cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
	//cvtGreyToBGR(sgbmDisp8U, sgbmDisp8U);
	//Mat sgbmDisparityShow;
	//sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

}

/**
 * \brief :to convert the grey disparity map to BGR color image(pseudo color image)
 *			\notice:the pixel with lower value in the grey image tends to be bluer in the RGB image
 *					the pixel with higher value in the grey image tends to be reder in the RGB image
 * \param greySrc :input grey image
 * \param dstBGR :output BGR image
 */
void cvtGreyToBGR(cv::Mat greySrc, cv::Mat& dstBGR)
{
	if(greySrc.empty())
	{
		return;
	}

	if(greySrc.type() != CV_8UC1)
	{
		greySrc.convertTo(greySrc, CV_8UC1);
	}

	if (!dstBGR.empty())
	{
		if (dstBGR.type() != CV_8UC3)
		{
			dstBGR.convertTo(dstBGR, CV_8UC3);
		}
	}


	std::vector<Mat> colorChannelBGR;
	Mat colorChannelB, colorChannelG, colorChannelR;

	//channel R
	colorChannelR = greySrc.clone();

	//channel B
	Mat colorImgB_trans = -greySrc + 255;
	colorChannelB = colorImgB_trans.clone();

	//channel G
	Mat colorImgG_trans1_mask;
	threshold(greySrc, colorImgG_trans1_mask, 127, 1, THRESH_BINARY_INV);
	Mat colorImgG_trans1;
	greySrc.copyTo(colorImgG_trans1, colorImgG_trans1_mask);
	colorImgG_trans1 = colorImgG_trans1 * 2;

	Mat colorImgG_trans2_mask;
	bitwise_not(colorImgG_trans1_mask, colorImgG_trans2_mask);
	Mat colorImgG_trans2;
	greySrc.copyTo(colorImgG_trans2, colorImgG_trans2_mask);
	colorImgG_trans2 = 510 - colorImgG_trans2 * 2;
	threshold(colorImgG_trans2, colorImgG_trans2, 254, 1, THRESH_TOZERO_INV);
	colorChannelG = colorImgG_trans1 + colorImgG_trans2;

	//merge
	colorChannelBGR.push_back(colorChannelB);
	colorChannelBGR.push_back(colorChannelG);
	colorChannelBGR.push_back(colorChannelR);
	merge(colorChannelBGR, dstBGR);
}


void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(1.0, 0.5, 1.0);
	pcl::PointXYZ o;
	o.x = 1.0;
	o.y = 0;
	o.z = 0;
	viewer.addSphere(o, 0.25, "sphere", 0);
	std::cout << "i only run once" << std::endl;

}

void viewerPsycho(pcl::visualization::PCLVisualizer& viewer)
{
	static unsigned count = 0;
	std::stringstream ss;
	ss << "Once per viewer loop: " << count++;
	viewer.removeShape("text", 0);
	viewer.addText(ss.str(), 200, 300, "text", 0);

	//FIXME: possible race condition here:
	user_data++;
}

void showPointCloud(Mat originBGRMap, Mat disparityMap, std::string stereoParamPath)
{
	if(originBGRMap.empty())
	{
		return;
	}

	if(disparityMap.empty())
	{
		return;
	}

	disparityMap.convertTo(disparityMap, CV_32F);
	originBGRMap.convertTo(originBGRMap, CV_32F);

	Mat Q;
	if(!stereoParamPath.empty())
	{
		FileStorage fn(stereoParamPath, FileStorage::READ);
		fn["Rectify_Q"] >> Q;
		fn.release();
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	int rowNum = originBGRMap.rows;
	int colNum = originBGRMap.cols;

	cloud->height = rowNum;
	cloud->width = colNum;
	cloud->points.resize(cloud->width * cloud->height);
	for(unsigned int u = 0; u < rowNum; u++)
	{
		unsigned int num_rows = u * colNum;
		for(unsigned int v = 0; v < colNum; v++)
		{ 
			unsigned int num = num_rows + v;
			double Xw = 0, Yw = 0, Zw = 0;

			if ((double)disparityMap.at<float>(u, v) == 0)
			{
				Xw = 0;
				Yw = 0;
				Zw = 0;
			}
			else
			{
				Mat xyd = (Mat_<double>(4, 1) <<
					u, v, (double)disparityMap.at<float>(u, v), 1);
				Mat xyzw = Q * xyd;
				Xw = xyzw.at<double>(0, 0) / xyzw.at<double>(3, 0);
				Yw = xyzw.at<double>(1, 0) / xyzw.at<double>(3, 0);
				Zw = xyzw.at<double>(2, 0) / xyzw.at<double>(3, 0);
			}

			//if (Zw > 4000)
			//{
			//	continue;
			//}

			cloud->points[num].b = originBGRMap.at<Vec3f>(u, v)[0];
			cloud->points[num].g = originBGRMap.at<Vec3f>(u, v)[1];
			cloud->points[num].r = originBGRMap.at<Vec3f>(u, v)[2];

			cloud->points[num].x = Xw / 100;
			cloud->points[num].y = Yw / 100;
			cloud->points[num].z = Zw / 100;
		}
	}

	pcl::io::savePCDFileASCII("res.pcd", *cloud);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pointcloudFilter(cloud, cloud_filtered, VOXEL_GRID);
	pcl::io::savePCDFileASCII("res_filtered.pcd", *cloud);

	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloud);
	viewer.runOnVisualizationThreadOnce(viewerOneOff);

	viewer.runOnVisualizationThread(viewerPsycho);
	while (!viewer.wasStopped())
	{
		//you can also do cool processing here
		//FIXME: Note that this is running in a separate thread from viewerPsycho
		//and you should guard against race conditions yourself...
		user_data++;
	}
}


void showPointCloudVisual(cv::Mat originBGRMap, cv::Mat disparityMap, std::string cameraPairParaPath)
{
	if (originBGRMap.empty())
	{
		return;
	}

	if (disparityMap.empty())
	{
		return;
	}
	if (cameraPairParaPath.empty())
	{
		return;
	}

	disparityMap.convertTo(disparityMap, CV_32F);
	originBGRMap.convertTo(originBGRMap, CV_32F);

	// read system parameters
	Mat Q, T;
	FileStorage fn(cameraPairParaPath, FileStorage::READ);
	fn["Rectify_Q"] >> Q;
	fn["StereoCalib_T"] >> T;
	fn.release();
	double focalLen = Q.at<double>(2, 3);
	double baseline = T.at<double>(0, 0);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	int rowNum = disparityMap.rows;
	int colNum = disparityMap.cols;

	cloud->height = rowNum;
	cloud->width = colNum;
	cloud->points.resize(cloud->width * cloud->height);

	//Mat rangeImage(originBGRMap.size(), CV_32FC1);

	////pcl::visualization::RangeImageVisualizer range_image_widget("Range Image");
	////range_image_widget.showRangeImage(*pclRangeImg);

	////pcl::PolygonMesh triangles;
	////pclMesh_OrganizedFastMesh(pclRangeImg, triangles);

	////boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_mesh(new pcl::visualization::PCLVisualizer("PCL Mesh"));
	////viewer_mesh->setBackgroundColor(0.5, 0.5, 0.5);
	////viewer_mesh->addPolygonMesh(triangles, "tin");
	////viewer_mesh->addCoordinateSystem();
	////while (!range_image_widget.wasStopped() && !viewer_mesh->wasStopped())
	////{
	////	range_image_widget.spinOnce();

	////	boost::this_thread::sleep(boost::posix_time::microseconds(100));
	////	viewer_mesh->spinOnce();

	////}
	for (unsigned int u = 0; u < rowNum; u++)
	{
		unsigned int num_rows = u * colNum;
		for (unsigned int v = 0; v < colNum; v++)
		{
			unsigned int num = num_rows + v;
			double Xw = 0, Yw = 0, Zw = 0;

			if((double)disparityMap.at<float>(u, v) == 0)
			{
				Xw = 0;
				Yw = 0;
				Zw = 0;
			}
			else
			{
				Mat xyd = (Mat_<double>(4, 1) <<
					u, v, (double)disparityMap.at<float>(u, v), 1);
				Mat xyzw = Q * xyd;
				Xw = xyzw.at<double>(0, 0) / xyzw.at<double>(3, 0);
				Yw = xyzw.at<double>(1, 0) / xyzw.at<double>(3, 0);
				Zw = xyzw.at<double>(2, 0) / xyzw.at<double>(3, 0);
			}

			if(Zw > 20000)
			{
				continue;
			}

			cloud->points[num].b = originBGRMap.at<Vec3f>(u, v)[0];
			cloud->points[num].g = originBGRMap.at<Vec3f>(u, v)[1];
			cloud->points[num].r = originBGRMap.at<Vec3f>(u, v)[2];

			cloud->points[num].x = Xw;
			cloud->points[num].y = Yw;
			cloud->points[num].z = Zw;

			originBGRMap.at<float>(u, v) = Zw;
		}
	}

	pcl::io::savePCDFileASCII("res.pcd", *cloud);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter(
		new pcl::PointCloud<pcl::PointXYZRGB>);
	pointcloudFilter(cloud, cloud_filter, VOXEL_GRID);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter_proj(
		new pcl::PointCloud<pcl::PointXYZRGB>);
	pointcloudFilter(cloud_filter, cloud_filter_proj, 
		MODEL_COEFFICIENTS);
	pcl::io::savePCDFileASCII("res_filter.pcd", *cloud_filter_proj);

	pcl::RangeImagePlanar pclRangeImg;
	pclRangeImg.setDisparityImage((float*)disparityMap.data,   colNum, rowNum, focalLen, baseline);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPt_NARF(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::RangeImage::Ptr rangeImg_(&pclRangeImg);
	pcl::PointCloud<int>::Ptr keypoint_indices(new pcl::PointCloud<int>);

	pclFeaturePt_NARF(*rangeImg_, keyPt_NARF, keypoint_indices);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Cloud Viewer"));
	//viewer->setBackgroundColor(255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void showPointCloudVisual_my2(cv::Mat originBGRMap, cv::Mat disparityMap, std::string cameraPairParaPath, std::string resPCLPath, bool isLeft)
{
	if (originBGRMap.empty())
	{
		return;
	}

	if (disparityMap.empty())
	{
		return;
	}
	if (cameraPairParaPath.empty())
	{
		return;
	}

	disparityMap.convertTo(disparityMap, CV_32F);
	originBGRMap.convertTo(originBGRMap, CV_32F);

	// read system parameters
	cv::Size imgSize;
	cv::Mat K_L, K_R, D_L, D_R;
	cv::Mat R_l_r, T_l_r;
	FileStorage fn(cameraPairParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["CameraInnerParaL"] >> K_L;
	fn["CameraDistParaL"] >> D_L;
	fn["CameraInnerParaR"] >> K_R;
	fn["CameraDistParaR"] >> D_R;
	fn["RotationL2R"] >> R_l_r;
	fn["TransationL2R"] >> T_l_r;
	fn.release();

	cv::Mat K_new;

	cv::Vec3d tt = cv::Vec3d(T_l_r.at<double>(0,0), T_l_r.at<double>(1,0), T_l_r.at<double>(2, 0));
	tt = -tt;
	double tt_theta_xz = -atan(tt[1] / tt[0]);
	cv::Matx33d Rz = cv::Matx33d(cos(tt_theta_xz), -sin(tt_theta_xz), 0,
		sin(tt_theta_xz), cos(tt_theta_xz), 0,
		0, 0, 1);
	double tt_theta_xy = -atan(tt[2] / sqrt(tt[0] * tt[0] + tt[1] * tt[1]));
	cv::Matx33d Ry = cv::Matx33d(cos(tt_theta_xy), 0, sin(tt_theta_xy),
		0, 1, 0,
		-sin(tt_theta_xy), 0, cos(tt_theta_xy));
	cv::Matx33d tt_rr = Ry * Rz;
	cv::Matx33d tt_rrr = tt_rr.inv(cv::DECOMP_SVD);

	if (isLeft)
	{
		K_new = K_L * R_l_r.inv(cv::DECOMP_SVD) * tt_rrr;
		K_new = K_new / 2;//for 2560*1440
		//K_new = K_new / 8;//for rectified images' size is 2560*1440 _*2

	}
	else
	{
		K_new = K_R * tt_rrr;
		K_new = K_new / 2;
		//K_new = K_new / 8;//for 2560*1440 _*2

	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	int rowNum = disparityMap.rows;
	int colNum = disparityMap.cols;

	cloud->height = rowNum;
	cloud->width = colNum;
	cloud->points.resize(cloud->width * cloud->height);

	double f0 = K_new.at<double>(0, 0);
	double f1 = K_new.at<double>(1, 1);
	double baseline = sqrt(T_l_r.dot(T_l_r));
	double u0 = K_new.at<double>(0, 2);
	double v0 = K_new.at<double>(1, 2);

	for (unsigned int v = 0; v < rowNum; v++)
	{
		unsigned int num_rows = v * colNum;
		double temp1 = (v - v0) * (v - v0) + f1 * f1;
		double temp2 = sqrt(temp1);
		double coeff = f0 * f1 / temp2;

		for (unsigned int u = 0; u < colNum; u++)
		{
			cv::Vec2d px = cv::Vec2d((u - u0) / f0, (v - v0) / f1);
			unsigned int num = num_rows + u;
			double Xw = 0, Yw = 0, Zw = 0;

			if ((double)disparityMap.at<float>(v, u) == 0)
			{
				Xw = 0;
				Yw = 0;
				Zw = 0;
			}
			else
			{
				double cur_disp = fabs(disparityMap.at<float>(v, u));
				Zw = baseline * coeff / cur_disp;
				Yw = px[1] * Zw;
				Xw = px[0] * sqrt(Zw * Zw + Yw * Yw);
			}

			if (Zw > 20000)
			{
				continue;
			}

			if(originBGRMap.at<Vec3f>(v, u)[0] == 0 && originBGRMap.at<Vec3f>(v, u)[1] == 0 && originBGRMap.at<Vec3f>(v, u)[2] == 0)
			{
				continue;
			}
			cloud->points[num].b = originBGRMap.at<Vec3f>(v, u)[0];
			cloud->points[num].g = originBGRMap.at<Vec3f>(v, u)[1];
			cloud->points[num].r = originBGRMap.at<Vec3f>(v, u)[2];


			cloud->points[num].x = Xw;
			cloud->points[num].y = Yw;
			cloud->points[num].z = Zw;

		}
	}

	pcl::io::savePCDFileASCII(resPCLPath + "res.pcd", *cloud);

	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud, cloud_filter, VOXEL_GRID);
	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter_proj(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud_filter, cloud_filter_proj,
	////	MODEL_COEFFICIENTS);
	////pcl::io::savePCDFileASCII("res_filter.pcd", *cloud_filter_proj);

	//////pcl::RangeImagePlanar pclRangeImg;
	//////pclRangeImg.setDisparityImage((float*)disparityMap.data, colNum, rowNum, f0, baseline);
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr keyPt_NARF(new pcl::PointCloud<pcl::PointXYZ>);
	//////pcl::RangeImage::Ptr rangeImg_(&pclRangeImg);
	//////pcl::PointCloud<int>::Ptr keypoint_indices(new pcl::PointCloud<int>);

	//////pclFeaturePt_NARF(*rangeImg_, keyPt_NARF, keypoint_indices);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Cloud Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void showPointCloudVisual_my22(cv::Mat originBGRMap, cv::Mat disparityMap, std::string cameraPairParaPath,
	std::string resPCLPath, bool isLeft)
{
	if (originBGRMap.empty())
	{
		return;
	}

	if (disparityMap.empty())
	{
		return;
	}
	if (cameraPairParaPath.empty())
	{
		return;
	}

	disparityMap.convertTo(disparityMap, CV_32F);
	originBGRMap.convertTo(originBGRMap, CV_32F);

	// read system parameters
	cv::Size imgSize;
	cv::Mat K_L, K_R, D_L, D_R;
	cv::Mat R_l_r, T_l_r;
	FileStorage fn(cameraPairParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["CameraInnerParaL"] >> K_L;
	fn["CameraDistParaL"] >> D_L;
	fn["CameraInnerParaR"] >> K_R;
	fn["CameraDistParaR"] >> D_R;
	fn["RotationL2R"] >> R_l_r;
	fn["TransationL2R"] >> T_l_r;
	fn.release();

	cv::Mat K_new;

	cv::Vec3d tt = cv::Vec3d(T_l_r.at<double>(0, 0), T_l_r.at<double>(1, 0), T_l_r.at<double>(2, 0));
	tt = -tt;
	double tt_theta_xz = -atan(tt[1] / tt[0]);
	cv::Matx33d Rz = cv::Matx33d(cos(tt_theta_xz), -sin(tt_theta_xz), 0,
		sin(tt_theta_xz), cos(tt_theta_xz), 0,
		0, 0, 1);
	double tt_theta_xy = -atan(tt[2] / sqrt(tt[0] * tt[0] + tt[1] * tt[1]));
	cv::Matx33d Ry = cv::Matx33d(cos(tt_theta_xy), 0, sin(tt_theta_xy),
		0, 1, 0,
		-sin(tt_theta_xy), 0, cos(tt_theta_xy));
	cv::Matx33d tt_rr = Ry * Rz;
	cv::Matx33d tt_rrr = tt_rr.inv(cv::DECOMP_SVD);

	if (isLeft)
	{
		K_new = K_L * R_l_r.inv(cv::DECOMP_SVD) * tt_rrr;
		K_new = K_new / 4;
	}
	else
	{
		K_new = K_R * tt_rrr;
		K_new = K_new / 4;
	}

	int x_expand_half = 640 * 1 / 2;
	int y_expand_half = 360 * 1 / 2;	
	K_new.at<double>(0, 2) = K_new.at<double>(0, 2) + x_expand_half;
	K_new.at<double>(1, 2) = K_new.at<double>(1, 2) + y_expand_half;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	int rowNum = disparityMap.rows;
	int colNum = disparityMap.cols;

	cloud->height = rowNum;
	cloud->width = colNum;
	cloud->points.resize(cloud->width * cloud->height);

	double f0 = K_new.at<double>(0, 0);
	double f1 = K_new.at<double>(1, 1);
	double baseline = sqrt(T_l_r.dot(T_l_r));
	double u0 = K_new.at<double>(0, 2);
	double v0 = K_new.at<double>(1, 2);

	for (unsigned int v = 0; v < rowNum; v++)
	{
		unsigned int num_rows = v * colNum;
		double temp1 = (v - v0) * (v - v0) + f1 * f1;
		double temp2 = sqrt(temp1);
		double coeff = f0 * f1 / temp2;

		for (unsigned int u = 0; u < colNum; u++)
		{
			cv::Vec2d px = cv::Vec2d((u - u0) / f0, (v - v0) / f1);
			unsigned int num = num_rows + u;
			double Xw = 0, Yw = 0, Zw = 0;

			if ((double)disparityMap.at<float>(v, u) == 0)
			{
				Xw = 0;
				Yw = 0;
				Zw = 0;
			}
			else
			{
				double cur_disp = fabs(disparityMap.at<float>(v, u));
				Zw = baseline * coeff / cur_disp;
				Yw = px[1] * Zw;
				Xw = px[0] * sqrt(Zw * Zw + Yw * Yw);
			}

			if (Zw > 20000)
			{
				continue;
			}
			if (originBGRMap.at<Vec3f>(v, u)[0] == 0 && originBGRMap.at<Vec3f>(v, u)[1] == 0 && originBGRMap.at<Vec3f>(v, u)[2] == 0)
			{
				continue;
			}

			cloud->points[num].b = originBGRMap.at<Vec3f>(v, u)[0];
			cloud->points[num].g = originBGRMap.at<Vec3f>(v, u)[1];
			cloud->points[num].r = originBGRMap.at<Vec3f>(v, u)[2];

			cloud->points[num].x = Xw;
			cloud->points[num].y = Yw;
			cloud->points[num].z = Zw;

		}
	}

	pcl::io::savePCDFileASCII(resPCLPath + "res.pcd", *cloud);

	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud, cloud_filter, VOXEL_GRID);
	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter_proj(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud_filter, cloud_filter_proj,
	////	MODEL_COEFFICIENTS);
	////pcl::io::savePCDFileASCII("res_filter.pcd", *cloud_filter_proj);

	//////pcl::RangeImagePlanar pclRangeImg;
	//////pclRangeImg.setDisparityImage((float*)disparityMap.data, colNum, rowNum, f0, baseline);
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr keyPt_NARF(new pcl::PointCloud<pcl::PointXYZ>);
	//////pcl::RangeImage::Ptr rangeImg_(&pclRangeImg);
	//////pcl::PointCloud<int>::Ptr keypoint_indices(new pcl::PointCloud<int>);

	//////pclFeaturePt_NARF(*rangeImg_, keyPt_NARF, keypoint_indices);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Cloud Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void showPointCloudVisual_my3(cv::Mat originBGRMap, cv::Mat disparityMap, std::string cameraPairParaPath,
	std::string resPCLPath, bool isLeft)
{
	if (originBGRMap.empty())
	{
		return;
	}

	if (disparityMap.empty())
	{
		return;
	}
	if (cameraPairParaPath.empty())
	{
		return;
	}

	disparityMap.convertTo(disparityMap, CV_32F);
	originBGRMap.convertTo(originBGRMap, CV_32F);

	// read system parameters
	cv::Size imgSize;
	cv::Mat K_L, K_R, D_L, D_R;
	cv::Mat R_l_r, T_l_r;
	FileStorage fn(cameraPairParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["CameraInnerParaL"] >> K_L;
	fn["CameraDistParaL"] >> D_L;
	fn["CameraInnerParaR"] >> K_R;
	fn["CameraDistParaR"] >> D_R;
	fn["RotationL2R"] >> R_l_r;
	fn["TransationL2R"] >> T_l_r;
	fn.release();

	cv::Mat K_new;

	cv::Vec3d tt = cv::Vec3d(T_l_r.at<double>(0, 0), T_l_r.at<double>(1, 0), T_l_r.at<double>(2, 0));
	tt = -tt;
	double tt_theta_xz = -atan(tt[1] / tt[0]);
	cv::Matx33d Rz = cv::Matx33d(cos(tt_theta_xz), -sin(tt_theta_xz), 0,
		sin(tt_theta_xz), cos(tt_theta_xz), 0,
		0, 0, 1);
	double tt_theta_xy = -atan(tt[2] / sqrt(tt[0] * tt[0] + tt[1] * tt[1]));
	cv::Matx33d Ry = cv::Matx33d(cos(tt_theta_xy), 0, sin(tt_theta_xy),
		0, 1, 0,
		-sin(tt_theta_xy), 0, cos(tt_theta_xy));
	cv::Matx33d tt_rr = Ry * Rz;
	cv::Matx33d tt_rrr = tt_rr.inv(cv::DECOMP_SVD);

	if (isLeft)
	{
		K_new = K_L * R_l_r.inv(cv::DECOMP_SVD) * tt_rrr;
		K_new = K_new / 2;
	}
	else
	{
		K_new = K_R * tt_rrr;
		K_new = K_new / 2;
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	int rowNum;
	int colNum;

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
		//ROIL = cv::Rect(cv::Point(190, 69), cv::Point(598, 359));//for 2560_1440/rectifyR/3_pattern0
		//ROIR = cv::Rect(cv::Point(28, 76), cv::Point(427, 359));
	}
	{
		ROIL = cv::Rect(cv::Point(268, 100), cv::Point(1279, 719));//for 2560_1440/rectifyL/14_pattern0
		ROIR = cv::Rect(cv::Point(0, 113), cv::Point(922, 719));
	}

	if(isLeft)
	{
		rowNum = 719 - 100 + 1;
		colNum = 1279 - 268 + 1;
	}
	else
	{
		rowNum = 719 - 113 + 1;
		colNum = 922 - 0 + 1;
	}

	cloud->height = rowNum;
	cloud->width = colNum;
	cloud->points.resize(cloud->width * cloud->height);

	double f0 = K_new.at<double>(0, 0);
	double f1 = K_new.at<double>(1, 1);
	double baseline = sqrt(T_l_r.dot(T_l_r));
	double u0 = K_new.at<double>(0, 2);
	double v0 = K_new.at<double>(1, 2);

	for (unsigned int v = 0; v < originBGRMap.rows; v++)
	{
		unsigned int num_rows = v * originBGRMap.cols;
		double temp1 = (v - v0) * (v - v0) + f1 * f1;
		double temp2 = sqrt(temp1);
		double coeff = f0 * f1 / temp2;

		for (unsigned int u = 0; u < originBGRMap.cols; u++)
		{
			cv::Vec2d px = cv::Vec2d((u - u0) / f0, (v - v0) / f1);
			unsigned int num = num_rows + u;
			double Xw = 0, Yw = 0, Zw = 0;

			if ((double)disparityMap.at<float>(v, u) == 0)
			{
				Xw = 0;
				Yw = 0;
				Zw = 0;
			}
			else
			{
				double cur_disp = fabs(disparityMap.at<float>(v, u));
				Zw = baseline * coeff / cur_disp;
				Yw = px[1] * Zw;
				Xw = px[0] * sqrt(Zw * Zw + Yw * Yw);
			}

			if (Zw > 20000)
			{
				continue;
			}
			if (originBGRMap.at<Vec3f>(v, u)[0] == 0 && originBGRMap.at<Vec3f>(v, u)[1] == 0 && originBGRMap.at<Vec3f>(v, u)[2] == 0)
			{
				continue;
			}

			if ((isLeft && (u >= ROIL.x && u < (ROIL.x + ROIL.width) && v >= ROIL.y && v < (ROIL.y + ROIL.height)))
				|| (!isLeft && (u >= ROIR.x && u < (ROIR.x + ROIR.width) && v >= ROIR.y && v < (ROIR.y + ROIR.height))))
			{
				if(isLeft)
				{
					num = (v - ROIL.y) * ROIL.width + (u - ROIL.x);
				}
				else
				{
					num = (v - ROIR.y) * ROIR.width + (u - ROIR.x);
				}
				cloud->points[num].b = originBGRMap.at<Vec3f>(v, u)[0];
				cloud->points[num].g = originBGRMap.at<Vec3f>(v, u)[1];
				cloud->points[num].r = originBGRMap.at<Vec3f>(v, u)[2];

				cloud->points[num].x = Xw;
				cloud->points[num].y = Yw;
				cloud->points[num].z = Zw;
			}

		}
	}

	pcl::io::savePCDFileASCII(resPCLPath + "res.pcd", *cloud);

	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud, cloud_filter, VOXEL_GRID);
	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter_proj(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud_filter, cloud_filter_proj,
	////	MODEL_COEFFICIENTS);
	////pcl::io::savePCDFileASCII("res_filter.pcd", *cloud_filter_proj);

	//////pcl::RangeImagePlanar pclRangeImg;
	//////pclRangeImg.setDisparityImage((float*)disparityMap.data, colNum, rowNum, f0, baseline);
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr keyPt_NARF(new pcl::PointCloud<pcl::PointXYZ>);
	//////pcl::RangeImage::Ptr rangeImg_(&pclRangeImg);
	//////pcl::PointCloud<int>::Ptr keypoint_indices(new pcl::PointCloud<int>);

	//////pclFeaturePt_NARF(*rangeImg_, keyPt_NARF, keypoint_indices);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Cloud Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void showPointCloudVisual_my33(cv::Mat originBGRMap, cv::Mat disparityMap, std::string cameraPairParaPath,
	std::string resPCLPath, bool isLeft)
{
	if (originBGRMap.empty())
	{
		return;
	}

	if (disparityMap.empty())
	{
		return;
	}
	if (cameraPairParaPath.empty())
	{
		return;
	}

	disparityMap.convertTo(disparityMap, CV_32F);
	originBGRMap.convertTo(originBGRMap, CV_32F);

	// read system parameters
	cv::Size imgSize;
	cv::Mat K_L, K_R, D_L, D_R;
	cv::Mat R_l_r, T_l_r;
	FileStorage fn(cameraPairParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["CameraInnerParaL"] >> K_L;
	fn["CameraDistParaL"] >> D_L;
	fn["CameraInnerParaR"] >> K_R;
	fn["CameraDistParaR"] >> D_R;
	fn["RotationL2R"] >> R_l_r;
	fn["TransationL2R"] >> T_l_r;
	fn.release();

	cv::Mat K_new;

	cv::Vec3d tt = cv::Vec3d(T_l_r.at<double>(0, 0), T_l_r.at<double>(1, 0), T_l_r.at<double>(2, 0));
	tt = -tt;
	double tt_theta_xz = -atan(tt[1] / tt[0]);
	cv::Matx33d Rz = cv::Matx33d(cos(tt_theta_xz), -sin(tt_theta_xz), 0,
		sin(tt_theta_xz), cos(tt_theta_xz), 0,
		0, 0, 1);
	double tt_theta_xy = -atan(tt[2] / sqrt(tt[0] * tt[0] + tt[1] * tt[1]));
	cv::Matx33d Ry = cv::Matx33d(cos(tt_theta_xy), 0, sin(tt_theta_xy),
		0, 1, 0,
		-sin(tt_theta_xy), 0, cos(tt_theta_xy));
	cv::Matx33d tt_rr = Ry * Rz;
	cv::Matx33d tt_rrr = tt_rr.inv(cv::DECOMP_SVD);

	if (isLeft)
	{
		K_new = K_L * R_l_r.inv(cv::DECOMP_SVD) * tt_rrr;
		K_new = K_new / 4;
	}
	else
	{
		K_new = K_R * tt_rrr;
		K_new = K_new / 4;
	}
	int x_expand_half = 640 * 1 / 2;
	int y_expand_half = 360 * 1 / 2;
	K_new.at<double>(0, 2) = K_new.at<double>(0, 2) + x_expand_half;
	K_new.at<double>(1, 2) = K_new.at<double>(1, 2) + y_expand_half;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	int rowNum;
	int colNum;

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
		//ROIL = cv::Rect(cv::Point(454, 229), cv::Point(1025, 660));//for 2560_2_1440_2/rectifyL/14_pattern0
		//ROIR = cv::Rect(cv::Point(201, 237), cv::Point(785, 600));
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

	if (isLeft)
	{
		rowNum = 660 - 229 + 1;
		colNum = 1025 - 454 + 1;
	}
	else
	{
		rowNum = 600 - 237 + 1;
		colNum = 785 - 201 + 1;
	}

	cloud->height = rowNum;
	cloud->width = colNum;
	cloud->points.resize(cloud->width * cloud->height);

	double f0 = K_new.at<double>(0, 0);
	double f1 = K_new.at<double>(1, 1);
	double baseline = sqrt(T_l_r.dot(T_l_r));
	double u0 = K_new.at<double>(0, 2);
	double v0 = K_new.at<double>(1, 2);

	for (unsigned int v = 0; v < originBGRMap.rows; v++)
	{
		unsigned int num_rows = v * originBGRMap.cols;
		double temp1 = (v - v0) * (v - v0) + f1 * f1;
		double temp2 = sqrt(temp1);
		double coeff = f0 * f1 / temp2;

		for (unsigned int u = 0; u < originBGRMap.cols; u++)
		{
			cv::Vec2d px = cv::Vec2d((u - u0) / f0, (v - v0) / f1);
			unsigned int num = num_rows + u;
			double Xw = 0, Yw = 0, Zw = 0;

			if ((double)disparityMap.at<float>(v, u) == 0)
			{
				Xw = 0;
				Yw = 0;
				Zw = 0;
			}
			else
			{
				double cur_disp = fabs(disparityMap.at<float>(v, u));
				Zw = baseline * coeff / cur_disp;
				Yw = px[1] * Zw;
				Xw = px[0] * sqrt(Zw * Zw + Yw * Yw);
			}

			if (Zw > 20000)
			{
				continue;
			}
			if (originBGRMap.at<Vec3f>(v, u)[0] == 0 && originBGRMap.at<Vec3f>(v, u)[1] == 0 && originBGRMap.at<Vec3f>(v, u)[2] == 0)
			{
				continue;
			}

			if ((isLeft && (u >= ROIL.x && u < (ROIL.x + ROIL.width) && v >= ROIL.y && v < (ROIL.y + ROIL.height)))
				|| (!isLeft && (u >= ROIR.x && u < (ROIR.x + ROIR.width) && v >= ROIR.y && v < (ROIR.y + ROIR.height))))
			{
				if (isLeft)
				{
					num = (v - ROIL.y) * ROIL.width + (u - ROIL.x);
				}
				else
				{
					num = (v - ROIR.y) * ROIR.width + (u - ROIR.x);
				}
				cloud->points[num].b = originBGRMap.at<Vec3f>(v, u)[0];
				cloud->points[num].g = originBGRMap.at<Vec3f>(v, u)[1];
				cloud->points[num].r = originBGRMap.at<Vec3f>(v, u)[2];

				cloud->points[num].x = Xw;
				cloud->points[num].y = Yw;
				cloud->points[num].z = Zw;
			}

		}
	}

	pcl::io::savePCDFileASCII(resPCLPath + "res.pcd", *cloud);

	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud, cloud_filter, VOXEL_GRID);
	////pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter_proj(
	////	new pcl::PointCloud<pcl::PointXYZRGB>);
	////pointcloudFilter(cloud_filter, cloud_filter_proj,
	////	MODEL_COEFFICIENTS);
	////pcl::io::savePCDFileASCII("res_filter.pcd", *cloud_filter_proj);

	//////pcl::RangeImagePlanar pclRangeImg;
	//////pclRangeImg.setDisparityImage((float*)disparityMap.data, colNum, rowNum, f0, baseline);
	//////pcl::PointCloud<pcl::PointXYZ>::Ptr keyPt_NARF(new pcl::PointCloud<pcl::PointXYZ>);
	//////pcl::RangeImage::Ptr rangeImg_(&pclRangeImg);
	//////pcl::PointCloud<int>::Ptr keypoint_indices(new pcl::PointCloud<int>);

	//////pclFeaturePt_NARF(*rangeImg_, keyPt_NARF, keypoint_indices);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Cloud Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

}

void pclFilter_my2(std::string pclPath, std::string dstPCL_path)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>(pclPath, *cloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>);

	//smooth pointcloud and get cloud with normal info
  // Create a KD-Tree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	//output cloud of mls method
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZ>);

	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;

	mls.setComputeNormals(false);

	mls.setInputCloud(cloud);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(1);

	// Smooth

	cout << "Moving Least Squares Smoothing..." << endl;

	mls.process(*cloud_smoothed);

	cout << "Smoothing finished!" << endl;

	//end of mls method
	//cloud_smoothed is the ouput smoothed cloud without normal info

	// Normal estimation
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(cloud_smoothed);
	n.setInputCloud(cloud_smoothed);
	n.setSearchMethod(tree2);
	n.setKSearch(15);

	cout << "Normal Estimation..." << endl;

	n.compute(*normals);

	cout << "Normal Estimation finished!!" << endl;

	// Concatenate the XYZ and normal fields
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud_smoothed, *normals, *cloud_with_normals);
	// cloud_with_normals = cloud_smoothed + normals



	// Create search tree
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree3(new pcl::search::KdTree<pcl::PointNormal>);
	tree3->setInputCloud(cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius(15);

	// Set typical values for the parameters
	gp3.setMu(5);
	gp3.setMaximumNearestNeighbors(400);
	gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
	gp3.setMinimumAngle(M_PI / 18); // 10 degrees
	gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
	gp3.setNormalConsistency(false);
	gp3.setConsistentVertexOrdering(true);

	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree3);

	cout << "Fast triangular Meshing..." << endl;

	gp3.reconstruct(triangles);

	cout << "Meshing finishd" << endl;

	// Additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	//Visualize Mesh result
	pcl::visualization::PCLVisualizer viewer("Mesh Visualizer");
	viewer.addPointCloud(cloud_smoothed, "Cloud");
	viewer.addPolygonMesh(triangles, "Triangular Mesh");

	std::string outname = "mesh_out.vtk";
	pcl::io::saveVTKFile(outname, triangles);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

}

void pclPoint_xyzrgb2xyz(
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud_origin,
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud_dst)
{
	int cloudSize = cloud_origin->points.size();
	for(int i = 0; i < cloudSize; i++)
	{
		pcl::PointXYZ pt;
		pt.x = cloud_origin->points[i].x;
		pt.y = cloud_origin->points[i].y;
		pt.z = cloud_origin->points[i].z;
		cloud_dst->points.push_back(pt);
	}

	cloud_dst->width = cloud_origin->width;
	cloud_dst->height = cloud_origin->height;
}

void pclMesh_OrganizedFastMesh(
	boost::shared_ptr<pcl::RangeImagePlanar> range_image_origin,
	pcl::PolygonMesh& triangle_mesh_dst)
{
	int type = 0;

	pcl::OrganizedFastMesh<pcl::PointWithRange>::Ptr tri(
		new pcl::OrganizedFastMesh<pcl::PointWithRange>);
	pcl::search::KdTree<pcl::PointWithRange>::Ptr tree(
		new pcl::search::KdTree<pcl::PointWithRange>);
	tree->setInputCloud(range_image_origin);

	tri->setTrianglePixelSize(2);
	tri->setInputCloud(range_image_origin);
	tri->setSearchMethod(tree);
	tri->setTriangulationType((pcl::OrganizedFastMesh<pcl::PointWithRange>::TriangulationType)type);
	tri->reconstruct(triangle_mesh_dst);


}

/**
 * \brief 计算点云的空间分辨率，算出点云的每个点与其临近点距离的平均值
 * \param cloud 
 * \return 
 */
double computeCloudResolution(
	const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<pcl::PointXYZRGBA> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

void pcl_CorrespGroup(
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> > cloud_model,
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> > cloud_scene,
	std::vector<Eigen::Matrix4f, 
				Eigen::aligned_allocator<Eigen::Matrix4f>>& rototranslations)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model_keypoints(
		new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_keypoints(
		new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::Normal>::Ptr model_normals(
		new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals(
		new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors(
		new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors(
		new pcl::PointCloud<pcl::SHOT352>());

	bool use_cloud_resolution_(false);
	bool use_hough_(true);
	float model_ss_(0.01f);
	float scene_ss_(0.03f);
	float rf_rad_(0.015f);
	float descr_rad_(0.02f);
	float cg_size_(0.01f);
	float cg_thresh_(5.0f);

	//
//  Set up resolution invariance
//
	if (use_cloud_resolution_)
	{
		float resolution = static_cast<float>(computeCloudResolution(cloud_model));
		if (resolution != 0.0f)
		{
			model_ss_ *= resolution;
			scene_ss_ *= resolution;
			rf_rad_ *= resolution;
			descr_rad_ *= resolution;
			cg_size_ *= resolution;
		}

		std::cout << "Model resolution:       " << resolution << std::endl;
		std::cout << "Model sampling size:    " << model_ss_ << std::endl;
		std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
		std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
		std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
		std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
	}

	//计算法向量
	pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> norm_est;
	//设置法向量计算时的临近点数，经验值；
	//10,较适合Kinect等获取的数据，处理其他稠密点云，需要自行调试该值
	norm_est.setKSearch(10);						
	norm_est.setInputCloud(cloud_model);
	norm_est.compute(*model_normals);

	norm_est.setInputCloud(cloud_scene);
	norm_est.compute(*scene_normals);

	//对点云进行下采样，获取稀疏的关键点；这些关键点与后续的3D描述子相关联
	pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
	uniform_sampling.setInputCloud(cloud_model);
	uniform_sampling.setRadiusSearch(model_ss_);
	uniform_sampling.filter(*model_keypoints);
	std::cout << "Model total points: " << cloud_model->size() 
		<< "; Selected Keypoints: " << model_keypoints->size() << std::endl;

	uniform_sampling.setInputCloud(cloud_scene);
	uniform_sampling.setRadiusSearch(scene_ss_);
	uniform_sampling.filter(*scene_keypoints);
	std::cout << "Scene total points: " << cloud_scene->size() 
		<< "; Selected Keypoints: " << scene_keypoints->size() << std::endl;

	//为模型和场景的每个关键点建立特征描述子，计算3D描述子
	//计算SHOT描述子
	pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> descr_est;
	descr_est.setRadiusSearch(descr_rad_);			//设置SHOT局部描述子的描述区域范围的大小
	descr_est.setInputCloud(model_keypoints);
	descr_est.setInputNormals(model_normals);
	descr_est.setSearchSurface(cloud_model);
	descr_est.compute(*model_descriptors);

	descr_est.setInputCloud(scene_keypoints);
	descr_est.setInputNormals(scene_normals);
	descr_est.setSearchSurface(cloud_scene);
	descr_est.compute(*scene_descriptors);

	//确定对应点对集合：匹配
	//构造模型描述子点云的KdTreeFLANN，
	//在欧式空间中，对于和场景描述子点云中每个点进行有效最近邻搜索
	//然后，添加场景描述子点云中的最近邻点到搜索点的对应点向量中：例，当两描述子间的平方距离小于某阈值时
	//
	//利用KD树结构找到模型与场景的对应点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<pcl::SHOT352> match_search;
	match_search.setInputCloud(model_descriptors);
	//对于每个场景的特征点描述子，寻找模型特征点描述子的最近邻点，
	//并将其加入对应点向量
	for (size_t i = 0; i < scene_descriptors->size(); ++i)
	{
		std::vector<int> neigh_indices(1);
		std::vector<float> neigh_sqr_dists(1);
		if (!pcl_isfinite(scene_descriptors->at(i).descriptor[0])) //忽略NaNs值
		{
			continue;
		}
		int found_neighs = match_search.nearestKSearch(
			scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
		//add match only if the squared descriptor distance is less than 0.25 
		//(SHOT descriptor distances are between 0 and 1 by design)
		if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) 
		{
			pcl::Correspondence corr(neigh_indices[0], 
				static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back(corr);
		}
	}
	std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

	//聚类
	std::vector<pcl::Correspondences> clustered_corrs;
	//使用Hough 3D Grouping算法：基于Hough投票过程：
	//该算法需要将关键点局部参考坐标系（LRF）作为参数传递，其与每个关键点相关联
	if (use_hough_)
	{
		//计算关键点的局部参考坐标系
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(
			new pcl::PointCloud<pcl::ReferenceFrame>());
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf(
			new pcl::PointCloud<pcl::ReferenceFrame>());

		//利用BOARDLocalReferenceFrameEstimation类计算LRF
		pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGBA, 
						pcl::Normal, pcl::ReferenceFrame> rf_est;
		rf_est.setFindHoles(true);
		rf_est.setRadiusSearch(rf_rad_);			//设置估计局部参考坐标系时当前点的邻域搜索半径

		rf_est.setInputCloud(model_keypoints);
		rf_est.setInputNormals(model_normals);
		rf_est.setSearchSurface(cloud_model);
		rf_est.compute(*model_rf);

		rf_est.setInputCloud(scene_keypoints);
		rf_est.setInputNormals(scene_normals);
		rf_est.setSearchSurface(cloud_scene);
		rf_est.compute(*scene_rf);

		//Clustering聚类化
		pcl::Hough3DGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA,
						pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
		clusterer.setHoughBinSize(cg_size_);						//设置Hough空间的采样间隔
		clusterer.setHoughThreshold(cg_thresh_);					//在Hough空间确定是否有实例存在的最少票数阈值，即允许的最小聚类大小
		clusterer.setUseInterpolation(true);						//设置是否对投票在Hough空间进行插值计算
		clusterer.setUseDistanceWeight(false);						//设置在投票时是否将对应点之间的距离作为权重参与计算

		clusterer.setInputCloud(model_keypoints);					//设置模型关键点
		clusterer.setInputRf(model_rf);								//设置模型对应的LRF
		clusterer.setSceneCloud(scene_keypoints);					//设置场景关键点
		clusterer.setSceneRf(scene_rf);								//设置场景对应的LRF
		clusterer.setModelSceneCorrespondences(model_scene_corrs);	//设置模型与场景的对应点对集合

		//clusterer.cluster (clustered_corrs);
		clusterer.recognize(rototranslations, clustered_corrs);		//结果包含变换矩阵和对应点聚类结果
	}
	else // Using GeometricConsistency使用几何一致性聚类算法
	{
		pcl::GeometricConsistencyGrouping<pcl::PointXYZRGBA, 
						pcl::PointXYZRGBA> gc_clusterer;
		gc_clusterer.setGCSize(cg_size_);								//设置检查几何一致性时的空间分辨率
		gc_clusterer.setGCThreshold(cg_thresh_);						//设置最小的聚类数量

		gc_clusterer.setInputCloud(model_keypoints);					//设置模型关键点
		gc_clusterer.setSceneCloud(scene_keypoints);					//设置场景关键点
		gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);	

		//gc_clusterer.cluster (clustered_corrs);
		gc_clusterer.recognize(rototranslations, clustered_corrs);		//结果：变换矩阵，对应点聚类结果
	}

	if (rototranslations.size() <= 0)
	{
		cout << "*** No instances found! ***" << endl;
		return;
	}
	cout << "Model instances found: " << rototranslations.size() << endl << endl;

	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
		std::cout << "        Correspondences belonging to this instance: " 
				  << clustered_corrs[i].size() << std::endl;

		// Print the rotation matrix and translation vector
		Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);		//旋转矩阵
		Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);		//平移向量

		printf("\n");
		printf("            | %6.3f %6.3f %6.3f | \n",
			rotation(0, 0), rotation(0, 1), rotation(0, 2));
		printf("        R = | %6.3f %6.3f %6.3f | \n",
			rotation(1, 0), rotation(1, 1), rotation(1, 2));
		printf("            | %6.3f %6.3f %6.3f | \n", 
			rotation(2, 0), rotation(2, 1), rotation(2, 2));
		printf("\n");

		printf("        t = < %0.3f, %0.3f, %0.3f >\n",
			translation(0), translation(1), translation(2));
	}
}
