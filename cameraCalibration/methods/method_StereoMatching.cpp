#include "method_StereoMatching.h"
#include <boost/container/container_fwd.hpp>
#include "parametersStereo.h"
#include <pcl/common/impl/accumulators.hpp>
#include <cvt/wbuffer>

using namespace cv;
using namespace std;

float operator* (Vec3f param1, Vec3f param2)
{
	return param1[0] * param2[0] + param1[1] * param2[1] + param1[2] * param2[2];
}

float operator* (Vec6f param1, Vec6f param2)
{
	return param1[0] * param2[0] + param1[1] * param2[1] + param1[2] * param2[2]
		 + param1[3] * param2[3] + param1[4] * param2[4] + param1[5] * param2[5];
}

//----------------------------------------------------
//--------------match cost computation----------------
//----------------------------------------------------
/**
* \brief compute the disparity using AD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
* \param leftImg
* \param rightImg
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeAD(cv::Mat leftImg, cv::Mat rightImg, int minDisparity, int numDisparity)
{
	if (leftImg.type() != CV_8UC1)
	{
		if (leftImg.channels() != 1)
		{
			cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		}
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.type() != CV_8UC1)
	{
		if (rightImg.channels() != 1)
		{
			cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		}
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = numDisparity - minDisparity; i < imgWidth; i++)
		{
			Mat leftToCompare(1, numDisparity, CV_8UC1, Scalar::all(leftImg.at<char>(j, i)));
			Mat rightToCompare = rightImg(Rect(i - numDisparity, j, numDisparity, 1));
			Mat allCost;
			absdiff(leftToCompare, rightToCompare, allCost);
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	return disparityMap;
}

/**
* \brief compute the disparity using SAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:processing time with baby stereo pair: 32.6058 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param minDisparity
* \param numDisparity
*/
cv::Mat stereomatch_1::computeSAD(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;
	int halfWinSize = winSize / 2;

	Mat leftKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat rightKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			leftKernel = leftImg(Rect(i - halfWinSize, j - halfWinSize, winSize, winSize));
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				int candiD = i - minDisparity - k;
				if (candiD >= halfWinSize)
				{
					rightKernel = rightImg(Rect(candiD - halfWinSize, j - halfWinSize, winSize, winSize));
					Mat differMap;
					absdiff(leftKernel, rightKernel, differMap);
					Scalar SAD_sum = sum(differMap);
					allCost.at<float>(k) = (float)SAD_sum[0];
				}
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "SAD����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using SAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:using integral image to optimize the speed
*					experiment shows that the processing time is dramatically reduced. (32.6058 sec -> 7.95375 sec)
*			processing time with baby stereo pair: 7.95375 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSAD_inteOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_integral;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		differ_ranges.push_back(differWhole);
		Mat differWholeInte;
		integral(differWhole, differWholeInte, CV_32F);
		differ_integral.push_back(differWholeInte);
	}

	int halfWinSize = winSize / 2;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				allCost.at<float>(k) = differ_integral[k].at<float>(j + halfWinSize, i + halfWinSize)
					- differ_integral[k].at<float>(j - halfWinSize, i + halfWinSize)
					- differ_integral[k].at<float>(j + halfWinSize, i - halfWinSize)
					+ differ_integral[k].at<float>(j - halfWinSize, i - halfWinSize);
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "SAD_integral����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using SAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:using boxfiltering algorithm to optimize the speed
*					experiment shows that the processing time is dramatically reduced. (32.6058 sec -> 3.52965 sec)
*			processing time with baby stereo pair: 3.52965 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSAD_BFOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_BF;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		differ_ranges.push_back(differWhole);

		differWhole.convertTo(differWhole, CV_32FC1);
		Mat differWholeBF;
		sqrBoxFilter(differWhole, differWholeBF, -1, Size(winSize, winSize), Point(-1, -1), false);
		differ_BF.push_back(differWholeBF);
	}

	int halfWinSize = winSize / 2;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = 0; i < imgWidth; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				allCost.at<float>(k) = (float)differ_BF[k].at<double>(j, i);
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "SAD_BFOpti����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using TAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
* \param leftImg
* \param rightImg
* \param threshold_T :the truncated threshold
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeTAD(cv::Mat leftImg, cv::Mat rightImg, int threshold_T, int minDisparity, int numDisparity)
{
	if (leftImg.type() != CV_8UC1)
	{
		if (leftImg.channels() != 1)
		{
			cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		}
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.type() != CV_8UC1)
	{
		if (rightImg.channels() != 1)
		{
			cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		}
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = numDisparity - minDisparity; i < imgWidth; i++)
		{
			Mat leftToCompare(1, numDisparity, CV_8UC1, Scalar::all(leftImg.at<char>(j, i)));
			Mat rightToCompare = rightImg(Rect(i - numDisparity, j, numDisparity, 1));
			Mat allCost;
			absdiff(leftToCompare, rightToCompare, allCost);
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * min(minLoc.x, threshold_T);
		}
	}

	return disparityMap;
}

/**
* \brief compute the disparity using STAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:processing time with baby stereo pair: 31.4675 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param threshold_T :the truncated threshold
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSTAD(cv::Mat leftImg, cv::Mat rightImg, int winSize, int threshold_T, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int halfWinSize = winSize / 2;
	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	Mat leftKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat rightKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			leftKernel = leftImg(Rect(i - halfWinSize, j - halfWinSize, winSize, winSize));
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				int candiD = i - minDisparity - k;
				if (candiD >= halfWinSize)
				{
					rightKernel = rightImg(Rect(candiD - halfWinSize, j - halfWinSize, winSize, winSize));
					Mat differMap;
					absdiff(leftKernel, rightKernel, differMap);
					Scalar SAD_sum = sum(differMap);
					allCost.at<float>(k) = (float)SAD_sum[0];
				}
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * min(minLoc.x, threshold_T);
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "STAD����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using STAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:using integral image to optimize the speed
*					experiment shows that the processing time is dramatically reduced. (31.4675 sec -> 9.52172 sec)
*			processing time with baby stereo pair: 9.52172 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param threshold_T
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSTAD_inteOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize, int threshold_T, int minDisparity,
	int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_integral;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		differ_ranges.push_back(differWhole);
		Mat differWholeInte;
		integral(differWhole, differWholeInte, CV_32F);
		differ_integral.push_back(differWholeInte);
	}

	int halfWinSize = winSize / 2;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				allCost.at<float>(k) = differ_integral[k].at<float>(j + halfWinSize, i + halfWinSize)
					- differ_integral[k].at<float>(j - halfWinSize, i + halfWinSize)
					- differ_integral[k].at<float>(j + halfWinSize, i - halfWinSize)
					+ differ_integral[k].at<float>(j - halfWinSize, i - halfWinSize);
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * min(minLoc.x, threshold_T);
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "STAD_inteOpti����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using STAD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:using Box-Filtering algorithm to optimize the speed
*					experiment shows that the processing time is dramatically reduced. (31.4675 sec -> 3.31762 sec)
*			processing time with baby stereo pair: 3.31762 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param threshold_T
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSTAD_BFOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize, int threshold_T, int minDisparity,
	int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_BF;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		differ_ranges.push_back(differWhole);

		differWhole.convertTo(differWhole, CV_32FC1);
		Mat differWholeBF;
		sqrBoxFilter(differWhole, differWholeBF, -1, Size(winSize, winSize), Point(-1, -1), false);
		differ_BF.push_back(differWholeBF);
	}

	int halfWinSize = winSize / 2;

	Mat rightKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				allCost.at<float>(k) = (float)differ_BF[k].at<double>(j, i);
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * min(minLoc.x, threshold_T);
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "STAD_BFOpti����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using SD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
* \param leftImg
* \param rightImg
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSD(cv::Mat leftImg, cv::Mat rightImg, int minDisparity, int numDisparity)
{
	if (leftImg.type() != CV_8UC1)
	{
		if (leftImg.channels() != 1)
		{
			cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		}
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.type() != CV_8UC1)
	{
		if (rightImg.channels() != 1)
		{
			cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		}
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = numDisparity - minDisparity; i < imgWidth; i++)
		{
			Mat leftToCompare(1, numDisparity, CV_8UC1, Scalar::all(leftImg.at<char>(j, i)));
			Mat rightToCompare = rightImg(Rect(i - numDisparity, j, numDisparity, 1));
			Mat allCost;
			absdiff(leftToCompare, rightToCompare, allCost);
			allCost = allCost.mul(allCost);
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	return disparityMap;
}

/**
* \brief compute the disparity using SSD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:processing time with baby stereo pair: 32.6881 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSSD(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int halfWinSize = winSize / 2;
	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	Mat leftKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat rightKernel(Size(winSize, winSize), CV_8U, Scalar::all(0));
	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			leftKernel = leftImg(Rect(i - halfWinSize, j - halfWinSize, winSize, winSize));
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				int candiD = i - minDisparity - k;
				if (candiD >= halfWinSize)
				{
					rightKernel = rightImg(Rect(candiD - halfWinSize, j - halfWinSize, winSize, winSize));
					Mat differMap;
					absdiff(leftKernel, rightKernel, differMap);
					differMap = differMap.mul(differMap);
					Scalar SAD_sum = sum(differMap);
					allCost.at<float>(k) = (float)SAD_sum[0];
				}
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "SSD����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using SSD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:using integral image to optimize the speed
*					experiment shows that the processing time is dramatically reduced.(32.6881 sec -> 9.30113 sec)
*			processing time with baby stereo pair: 9.30113 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSSD_inteOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_integral;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		pow(differWhole, 2, differWhole);
		differ_ranges.push_back(differWhole);
		Mat differWholeInte;
		integral(differWhole, differWholeInte, CV_32F);
		differ_integral.push_back(differWholeInte);
	}

	int halfWinSize = winSize / 2;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				allCost.at<float>(k) = differ_integral[k].at<float>(j + halfWinSize, i + halfWinSize)
					- differ_integral[k].at<float>(j - halfWinSize, i + halfWinSize)
					- differ_integral[k].at<float>(j + halfWinSize, i - halfWinSize)
					+ differ_integral[k].at<float>(j - halfWinSize, i - halfWinSize);
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "SSD_inteOpti����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

/**
* \brief compute the disparity using SSD algorithm with fixed window size(FW) and winner takes all(WTA) strategy
*			\notice:using Box-Filtering algorithm to optimize the speed
*					experiment shows that the processing time is dramatically reduced.(32.6881 sec -> 3.2409 sec)
*			processing time with baby stereo pair: 3.2409 sec
* \param leftImg
* \param rightImg
* \param winSize
* \param minDisparity
* \param numDisparity
* \return
*/
cv::Mat stereomatch_1::computeSSD_BFOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_BF;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		pow(differWhole, 2, differWhole);
		differ_ranges.push_back(differWhole);

		differWhole.convertTo(differWhole, CV_32FC1);
		Mat differWholeBF;
		sqrBoxFilter(differWhole, differWholeBF, -1, Size(winSize, winSize), Point(-1, -1), false);
		differ_BF.push_back(differWholeBF);
	}

	int halfWinSize = winSize / 2;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = halfWinSize; j < imgHeight - halfWinSize; j++)
	{
		for (int i = halfWinSize; i < imgWidth - halfWinSize; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				allCost.at<float>(k) = (float)differ_BF[k].at<double>(j, i);
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "SSD_BFOpti����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

void stereomatch_1::getInputImgNCC(cv::Mat src, std::vector<std::vector<cv::Mat> > & dst, int winSize)
{
	int width = src.cols;
	int height = src.rows;

	if(winSize % 2 == 0)
	{
		return;
	}
	int halfWinSize = winSize / 2;

	if(src.channels() == 3)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	Mat src_border;
	copyMakeBorder(src, src_border, halfWinSize, halfWinSize, halfWinSize, halfWinSize, BORDER_REFLECT);
	src_border.convertTo(src_border, CV_32FC1);

	Mat meanSrc;
	boxFilter(src, meanSrc, CV_32FC1, Size(winSize, winSize));

	for (int y = 0; y < height; y++)
	{
		std::vector<cv::Mat> supportWin_row;
		for (int x = 0; x < width; x++)
		{
			Mat supportWin = src_border(Rect(x, y, winSize, winSize)) - meanSrc.at<float>(y, x);
			supportWin_row.push_back(supportWin);
 		}
		dst.push_back(supportWin_row);
	}
}

/**
 * \brief ��DSI��disparity space image���м���NCCƥ�����ֵ
 * \param leftImg ��У����������ͼ����ͼ��
 * \param rightImg ��У����������ͼ����ͼ��
 * \param dispType ��ͼ����вο�ͼ���ѡ��������ͼ��������ͼΪ�ο�ͼ��
 * \param winSize ���Ӵ��߳�
 * \param minDisparity ����С�Ӳ�ֵ
 * \param numDisparity ��DSI�е���ɢ�Ӳ�ֵ����
 */
cv::Mat stereomatch_1::computeNCC(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType, int winSize, int minDisparity,
	int numDisparity)
{
	int width = leftImg.cols;
	int height = leftImg.rows;

	int min_offset = minDisparity;
	int max_offset = minDisparity + numDisparity - 1;

	if (leftImg.size != rightImg.size)
	{
		return Mat();
	}

	if (winSize % 2 == 0)
	{
		return Mat();
	}

	if (leftImg.channels() == 3)
	{
		cvtColor(leftImg, leftImg, COLOR_RGB2GRAY);
	}

	if (rightImg.channels() == 3)
	{
		cvtColor(rightImg, rightImg, COLOR_RGB2GRAY);
	}

	// calculate each pixel's ASW value
	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > max_asw; // store min ASW value
	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max()); //tmp��ʾͼ���һ������
		max_asw.push_back(tmp);
	}

	if (dispType == DISPARITY_LEFT)
	{
		Mat right_border;
		copyMakeBorder(rightImg, right_border, 0, 0, max_offset, 0, BORDER_REFLECT);

		vector<vector<Mat> > leftSupWins, rightSupWins;
		getInputImgNCC(leftImg, leftSupWins, winSize);
		getInputImgNCC(right_border, rightSupWins, winSize);

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				Mat leftWin = leftSupWins[y][x];
				for(int offset = min_offset; offset < max_offset; offset++)
				{
					Mat rightWin = rightSupWins[y][x + max_offset - offset];
					double cost_d = sum(leftWin.mul(rightWin))[0] 
    					/ (sum(leftWin.mul(leftWin))[0] * sum(rightWin.mul(rightWin))[0]);

					if (cost_d < max_asw[y][x])
					{
						max_asw[y][x] = cost_d;
						// for better visualization
						depth.at<float>(y, x) = (float)offset;
					}
				}
			}
		}
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		Mat left_border;
		copyMakeBorder(leftImg, left_border, 0, 0, 0, max_offset, BORDER_REFLECT);

		vector<vector<Mat> > leftSupWins, rightSupWins;
		getInputImgNCC(left_border, leftSupWins, winSize);
		getInputImgNCC(rightImg, rightSupWins, winSize);

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				Mat rightWin = rightSupWins[y][x];
				for (int offset = min_offset; offset < max_offset; offset++)
				{
					Mat leftWin = leftSupWins[y][x + offset];
					double cost_d = sum(leftWin.mul(rightWin))[0]
						/ (sum(leftWin.mul(leftWin))[0] * sum(rightWin.mul(rightWin))[0]);

					if (cost_d > max_asw[y][x])
					{
						max_asw[y][x] = cost_d;
						// for better visualization
						depth.at<float>(y, x) = (float)offset;
					}
				}
			}
		}
	}

	return depth;
}

void stereomatch_1::computeNCC(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_ds, DisparityType dispType, int winSize,
	int minDisparity, int numDisparity)
{
	int width = leftImg.cols;
	int height = leftImg.rows;

	int min_offset = minDisparity;
	int max_offset = minDisparity + numDisparity - 1;

	if (leftImg.size != rightImg.size)
	{
		return;
	}

	if (winSize % 2 == 0)
	{
		return;
	}

	if(!cost_ds.empty())
	{
		cost_ds.clear();
	}

	if (leftImg.channels() == 3)
	{
		cvtColor(leftImg, leftImg, COLOR_RGB2GRAY);
	}

	if (rightImg.channels() == 3)
	{
		cvtColor(rightImg, rightImg, COLOR_RGB2GRAY);
	}

	// calculate each pixel's NCC cost
	if (dispType == DISPARITY_LEFT)
	{
		Mat right_border;
		copyMakeBorder(rightImg, right_border, 0, 0, max_offset, 0, BORDER_REFLECT);

		vector<vector<Mat> > leftSupWins, rightSupWins;
		getInputImgNCC(leftImg, leftSupWins, winSize);
		getInputImgNCC(right_border, rightSupWins, winSize);

		for (int offset = min_offset; offset <= max_offset; offset++)
		{
			Mat curCost_(height, width, CV_32FC1);
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					Mat leftWin = leftSupWins[y][x];
					Mat rightWin = rightSupWins[y][x + max_offset - offset];
					curCost_.at<float>(y, x) = sum(leftWin.mul(rightWin))[0]
						/ (sum(leftWin.mul(leftWin))[0] * sum(rightWin.mul(rightWin))[0]);
				}
			}
			Mat curCost_norm;
			normalize(curCost_, curCost_norm, 0, 1, NORM_MINMAX);
			cost_ds.push_back(curCost_norm);
		}
	}
	else if (dispType == DISPARITY_RIGHT)
	{
		Mat left_border;
		copyMakeBorder(leftImg, left_border, 0, 0, 0, max_offset, BORDER_REFLECT);

		vector<vector<Mat> > leftSupWins, rightSupWins;
		getInputImgNCC(left_border, leftSupWins, winSize);
		getInputImgNCC(rightImg, rightSupWins, winSize);

		for (int offset = min_offset; offset <= max_offset; offset++)
		{
			Mat curCost_(height, width, CV_32FC1);
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					Mat rightWin = rightSupWins[y][x];
					Mat leftWin = leftSupWins[y][x + offset];
					curCost_.at<float>(y, x) = sum(leftWin.mul(rightWin))[0]
						/ (sum(leftWin.mul(leftWin))[0] * sum(rightWin.mul(rightWin))[0]);
				}
			}
			Mat curCost_norm;
			normalize(curCost_, curCost_norm, 0, 1, NORM_MINMAX);
			cost_ds.push_back(curCost_norm);
		}
	}
}

void stereomatch_1::computeSimilarity(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_d_imgs,
	double regularity, double thresC, double thresG, DisparityType dispType,
	int minDisparity, int numDisparity)
{
	int width = leftImg.cols;
	int height = leftImg.rows;

	int min_offset = minDisparity;
	int max_offset = minDisparity + numDisparity - 1;

	if (!cost_d_imgs.empty())
	{
		cost_d_imgs.clear();
	}

	if (leftImg.size != rightImg.size)
	{
		return;
	}

	double regularityR = 1 - regularity;

	if (leftImg.channels() == 3 && rightImg.channels() == 3)
	{
		if (dispType == DISPARITY_LEFT)
		{
			Mat right_border;
			copyMakeBorder(rightImg, right_border, 0, 0, max_offset, 0, BORDER_REFLECT);

			//�ݶ�����
			Mat sobel_x_left, soble_x_right;
			Mat soble_x_kernel = (Mat_<char>(3, 3) << -3, 0, 3,
				-10, 0, 10,
				-3, 0, 3);
			filter2D(leftImg, sobel_x_left, CV_32F, soble_x_kernel);
			filter2D(right_border, soble_x_right, CV_32F, soble_x_kernel);

			for (int offset = min_offset; offset <= max_offset; offset++)
			{
				Mat color_temp;
				absdiff(leftImg, right_border(Rect(max_offset - offset, 0, width, height)), color_temp);

				vector<Mat> color_channels;
				split(color_temp, color_channels);
				Mat color_ = (color_channels[0] + color_channels[1] + color_channels[2]) / 3;

				Mat compare_thresC(color_.size(), CV_8UC1);
				compare(color_, thresC, compare_thresC, CMP_GT);

				Mat curCost_color = color_.mul(compare_thresC / 255) + thresC * (compare_thresC / 255);
				curCost_color.convertTo(curCost_color, CV_32FC1);


				color_channels.clear();
				Mat curGridient_temp;
				absdiff(sobel_x_left, soble_x_right(Rect(max_offset - offset, 0, width, height)), curGridient_temp);
				split(curGridient_temp, color_channels);

				Mat curGridient_ = (color_channels[0] + color_channels[1] + color_channels[2]) / 3;
				Mat compare_thresG(curGridient_.size(), CV_8UC1);
				compare(curGridient_, thresG, compare_thresG, CMP_GT);

				Mat bitImg = compare_thresG / 255;
				Mat bit_not_img;
				bitwise_not(bitImg, bit_not_img);
				bitImg.convertTo(bitImg, CV_32FC1);
				bit_not_img.convertTo(bit_not_img, CV_32FC1);
				Mat curCost_grident = curGridient_.mul(bitImg) + thresG * bit_not_img;

				Mat curCost_d = regularityR * curCost_color + regularity * curCost_grident;
				cost_d_imgs.push_back(curCost_d);
			}
		}
		else if (dispType == DISPARITY_RIGHT)
		{
			Mat left_border;
			copyMakeBorder(leftImg, left_border, 0, 0, 0, max_offset, BORDER_REFLECT);

			Mat sobel_x_left, soble_x_right;
			Mat soble_x_kernel = (Mat_<char>(3, 3) << -3, 0, 3,
				-10, 0, 10,
				-3, 0, 3);
			filter2D(left_border, sobel_x_left, CV_32F, soble_x_kernel);
			filter2D(rightImg, soble_x_right, CV_32F, soble_x_kernel);

			for (int offset = min_offset; offset <= max_offset; offset++)
			{
				Mat color_temp;
				absdiff(left_border(Rect(offset, 0, width, height)), rightImg, color_temp);
				vector<Mat> color_channels;
				split(color_temp, color_channels);
				Mat color_ = (color_channels[0] + color_channels[1] + color_channels[2]) / 3;

				Mat compare_thresC(color_.size(), CV_8UC1);
				compare(color_, thresC, compare_thresC, CMP_GT);

				Mat curCost_color = color_.mul(compare_thresC / 255) + thresC * (compare_thresC / 255);
				curCost_color.convertTo(curCost_color, CV_32FC1);


				color_channels.clear();
				Mat curGridient_temp;
				absdiff(sobel_x_left(Rect(offset, 0, width, height)), soble_x_right, curGridient_temp);
				split(curGridient_temp, color_channels);
				Mat curGridient_ = (color_channels[0] + color_channels[1] + color_channels[2]) / 3;
				
				Mat compare_thresG(curGridient_.size(), CV_8UC1);
				compare(curGridient_, thresG, compare_thresG, CMP_GT);

				Mat curCost_grident;
				curCost_grident = curGridient_.mul(compare_thresG / 255) + thresG * (compare_thresG / 255);


				Mat curCost_d = regularityR * curCost_color + regularity * curCost_grident;
				cost_d_imgs.push_back(curCost_d);
			}
		}
	}
	else if (leftImg.channels() == 1 && rightImg.channels() == 1)
	{
		if (dispType == DISPARITY_LEFT)
		{
			Mat right_border;
			copyMakeBorder(rightImg, right_border, 0, 0, max_offset, 0, BORDER_REFLECT);

			Mat sobel_x_left, soble_x_right;
			Mat soble_x_kernel = (Mat_<char>(3, 3) << -3, 0, 3,
				-10, 0, 10,
				-3, 0, 3);
			filter2D(leftImg, sobel_x_left, CV_32F, soble_x_kernel);
			filter2D(right_border, soble_x_right, CV_32F, soble_x_kernel);

			for (int offset = min_offset; offset <= max_offset; offset++)
			{
				Mat color_;
				absdiff(leftImg, right_border(Rect(max_offset - offset, 0, width, height)), color_);
				Mat compare_thresC(color_.size(), CV_8UC1);
				compare(color_, thresC, compare_thresC, CMP_GT);

				Mat curCost_color;
				curCost_color = color_.mul(compare_thresC / 255) + thresC * (compare_thresC / 255);
				curCost_color.convertTo(curCost_color, CV_32FC1);

				Mat curGridient_;
				absdiff(sobel_x_left, soble_x_right(Rect(max_offset - offset, 0, width, height)), curGridient_);
				Mat compare_thresG(curGridient_.size(), CV_8UC1);
				compare(curGridient_, thresG, compare_thresG, CMP_GT);

				Mat curCost_grident;
				curCost_grident = curGridient_.mul(compare_thresG / 255) + thresG * (compare_thresG / 255);

				Mat curCost_d = regularityR * curCost_color + regularity * curCost_grident;
				cost_d_imgs.push_back(curCost_d);
			}
		}
		else if(dispType == DISPARITY_RIGHT)
		{
			Mat left_border;
			copyMakeBorder(leftImg, left_border, 0, 0, 0, max_offset, BORDER_REFLECT);

			Mat sobel_x_left, soble_x_right;
			Mat soble_x_kernel = (Mat_<char>(3, 3) << -3, 0, 3,
				-10, 0, 10,
				-3, 0, 3);
			filter2D(left_border, sobel_x_left, CV_32F, soble_x_kernel);
			filter2D(rightImg, soble_x_right, CV_32F, soble_x_kernel);

			for (int offset = min_offset; offset <= max_offset; offset++)
			{
				Mat color_;
				absdiff(left_border(Rect(offset, 0, width, height)), rightImg, color_);
				Mat compare_thresC(color_.size(), CV_8UC1);
				compare(color_, thresC, compare_thresC, CMP_GT);

				Mat curCost_color;
				curCost_color = color_.mul(compare_thresC / 255) + thresC * (compare_thresC / 255);
				curCost_color.convertTo(curCost_color, CV_32FC1);

				Mat curGridient_;
				absdiff(sobel_x_left(Rect(offset, 0, width, height)), soble_x_right, curGridient_);
				Mat compare_thresG(curGridient_.size(), CV_8UC1);
				compare(curGridient_, thresG, compare_thresG, CMP_GT);

				Mat curCost_grident;
				curCost_grident = curGridient_.mul(compare_thresG / 255) + thresG * (compare_thresG / 255);

				Mat curCost_d = regularityR * curCost_color + regularity * curCost_grident;
				cost_d_imgs.push_back(curCost_d);
			}
		}
	}

	//// calculate each pixel's ASW value
	//Mat depth(height, width, CV_32FC1);
	//vector< vector<double> > min_asw; // store min ASW value
	//for (int i = 0; i < height; ++i)
	//{
	//	vector<double> tmp(width, numeric_limits<double>::min()); //tmp��ʾͼ���һ������
	//	min_asw.push_back(tmp);
	//}

	//for (int x = 0; x < width; x++)
	//{
	//	for (int y = 0; y < height; y++)
	//	{
	//		for (int disp = 0; disp < numDisparity; disp++)
	//		{
	//			double curCost_ = cost_d_imgs[disp].at<float>(y, x);

	//			if (curCost_ > min_asw[y][x])
	//			{
	//				min_asw[y][x] = curCost_;
	//				// for better visualization
	//				depth.at<float>(y, x) = (float)(disp + minDisparity);
	//			}
	//		}
	//	}
	//}

	//depth.convertTo(depth, CV_8UC1);

}

void stereomatch_1::computeSimilarity(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_d_imgs, double regularity,
	double thresC, double thresG, DisparityType dispType, int winSize, int minDisparity, int numDisparity)
{
	if(winSize % 2 == 0)
	{
		return;
	}
	int halfWinSize = winSize / 2;

	computeSimilarity(leftImg, rightImg, cost_d_imgs, regularity, thresC, thresG, dispType, minDisparity, numDisparity);

	for(std::vector<cv::Mat>::iterator itor = cost_d_imgs.begin(); itor != cost_d_imgs.end(); itor++)
	{
		Mat imgDst;
		copyMakeBorder(*itor, imgDst, halfWinSize, halfWinSize, halfWinSize, halfWinSize, BORDER_REFLECT);
		*itor = imgDst;
	}
}

cv::Mat stereomatch_1::computeShiftableWin(cv::Mat leftImg, cv::Mat rightImg, int winSize, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	//if (winSize % 2 == 0)
	//{
	//	std::cout << "winsize must be odd" << std::endl;
	//	return Mat();
	//}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_BF;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		differ_ranges.push_back(differWhole);

		differWhole.convertTo(differWhole, CV_32FC1);
		std::vector<Mat> differBF_SW;
		Mat differBF;
		for(int j = 0; j < winSize; j++)
		{
			for(int k = 0; k < winSize; k++)
			{
				sqrBoxFilter(differWhole, differBF, -1, Size(winSize, winSize), Point(j, k), false);
				differBF_SW.push_back(differBF);
			}
		}
		Mat differBFWhole;
		merge(differBF_SW, differBFWhole);
		differ_BF.push_back(differBFWhole);
	}

	int halfWinSize = winSize / 2;
	int shiftNum = winSize * winSize;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = 0; i < imgWidth; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			Vec<double, 9> pixelInd;
			for (int k = 0; k < numDisparity; k++)
			{
				pixelInd = differ_BF[k].at<Vec<double, 9>>(j, i);
				Point minLoc;
				minMaxLoc(pixelInd, NULL, NULL, &minLoc, NULL);
				allCost.at<float>(k) = pixelInd[minLoc.x];
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "ShiftableWin����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;

}

cv::Mat stereomatch_1::computeMultiWin(cv::Mat leftImg, cv::Mat rightImg, int winSize, int winNum, int minDisparity, int numDisparity)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	//if (winSize % 2 == 0)
	//{
	//	std::cout << "winsize must be odd" << std::endl;
	//	return Mat();
	//}

	int imgHeight = leftImg.rows;
	int imgWidth = leftImg.cols;

	//optimization:using integral image
	std::vector<Mat> differ_ranges;
	std::vector<Mat> differ_BF;
	for (int i = 0; i < numDisparity; i++)
	{
		Mat differWhole(leftImg.size(), CV_8U, Scalar::all(0));
		absdiff(leftImg(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)),
			rightImg(Rect(0, 0, imgWidth - minDisparity - i, imgHeight)),
			differWhole(Rect(minDisparity + i, 0, imgWidth - minDisparity - i, imgHeight)));
		differ_ranges.push_back(differWhole);

		differWhole.convertTo(differWhole, CV_32FC1);
		Mat differWholeBF;
		sqrBoxFilter(differWhole, differWholeBF, -1, Size(winSize, winSize), Point(-1, -1), false);
		differ_BF.push_back(differWholeBF);
	}

	int halfWinSize = winSize / 2;

	Mat disparityMap(imgHeight, imgWidth, CV_8U, Scalar::all(0));

	for (int j = 0; j < imgHeight; j++)
	{
		for (int i = 0; i < imgWidth; i++)
		{
			Mat allCost(1, numDisparity, CV_32F, Scalar::all(0));
			for (int k = 0; k < numDisparity; k++)
			{
				std::multiset<double> multiWins;
				switch(winNum)
				{
				case 9:
				{
					int jValuedB = max(0, j - winSize);
					int iValuedB= max(0, i - winSize);
					int jValuedT = min(j + winSize, imgHeight - 1);
					int iValuedT = min(i + winSize, imgWidth - 1);

					multiWins.insert(differ_BF[k].at<double>(jValuedB, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(jValuedB, i));
					multiWins.insert(differ_BF[k].at<double>(jValuedB, iValuedT));
					multiWins.insert(differ_BF[k].at<double>(j, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(j, iValuedT));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, i));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, iValuedT));

					std::multiset<double>::iterator itor = multiWins.begin();
					allCost.at<float>(k) = (float)differ_BF[k].at<double>(j, i)
						+ *itor + *itor++ + *itor++ + *itor++;
					break;
				}
				case 25:
				{
					int jValuedB = max(0, j - winSize);
					int iValuedB = max(0, i - winSize);
					int jValuedT = min(j + winSize, imgHeight - 1);
					int iValuedT = min(i + winSize, imgWidth - 1);

					int j2ValuedB = max(0, j - 2 * winSize);
					int i2ValuedB = max(0, i - 2 * winSize);
					int j2ValuedT = min(j + 2 * winSize, imgHeight - 1);
					int i2ValuedT = min(i + 2 * winSize, imgWidth - 1);

					multiWins.insert(differ_BF[k].at<double>(jValuedB, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(jValuedB, i));
					multiWins.insert(differ_BF[k].at<double>(jValuedB, iValuedT));
					multiWins.insert(differ_BF[k].at<double>(j, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(j, iValuedT));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, i));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, iValuedT));

					multiWins.insert(differ_BF[k].at<double>(j2ValuedB, i2ValuedB));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedB, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedB, i));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedB, iValuedT));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedB, i2ValuedT));

					multiWins.insert(differ_BF[k].at<double>(jValuedB, i2ValuedB));
					multiWins.insert(differ_BF[k].at<double>(j, i2ValuedB));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, i2ValuedB));

					multiWins.insert(differ_BF[k].at<double>(j2ValuedT, i2ValuedB));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedT, iValuedB));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedT, i));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedT, iValuedT));
					multiWins.insert(differ_BF[k].at<double>(j2ValuedT, i2ValuedT));

					multiWins.insert(differ_BF[k].at<double>(jValuedB, i2ValuedT));
					multiWins.insert(differ_BF[k].at<double>(j, i2ValuedT));
					multiWins.insert(differ_BF[k].at<double>(jValuedT, i2ValuedT));

					std::multiset<double>::iterator itor = multiWins.begin();
					allCost.at<float>(k) = (float)differ_BF[k].at<double>(j, i)
						+ *itor + *itor++ + *itor++ + *itor++ + *itor++ + *itor++ + *itor++ + *itor++;
					break;
				}
				}
			}
			Point minLoc;
			minMaxLoc(allCost, NULL, NULL, &minLoc, NULL);
			disparityMap.at<char>(j, i) = 16 * minLoc.x;
		}
	}

	double  time = ((double)getTickCount() - start) / getTickFrequency();
	std::cout << "MultipleWin����ʱ��Ϊ��" << time << "��" << std::endl;

	return disparityMap;
}

cv::Mat stereomatch_1::ncc(cv::Mat in1, cv::Mat in2, std::string type, bool add_constant)
{
	int width = in1.size().width;
	int height = in1.size().height;
	int max_offset = 20;
	int kernel_size = 41; // window size

	Mat left;
	cvtColor(in1, left, COLOR_BGR2GRAY);
	Mat right;
	cvtColor(in2, right, COLOR_BGR2GRAY);

	if (add_constant)
	{
		right += 10;
	}

	Mat depth(height, width, 0);
	vector< vector<double> > max_ncc; // store max NCC value

	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, -2);
		max_ncc.push_back(tmp);
	}

	for (int offset = 1; offset <= max_offset; offset++)
	{
		Mat tmp(height, width, 0);
		// shift image depend on type to save calculation time
		if (type == "left")
		{
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < offset; x++)
				{
					tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
				}

				for (int x = offset; x < width; x++)
				{
					tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
				}
			}
		}
		else if (type == "right")
		{
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width - offset; x++)
				{
					tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
				}

				for (int x = width - offset; x < width; x++)
				{
					tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
				}
			}
		}
		else
		{
			Mat tmp(0, 0, 0);
			return tmp;
		}

		// calculate each pixel's NCC value
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int start_x = max(0, x - kernel_size);
				int start_y = max(0, y - kernel_size);
				int end_x = min(width - 1, x + kernel_size);
				int end_y = min(height - 1, y + kernel_size);
				double n = (end_y - start_y) * (end_x - start_x);
				double res_ncc = 0;

				if (type == "left")
				{
					double left_mean = 0, right_mean = 0;
					double left_std = 0, right_std = 0;
					double numerator = 0;

					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							left_mean += left.at<uchar>(i, j);
							right_mean += tmp.at<uchar>(i, j);
						}
					}

					left_mean /= n;
					right_mean /= n;

					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							left_std += pow(left.at<uchar>(i, j) - left_mean, 2);
							right_std += pow(tmp.at<uchar>(i, j) - right_mean, 2);
							numerator += (left.at<uchar>(i, j) - left_mean) * (tmp.at<uchar>(i, j) - right_mean);
						}
					}

					numerator /= n;
					left_std /= n;
					right_std /= n;
					res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
				}
				else
				{
					double left_mean = 0, right_mean = 0;
					double left_std = 0, right_std = 0;
					double numerator = 0;

					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							left_mean += tmp.at<uchar>(i, j);
							right_mean += right.at<uchar>(i, j);
						}
					}

					left_mean /= n;
					right_mean /= n;

					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							left_std += pow(tmp.at<uchar>(i, j) - left_mean, 2);
							right_std += pow(right.at<uchar>(i, j) - right_mean, 2);
							numerator += (tmp.at<uchar>(i, j) - left_mean) * (right.at<uchar>(i, j) - right_mean);
						}
					}

					numerator /= n;
					left_std /= n;
					right_std /= n;
					res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
				}

				// greater NCC value found
				if (res_ncc > max_ncc[y][x])
				{
					max_ncc[y][x] = res_ncc;
					// for better visualization
					depth.at<uchar>(y, x) = (uchar)(offset * 3);
				}
			}
		}
	}

	return depth;
}

cv::Mat stereomatch_1::asw(cv::Mat in1, cv::Mat in2, std::string type)
{
	int width = in1.size().width;
	int height = in1.size().height;
	int max_offset = 330;
	int min_offset = 186;
	int kernel_size = 35; // window size
	double k = 3, gamma_c = 30, gamma_g = 20; // ASW parameters

	Mat depth(height, width, 0);
	vector< vector<double> > min_asw; // store min ASW value

	Mat left;
	cvtColor(in1, left, COLOR_BGR2GRAY);
	Mat right;
	cvtColor(in2, right, COLOR_BGR2GRAY);

	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max());
		min_asw.push_back(tmp);
	}

	for (int offset = min_offset; offset <= max_offset; offset++)
	{
		Mat tmp(height, width, 0);
		// shift image depend on type to save calculation time
		if (type == "left")				//����ͼ��Ϊ�ο�ͼ��left
		{
			for (int y = 0; y < height; y++)
			{
				//��ԭͼ���ͼ��߽�����
				for (int x = 0; x < offset; x++)
				{
					tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
				}

				//�����ƶ�ͼ�񣬻�� �Ӳ�ֵ=offset ʱ ���м�ͼ��
				for (int x = offset; x < width; x++)
				{
					tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);		//��ͼ����
				}
			}
		}
		else if (type == "right")		//����ͼ��Ϊ�ο�ͼ��right
		{
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width - offset; x++)
				{
					tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);		//��ͼ����
				}

				for (int x = width - offset; x < width; x++)
				{
					tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
				}
			}
		}
		else
		{
			Mat tmp_(0, 0, 0);
			return tmp_;
		}

		// calculate each pixel's ASW value
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int start_x = max(0, x - kernel_size);
				int start_y = max(0, y - kernel_size);
				int end_x = min(width - 1, x + kernel_size);
				int end_y = min(height - 1, y + kernel_size);
				double E = 0;

				if (type == "left")
				{
					double numerator = 0;
					double denominator = 0;

					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							double delta_c1 = fabs(left.at<uchar>(i, j) - left.at<uchar>(y, x));
							double delta_c2 = fabs(tmp.at<uchar>(i, j) - tmp.at<uchar>(y, x));
							double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
							double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
							double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
							numerator += w1 * w2 * fabs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
							denominator += w1 * w2;
						}
					}

					E = numerator / denominator;
				}
				else
				{
					double numerator = 0;
					double denominator = 0;

					for (int i = start_y; i <= end_y; i++)
					{
						for (int j = start_x; j <= end_x; j++)
						{
							double delta_c1 = fabs(right.at<uchar>(i, j) - right.at<uchar>(y, x));
							double delta_c2 = fabs(tmp.at<uchar>(i, j) - tmp.at<uchar>(y, x));
							double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
							double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
							double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
							numerator += w1 * w2 * fabs(right.at<uchar>(i, j) - tmp.at<uchar>(i, j));
							denominator += w1 * w2;
						}
					}

					E = numerator / denominator;
				}

				// smaller ASW found
				if (E < min_asw[y][x])
				{
					min_asw[y][x] = E;
					// for better visualization
					depth.at<uchar>(y, x) = (uchar)(offset * 3);
				}
			}
		}
	}

	return depth;
}

cv::Mat stereomatch_1::computeAdaptiveWeight(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType, int winSize, int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity;
	int min_offset = minDisparity;
	int kernel_size = winSize; // window size
	double k = 3, gamma_c = 30, gamma_g = winSize*2/3; // ASW parameters

	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value

	//�ûҶ�ֵ��ʾ��ɫ���þ��ԻҶȲ�ֵ��ʾ���ص����ɫ����
	Mat left;
	cvtColor(leftImg, left, COLOR_BGR2GRAY);
	Mat right;
	cvtColor(rightImg, right, COLOR_BGR2GRAY);

	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max());
		min_asw.push_back(tmp);
	}

	vector<Mat> weightAllDirectLeft;
	vector<Mat> weightAllDirectRight;

	for (int j = - kernel_size/2; j < kernel_size/2 + 1; j++)
	{
		for (int i = - kernel_size/2; i < kernel_size/2 + 1; i++)
		{
			Mat weightOneDirectLeft(height, width, CV_32FC1);
			Mat weightOneDirectRight(height, width, CV_32FC1);
			if (i == 0 && j == 0)
			{
				continue;
			}
			double delta_g = sqrt(i * i + j * j);		//�ռ����
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					int neightbor_x = min(max(0, x + i), width-1);
					int neightbor_y = min(max(0, y + j), height -1);

					double delta_c1 = fabs(left.at<uchar>(neightbor_y, neightbor_x) - left.at<uchar>(y, x));		//��ɫ���룬���ԻҶȲ�ֵ
					double delta_c2 = fabs(right.at<uchar>(neightbor_y, neightbor_x) - right.at<uchar>(y, x));		//��ɫ����

					weightOneDirectLeft.at<float>(y, x) = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
					weightOneDirectRight.at<float>(y, x) = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
				}
			}
			weightAllDirectLeft.push_back(weightOneDirectLeft);
			weightAllDirectRight.push_back(weightOneDirectRight);
		}
	}

	for (int offset = min_offset; offset <= max_offset; offset++)
	{
		// calculate each pixel's ASW value
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double E = 0;

				if (dispType == DISPARITY_LEFT)
				{
					double numerator = 0;
					double denominator = 0;

					for (int i = 0; i < kernel_size * kernel_size-1; i++)
					{
						int kernel_x = 0, kernel_y = 0;
						if(i > kernel_size * kernel_size / 2)
						{
							kernel_x = (i+1) / kernel_size;
							kernel_y = (i+1) % kernel_size;
						}
						else
						{
							kernel_x = i / kernel_size;
							kernel_y = i % kernel_size;
						}
						int neightbor_x = min(max(0, x - kernel_size / 2 + kernel_x), width - 1);
						int neightbor_y = min(max(0, y - kernel_size / 2 + kernel_y), height - 1);

						numerator += weightAllDirectLeft[i].at<float>(y, x) 
							* weightAllDirectRight[i].at<float>(y, max(0, x-offset)) 
							* fabs(left.at<uchar>(neightbor_y, neightbor_x) - right.at<uchar>(neightbor_y, max(0, neightbor_x-offset)));
						denominator += weightAllDirectLeft[i].at<float>(y, x)
							* weightAllDirectRight[i].at<float>(y, max(0,x-offset));
					}

					E = numerator / denominator;
				}
				else
				{
					double numerator = 0;
					double denominator = 0;

					for (int i = 0; i < kernel_size * kernel_size-1; i++)
					{
												int kernel_x = 0, kernel_y = 0;
						if(i > kernel_size * kernel_size / 2)
						{
							kernel_x = (i+1) / kernel_size;
							kernel_y = (i+1) % kernel_size;
						}
						else
						{
							kernel_x = i / kernel_size;
							kernel_y = i % kernel_size;
						}
						int neightbor_x = min(max(0, x - kernel_size / 2 + kernel_x), width - 1);
						int neightbor_y = min(max(0, y - kernel_size / 2 + kernel_y), height - 1);

						numerator += weightAllDirectLeft[i].at<float>(y, min(x+offset, width-1))
							* weightAllDirectRight[i].at<float>(y, x) 
							* fabs(right.at<uchar>(neightbor_y, neightbor_x) - left.at<uchar>(neightbor_y, min(neightbor_x + offset, width-1)));
						denominator += weightAllDirectLeft[i].at<float>(y, min(x+offset, width - 1))
							* weightAllDirectRight[i].at<float>(y, x);
					}

					E = numerator / denominator;
				}

				// smaller ASW found
				if (E < min_asw[y][x])
				{
					min_asw[y][x] = E;
					// for better visualization
					depth.at<float>(y, x) = (float)offset;
				}
			}
		}
	}

	return depth;
}

/**
 * \brief adaptive support weight with fixed window size
 * \param leftImg 
 * \param rightImg 
 * \param dispType 
 * \param minDisparity 
 * \param numDisparity 
 * \return 
 */
cv::Mat stereomatch_1::computeAdaptiveWeight_direct8(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType, int winSize, int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity;
	int min_offset = minDisparity;
	int kernel_size = winSize; // window size
	double k = 3, gamma_c = 30, gamma_g = winSize * 2 / 3; // ASW parameters

	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value

	//�ûҶ�ֵ��ʾ��ɫ���þ��ԻҶȲ�ֵ��ʾ���ص����ɫ����
	Mat left;
	cvtColor(leftImg, left, COLOR_BGR2GRAY);
	Mat right;
	cvtColor(rightImg, right, COLOR_BGR2GRAY);

	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max());
		min_asw.push_back(tmp);
	}

	vector<Mat> weightAllDirectLeft;
	vector<Mat> weightAllDirectRight;

	for (int j = -kernel_size / 2; j < kernel_size / 2 + 1; j++)
	{
		for (int i = -kernel_size / 2; i < kernel_size / 2 + 1; i++)
		{
			if (i == 0 && j == 0)
			{
				continue;
			}
			if (i == j || i == 0 || j == 0 || (i + j) == kernel_size - 1)
			{
				Mat weightOneDirectLeft(height, width, CV_32FC1);
				Mat weightOneDirectRight(height, width, CV_32FC1);
				double delta_g = sqrt(i * i + j * j);		//�ռ����
				for (int y = 0; y < height; y++)
				{
					for (int x = 0; x < width; x++)
					{
						int neightbor_x = min(max(0, x + i), width - 1);
						int neightbor_y = min(max(0, y + j), height - 1);

						double delta_c1 = fabs(left.at<uchar>(neightbor_y, neightbor_x) - left.at<uchar>(y, x));		//��ɫ���룬���ԻҶȲ�ֵ
						double delta_c2 = fabs(right.at<uchar>(neightbor_y, neightbor_x) - right.at<uchar>(y, x));		//��ɫ����

						weightOneDirectLeft.at<float>(y, x) = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
						weightOneDirectRight.at<float>(y, x) = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
					}
				}
				weightAllDirectLeft.push_back(weightOneDirectLeft);
				weightAllDirectRight.push_back(weightOneDirectRight);
			}
		}
	}

	for (int offset = min_offset; offset <= max_offset; offset++)
	{
		// calculate each pixel's ASW value
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double E = 0;

				if (dispType == DISPARITY_LEFT)
				{
					double numerator = 0;
					double denominator = 0;

					int count = 0;

					for (int j = -kernel_size / 2; j < kernel_size / 2 + 1; j++)
					{
						for (int i = -kernel_size / 2; i < kernel_size / 2 + 1; i++)
						{
							if (i == 0 && j == 0 || count >= 4 * (kernel_size-1))
							{
								continue;
							}
							if (i == j || i == 0 || j == 0 || (i + j) == kernel_size - 1)
							{
								int neightbor_x = min(max(0, x + i), width - 1);
								int neightbor_y = min(max(0, y + j), height - 1);

								numerator += weightAllDirectLeft[count].at<float>(y, x)
									* weightAllDirectRight[count].at<float>(y, max(0, x - offset))
									* fabs(left.at<uchar>(neightbor_y, neightbor_x) - right.at<uchar>(neightbor_y, max(0, neightbor_x - offset)));
								denominator += weightAllDirectLeft[count].at<float>(y, x)
									* weightAllDirectRight[count].at<float>(y, max(0, x - offset));

								count++;
							}
						}
					}

					E = numerator / denominator;
				}
				else
				{
					double numerator = 0;
					double denominator = 0;

					int count = 0;

					for (int j = -kernel_size / 2; j < kernel_size / 2 + 1; j++)
					{
						for (int i = -kernel_size / 2; i < kernel_size / 2 + 1; i++)
						{
							if (i == 0 && j == 0 || count >= 4 * (kernel_size - 1))
							{
								continue;
							}
							if (i == j || i == 0 || j == 0 || (i + j) == kernel_size - 1)
							{

								int neightbor_x = min(max(0, x + i), width - 1);
								int neightbor_y = min(max(0, y + j), height - 1);

								numerator += weightAllDirectLeft[i].at<float>(y, min(x + offset, width - 1))
									* weightAllDirectRight[i].at<float>(y, x)
									* fabs(right.at<uchar>(neightbor_y, neightbor_x) - left.at<uchar>(neightbor_y, min(neightbor_x + offset, width - 1)));
								denominator += weightAllDirectLeft[i].at<float>(y, min(x + offset, width - 1))
									* weightAllDirectRight[i].at<float>(y, x);

								count++;
							}
						}
					}

					E = numerator / denominator;
				}

				// smaller ASW found
				if (E < min_asw[y][x])
				{
					min_asw[y][x] = E;
					// for better visualization
					depth.at<float>(y, x) = (float)offset;
				}
			}
		}
	}

	return depth;


}

float stereomatch_1::getColorDist(cv::Vec3b pointA, cv::Vec3b pointB)
{
	return fabs(pointA[0] - pointB[0]) 
	+ fabs(pointA[1] - pointB[1]) 
	+ fabs(pointA[2] - pointB[2]);
}

void stereomatch_1::getWinGeoDist(cv::Mat originImg, cv::Mat& winDistImg, int winSize, int iterTime)
{
	if(originImg.cols != winSize + 2)
	{
		return;
	}
	if(winDistImg.cols != winSize + 2)
	{
		return;
	}

	for (int iterCount = 0; iterCount < iterTime; iterCount++)
	{
		if (iterCount / 2 == 1)
		{
			for (int winR = 1; winR <= winSize; winR++)
			{
				for (int winC = 1; winC <= winSize; winC++)
				{
					float curL = winDistImg.at<float>(winR, winC - 1) 
						+ getColorDist(originImg.at<Vec3b>(winR, winC - 1), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curL);
			
					float curUL = winDistImg.at<float>(winR-1, winC - 1)
						+ getColorDist(originImg.at<Vec3b>(winR-1, winC - 1), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curUL);

					float curU = winDistImg.at<float>(winR-1, winC)
						+ getColorDist(originImg.at<Vec3b>(winR-1, winC), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curU);

					float curUR = winDistImg.at<float>(winR-1, winC+1)
						+ getColorDist(originImg.at<Vec3b>(winR-1, winC + 1), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curUR);
				}
			}
		}
		else if(iterCount / 2 == 0)
		{
			for (int winR = winSize; winR > 0; winR--)
			{
				for (int winC = winSize; winC > 0; winC--)
				{
					float curR = winDistImg.at<float>(winR, winC + 1)
						+ getColorDist(originImg.at<Vec3b>(winR, winC + 1), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curR);

					float curBR = winDistImg.at<float>(winR + 1, winC + 1)
						+ getColorDist(originImg.at<Vec3b>(winR + 1, winC + 1), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curBR);

					float curB = winDistImg.at<float>(winR + 1, winC)
						+ getColorDist(originImg.at<Vec3b>(winR + 1, winC), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curB);

					float curBL = winDistImg.at<float>(winR + 1, winC - 1)
						+ getColorDist(originImg.at<Vec3b>(winR + 1, winC - 1), originImg.at<Vec3b>(winR, winC));
					winDistImg.at<float>(winR, winC) = min(winDistImg.at<float>(winR, winC), curBL);
				}
			}
		}
	}
}

void stereomatch_1::getGeodesicDist(cv::Mat originImg, map<Point, Mat, MY_COMP_Point2i>& weightGeoDist, int winSize, int iterTime)
{
	if(winSize % 2 == 0)
	{
		return;
	}

	int width = originImg.cols;
	int height = originImg.rows;
	int halfWinSize = winSize / 2;

	Mat externOrigin;
	copyMakeBorder(originImg, externOrigin, halfWinSize+1, halfWinSize+1, halfWinSize+1, halfWinSize+1, BORDER_REFLECT);

	if(!weightGeoDist.empty())
	{
		weightGeoDist.clear();
	}
	for(int i = halfWinSize+1; i < (width + halfWinSize + 1); i++)
	{
		for(int j = halfWinSize+1; j < (height + halfWinSize + 1); j++)
		{
			Mat winImg = externOrigin(Range(j - halfWinSize - 1, j + halfWinSize + 2),
				Range(i - halfWinSize - 1, i + halfWinSize + 2));
			Mat winDistImg(winSize + 2, winSize + 2, CV_32FC1, numeric_limits<float>::max());
			winDistImg.at<float>(halfWinSize + 1, halfWinSize + 1) = 0;

			getWinGeoDist(winImg, winDistImg, winSize, iterTime);
			weightGeoDist[Point(i - halfWinSize-1, j - halfWinSize-1)] = 
				winDistImg(Rect(1, 1, winSize, winSize)).clone();
		}
	}
}

/**
 * \brief geodesic support weight algorithm
 * \param leftImg 
 * \param rightImg 
 * \param dispType 
 * \param winSize 
 * \param minDisparity 
 * \param numDisparity 
 * \return 
 */
cv::Mat stereomatch_1::computeAdaptiveWeight_geodesic(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType, int winSize,
	int minDisparity, int numDisparity)

{
	if(winSize % 2 == 0)
	{
		return Mat();
	}

	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity;
	int min_offset = minDisparity;
	int kernel_size = winSize; // window size
	int halfKernelSize = winSize / 2;
	double k = 3, gamma_c = 30, gamma_g = winSize * 2 / 3; // ASW parameters

	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value

	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max());
		min_asw.push_back(tmp);
	}

	map<Point, Mat, MY_COMP_Point2i> weightAllLeft, weightAllRight;

	getGeodesicDist(leftImg, weightAllLeft, winSize, 3);
	getGeodesicDist(rightImg, weightAllRight, winSize, 3);

	for (int offset = min_offset; offset <= max_offset; offset++)
	{
		// calculate each pixel's ASW value
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double E = 0;

				if (dispType == DISPARITY_LEFT)
				{
					double numerator = 0;
					double denominator = 0;

					for (int j = 0; j < kernel_size; j++)
					{
						for (int i = 0; i < kernel_size; i++)
						{
							int neightbor_x = min(max(0, x - kernel_size / 2 + i), width - 1);
							int neightbor_y = min(max(0, y - kernel_size / 2 + j), height - 1);

							numerator += weightAllLeft[Point(x, y)].at<float>(j, i)
								* weightAllRight[Point(max(0, x - offset), y)].at<float>(j, i)
								* getColorDist(leftImg.at<Vec3b>(neightbor_y, neightbor_x), rightImg.at<Vec3b>(neightbor_y, max(0, neightbor_x - offset)));
							denominator += weightAllLeft[Point(x, y)].at<float>(j, i)
								* weightAllRight[Point(max(0, x - offset), y)].at<float>(j, i);
						}
					}

					E = numerator / denominator;
				}
				else
				{
					double numerator = 0;
					double denominator = 0;

					for (int j = -kernel_size / 2; j < kernel_size / 2 + 1; j++)
					{
						for (int i = -kernel_size / 2; i < kernel_size / 2 + 1; i++)
						{
								int neightbor_x = min(max(0, x + i), width - 1);
								int neightbor_y = min(max(0, y + j), height - 1);

								numerator += weightAllLeft[Point(min(x + offset, width - 1), y)].at<float>(j + kernel_size / 2, i + kernel_size / 2)
									* weightAllRight[Point(x, y)].at<float>(j + kernel_size / 2, i + kernel_size / 2)
									* getColorDist(rightImg.at<Vec3b>(neightbor_y, neightbor_x),
										leftImg.at<Vec3b>(neightbor_y, min(width - 1, neightbor_x + offset)));
								denominator += weightAllLeft[Point(min(x + offset, width - 1), y)].at<float>(j + kernel_size / 2, i + kernel_size / 2)
									* weightAllRight[Point(x, y)].at<float>(j + kernel_size / 2, i + kernel_size / 2);
						}
					}

					E = numerator / denominator;
				}

				// smaller ASW found
				if (E < min_asw[y][x])
				{
					min_asw[y][x] = E;
					// for better visualization
					depth.at<float>(y, x) = (float)offset;
				}
			}
		}
	}

	return depth;
}

void stereomatch_1::createBilGrid(cv::Mat image, std::map<cv::Vec3i, std::pair<double, int>, MY_COMP_vec3i>& bilGrid,
	double sampleRateS, double sampleRateR)
{
	if(sampleRateS <= 0)
	{
		sampleRateS = 16;
	}

	if(sampleRateR <= 0)
	{
		sampleRateR = 0.07;
	}

	if(image.type() != CV_32FC1)
	{
		image.convertTo(image, CV_32FC1);
	}
	normalize(image, image, 0, 1, cv::NORM_MINMAX);

	int width = image.cols;
	int height = image.rows;

	double maxValue = 0.0;
	minMaxLoc(image, NULL, &maxValue, NULL, NULL);

	int gridSize_range = cvRound(maxValue / sampleRateR);
	int gridSize_width = cvRound((width - 1) / sampleRateS);
	int gridSize_height = cvRound((height - 1) / sampleRateS);

	//grid initialization
	for (int x = -2; x <= gridSize_width + 2; x++)
	{
		for (int y = -2; y <= gridSize_height + 2; y++)
		{
			for (int z = -2; z <= gridSize_range + 2; z++)
			{
				bilGrid[Vec3i(x, y, z)] = make_pair(0.0, 0);
			}
		}
	}

	//grid filling
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			Vec3i newKey;
			newKey[0] = cvRound(i / sampleRateS);
			newKey[1] = cvRound(j / sampleRateS);
			newKey[2] = cvRound(image.at<float>(j, i) / sampleRateR);

			bilGrid[newKey] = make_pair(bilGrid[newKey].first + image.at<float>(j, i),
				bilGrid[newKey].second + 1);
		}
	}

	//processing to obtain a new bilateral grid
	//gaussian convolution along z axis
	for (int x = 0; x <= gridSize_width; x++)
	{
		for (int y = 0; y <= gridSize_height; y++)
		{
			for (int z = 0; z <= gridSize_range; z++)
			{
				bilGrid[Vec3i(x, y, z)] = make_pair(
					bilGrid[Vec3i(x, y, z-2)].first 
					+ 4 * bilGrid[Vec3i(x, y, z - 1)].first + 6 * bilGrid[Vec3i(x, y, z)].first
					+ 4 * bilGrid[Vec3i(x, y, z + 1)].first + bilGrid[Vec3i(x, y, z+2)].first,
					bilGrid[Vec3i(x, y, z - 2)].second 
					+ 4 * bilGrid[Vec3i(x, y, z - 1)].second + 6 * bilGrid[Vec3i(x, y, z)].second
					+ 4 * bilGrid[Vec3i(x, y, z + 1)].second + bilGrid[Vec3i(x, y, z + 2)].second
				);
			}
		}
	}
	//gaussian convolution along y axis
	for (int z = 0; z <= gridSize_range; z++)
	{
		for (int x = 0; x <= gridSize_width; x++)
		{
			for (int y = 0; y <= gridSize_height; y++)
			{
				bilGrid[Vec3i(x, y, z)] = make_pair(
					bilGrid[Vec3i(x, y - 2, z)].first
					+ 4 * bilGrid[Vec3i(x, y - 1, z)].first + 6 * bilGrid[Vec3i(x, y, z)].first
					+ 4 * bilGrid[Vec3i(x, y + 1, z)].first + bilGrid[Vec3i(x, y + 2, z)].first,
					bilGrid[Vec3i(x, y, z - 2)].second
					+ 4 * bilGrid[Vec3i(x, y - 1, z)].second + 6 * bilGrid[Vec3i(x, y, z)].second
					+ 4 * bilGrid[Vec3i(x, y + 1, z)].second + bilGrid[Vec3i(x, y + 2, z)].second
				);
			}
		}
	}
	//gaussian convolution along x axis
	for (int y = 0; y <= gridSize_height; y++)
	{
		for (int z = 0; z <= gridSize_range; z++)
		{
			for (int x = 0; x <= gridSize_width; x++)
			{
				bilGrid[Vec3i(x, y, z)] = make_pair(
					bilGrid[Vec3i(x - 2, y, z)].first
					+ 4 * bilGrid[Vec3i(x - 1, y, z)].first + 6 * bilGrid[Vec3i(x, y, z)].first
					+ 4 * bilGrid[Vec3i(x + 1, y, z)].first + bilGrid[Vec3i(x + 2, y, z)].first,
					bilGrid[Vec3i(x - 2, y, z)].second
					+ 4 * bilGrid[Vec3i(x - 1, y, z)].second + 6 * bilGrid[Vec3i(x, y, z)].second
					+ 4 * bilGrid[Vec3i(x + 1, y, z)].second + bilGrid[Vec3i(x + 2, y, z)].second
				);
			}
		}
	}
}

void stereomatch_1::createBilGrid(cv::Mat imageL, cv::Mat imageR, std::map<cv::Vec4i, std::pair<double, int>, MY_COMP_vec4i>& bilGrid,
	int disparity, DisparityType dispType, double sampleRateS, double sampleRateR)
{
	if (sampleRateS <= 0)
	{
		sampleRateS = 16;
	}

	if (sampleRateR <= 0)
	{
		sampleRateR = 0.07;
	}

	if (imageL.type() != CV_32FC1)
	{
		imageL.convertTo(imageL, CV_32FC1);
	}
	if (imageR.type() != CV_32FC1)
	{
		imageR.convertTo(imageR, CV_32FC1);
	}
	if(imageL.size != imageR.size)
	{
		return;
	}

	int width = imageL.cols;
	int height = imageL.rows;

	//double maxValueL = 0.0;
	//minMaxLoc(imageL, NULL, &maxValueL, NULL, NULL);
	//double maxValueR = 0.0;
	//minMaxLoc(imageR, NULL, &maxValueR, NULL, NULL);

	double maxValueL = 255.0;
	double maxValueR = 255.0;
	int gridSize_rangeL = cvRound(maxValueL / sampleRateR);
	int gridSize_rangeR = cvRound(maxValueR / sampleRateR);
	int gridSize_width = cvRound((width - 1) / sampleRateS);
	int gridSize_height = cvRound((height - 1) / sampleRateS);

	//grid initialization
	for (int x = -2; x <= gridSize_width + 2; x++)
	{
		for (int y = -2; y <= gridSize_height + 2; y++)
		{
			for (int z = -2; z <= gridSize_rangeL + 2; z++)
			{
				for (int w = -2; w <= gridSize_rangeR + 2; w++)
				{
					bilGrid[Vec4i(x, y, z, w)] = make_pair(0.0, 0);
				}
			}
		}
	}

	//grid filling
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				Vec4i newKey;
				newKey[0] = cvRound(i / sampleRateS);
				newKey[1] = cvRound(j / sampleRateS);
				newKey[2] = cvRound(imageL.at<float>(j, i) / sampleRateR);
				newKey[3] = cvRound(imageR.at<float>(j, max(0, i - disparity)) / sampleRateR);

				bilGrid[newKey] = make_pair(bilGrid[newKey].first
					+ fabs(imageL.at<float>(j, i) - imageR.at<float>(j, max(0, i - disparity))),
					bilGrid[newKey].second + 1);
			}
		}
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				Vec4i newKey;
				newKey[0] = cvRound(i / sampleRateS);
				newKey[1] = cvRound(j / sampleRateS);
				newKey[2] = cvRound(imageL.at<float>(j, min(i+disparity, width)) / sampleRateR);
				newKey[3] = cvRound(imageR.at<float>(j, i) / sampleRateR);

				bilGrid[newKey] = make_pair(bilGrid[newKey].first
					+ fabs(imageL.at<float>(j, min(i+disparity, width)) - imageR.at<float>(j, i)),
					bilGrid[newKey].second + 1);
			}
		}
	}

	//processing to obtain a new bilateral grid
	//gaussian convolution along w axis
	for (int x = 0; x <= gridSize_width; x++)
	{
		for (int y = 0; y <= gridSize_height; y++)
		{
			for (int z = 0; z <= gridSize_rangeL; z++)
			{
				for (int w = 0; w <= gridSize_rangeR; w++)
				{
					bilGrid[Vec4i(x, y, z, w)] = make_pair(
						bilGrid[Vec4i(x, y, z, w- 2)].first
						+ 4 * bilGrid[Vec4i(x, y, z, w - 1)].first + 6 * bilGrid[Vec4i(x, y, z, w)].first
						+ 4 * bilGrid[Vec4i(x, y, z, w + 1)].first + bilGrid[Vec4i(x, y, z, w + 2)].first,
						bilGrid[Vec4i(x, y, z, w - 2)].second
						+ 4 * bilGrid[Vec4i(x, y, z, w - 1)].second + 6 * bilGrid[Vec4i(x, y, z, w)].second
						+ 4 * bilGrid[Vec4i(x, y, z, w + 1)].second + bilGrid[Vec4i(x, y, z, w + 2)].second
					);
				}
			}
		}
	}
	//gaussian convolution along w axis
	for (int x = 0; x <= gridSize_width; x++)
	{
		for (int y = 0; y <= gridSize_height; y++)
		{
			for (int w = 0; w <= gridSize_rangeR; w++)
			{
				for (int z = 0; z <= gridSize_rangeL; z++)
				{
					bilGrid[Vec4i(x, y, z, w)] = make_pair(
						bilGrid[Vec4i(x, y, z - 2, w)].first
						+ 4 * bilGrid[Vec4i(x, y, z - 1, w)].first + 6 * bilGrid[Vec4i(x, y, z, w)].first
						+ 4 * bilGrid[Vec4i(x, y, z + 1, w)].first + bilGrid[Vec4i(x, y, z + 2, w)].first,
						bilGrid[Vec4i(x, y, z - 2, w)].second
						+ 4 * bilGrid[Vec4i(x, y, z - 1, w)].second + 6 * bilGrid[Vec4i(x, y, z, w)].second
						+ 4 * bilGrid[Vec4i(x, y, z + 1, w)].second + bilGrid[Vec4i(x, y, z + 2, w)].second
					);
				}
			}
		}
	}

	//gaussian convolution along y axis
	for (int w = 0; w <= gridSize_rangeR; w++)
	{
		for (int z = 0; z <= gridSize_rangeL; z++)
		{
			for (int x = 0; x <= gridSize_width; x++)
			{
				for (int y = 0; y <= gridSize_height; y++)
				{
					bilGrid[Vec4i(x, y, z)] = make_pair(
						bilGrid[Vec4i(x, y - 2, z, w)].first
						+ 4 * bilGrid[Vec4i(x, y - 1, z, w)].first + 6 * bilGrid[Vec4i(x, y, z, w)].first
						+ 4 * bilGrid[Vec4i(x, y + 1, z, w)].first + bilGrid[Vec4i(x, y + 2, z, w)].first,
						bilGrid[Vec4i(x, y, z - 2, w)].second
						+ 4 * bilGrid[Vec4i(x, y - 1, z, w)].second + 6 * bilGrid[Vec4i(x, y, z, w)].second
						+ 4 * bilGrid[Vec4i(x, y + 1, z, w)].second + bilGrid[Vec4i(x, y + 2, z, w)].second
					);
				}
			}
		}
	}
	//gaussian convolution along x axis
	for (int y = 0; y <= gridSize_height; y++)
	{
		for (int z = 0; z <= gridSize_rangeL; z++)
		{
			for (int w = 0; w <= gridSize_rangeR; w++)
			{
				for (int x = 0; x <= gridSize_width; x++)
				{
					bilGrid[Vec4i(x, y, z)] = make_pair(
						bilGrid[Vec4i(x - 2, y, z, w)].first
						+ 4 * bilGrid[Vec4i(x - 1, y, z, w)].first + 6 * bilGrid[Vec4i(x, y, z, w)].first
						+ 4 * bilGrid[Vec4i(x + 1, y, z, w)].first + bilGrid[Vec4i(x + 2, y, z, w)].first,
						bilGrid[Vec4i(x - 2, y, z, w)].second
						+ 4 * bilGrid[Vec4i(x - 1, y, z, w)].second + 6 * bilGrid[Vec4i(x, y, z, w)].second
						+ 4 * bilGrid[Vec4i(x + 1, y, z, w)].second + bilGrid[Vec4i(x + 2, y, z, w)].second
					);
				}
			}
		}
	}
}

void stereomatch_1::createBilGrid(cv::Mat imageL, cv::Mat imageR,
	std::map<int, std::map<int, std::map<int, std::map<int, std::pair<double, int> > > > >& bilGrid, int disparity,
	DisparityType dispType, double sampleRateS, double sampleRateR)
{
	if (sampleRateS <= 0)
	{
		sampleRateS = 16;
	}

	if (sampleRateR <= 0)
	{
		sampleRateR = 0.07;
	}

	if (imageL.type() != CV_32FC1)
	{
		imageL.convertTo(imageL, CV_32FC1);
	}
	if (imageR.type() != CV_32FC1)
	{
		imageR.convertTo(imageR, CV_32FC1);
	}
	if (imageL.size != imageR.size)
	{
		return;
	}

	int width = imageL.cols;
	int height = imageL.rows;

	//double maxValueL = 0.0;
	//minMaxLoc(imageL, NULL, &maxValueL, NULL, NULL);
	//double maxValueR = 0.0;
	//minMaxLoc(imageR, NULL, &maxValueR, NULL, NULL);

	double maxValueL = 255.0;
	double maxValueR = 255.0;
	int gridSize_rangeL = cvRound(maxValueL / sampleRateR);
	int gridSize_rangeR = cvRound(maxValueR / sampleRateR);
	int gridSize_width = cvRound((width - 1) / sampleRateS);
	int gridSize_height = cvRound((height - 1) / sampleRateS);

	//grid initialization
	for (int x = 0; x <= gridSize_width; x++)
	{
		std::map<int, std::map<int, std::map<int, std::pair<double, int> > > > map_x;
		for (int y = 0; y <= gridSize_height; y++)
		{
			std::map<int, std::map<int, std::pair<double, int> > > map_xy;
			for (int z = 0; z <= gridSize_rangeL; z++)
			{
				std::map<int, std::pair<double, int> > map_xyz;
				for (int w = 0; w <= gridSize_rangeR; w++)
				{
					map_xyz[w] = make_pair(0.0, 0);
				}
				map_xy[z] = map_xyz;
			}
			map_x[y] = map_xy;
		}
		bilGrid[x] = map_x;
	}

	//grid filling
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				Vec4i newKey;
				newKey[0] = cvRound(i / sampleRateS);
				newKey[1] = cvRound(j / sampleRateS);
				newKey[2] = cvRound(imageL.at<float>(j, i) / sampleRateR);
				newKey[3] = cvRound(imageR.at<float>(j, max(0, i - disparity)) / sampleRateR);

				bilGrid[newKey[0]][newKey[1]][newKey[2]][newKey[3]] = make_pair(
					bilGrid[newKey[0]][newKey[1]][newKey[2]][newKey[3]].first
					+ fabs(imageL.at<float>(j, i) - imageR.at<float>(j, max(0, i - disparity))),
					bilGrid[newKey[0]][newKey[1]][newKey[2]][newKey[3]].second + 1);
			}
		}
	}
	else if (dispType == DISPARITY_RIGHT)
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				Vec4i newKey;
				newKey[0] = cvRound(i / sampleRateS);
				newKey[1] = cvRound(j / sampleRateS);
				newKey[2] = cvRound(imageL.at<float>(j, min(i + disparity, width)) / sampleRateR);
				newKey[3] = cvRound(imageR.at<float>(j, i) / sampleRateR);

				bilGrid[newKey[0]][newKey[1]][newKey[2]][newKey[3]] = make_pair(
					bilGrid[newKey[0]][newKey[1]][newKey[2]][newKey[3]].first
					+ fabs(imageL.at<float>(j, min(i + disparity, width)) - imageR.at<float>(j, i)),
					bilGrid[newKey[0]][newKey[1]][newKey[2]][newKey[3]].second + 1);
			}
		}
	}

	//processing to obtain a new bilateral grid
	//gaussian convolution along w axis
	for (int x = 0; x <= gridSize_width; x++)
	{
		for (int y = 0; y <= gridSize_height; y++)
		{
			for (int z = 0; z <= gridSize_rangeL; z++)
			{
				for (int w = 0; w <= gridSize_rangeR; w++)
				{
					if(w == 0)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.6 * bilGrid[x][y][z][w].first
							+ 0.3 * bilGrid[x][y][z][w + 1].first + 0.1 * bilGrid[x][y][z][w + 2].first,
							0.6 * bilGrid[x][y][z][w].second
							+ 0.3 * bilGrid[x][y][z][w + 1].second + 0.1 * bilGrid[x][y][z][w + 2].second
						);
					}
					else if(w == 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.2 * bilGrid[x][y][z][w - 1].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x][y][z][w + 1].first + 0.1 * bilGrid[x][y][z][w + 2].first,
							0.2 * bilGrid[x][y][z][w - 1].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x][y][z][w + 1].second + 0.1 * bilGrid[x][y][z][w + 2].second
						);
					}
					else if(w == gridSize_rangeR-1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x][y][z][w - 2].first
							+ 0.2 * bilGrid[x][y][z][w - 1].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x][y][z][w + 1].first,
							0.1 * bilGrid[x][y][z][w - 2].second
							+ 0.2 * bilGrid[x][y][z][w - 1].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x][y][z][w + 1].second
						);
					}
					else if( w == gridSize_rangeR)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x][y][z][w - 2].first
							+ 0.3 * bilGrid[x][y][z][w - 1].first + 0.6 * bilGrid[x][y][z][w].first,
							0.1 * bilGrid[x][y][z][w - 2].second
							+ 0.3 * bilGrid[x][y][z][w - 1].second + 0.6 * bilGrid[x][y][z][w].second
						);
					}
					else
					{
						bilGrid[x][y][z][w] = make_pair(
							0.0625 * bilGrid[x][y][z][w - 2].first
							+ 0.25 * bilGrid[x][y][z][w - 1].first + 0.375 * bilGrid[x][y][z][w].first
							+ 0.25 * bilGrid[x][y][z][w + 1].first + 0.0625 * bilGrid[x][y][z][w + 2].first,
							0.0625 * bilGrid[x][y][z][w - 2].second
							+ 0.25 * bilGrid[x][y][z][w - 1].second + 0.375 * bilGrid[x][y][z][w].second
							+ 0.25 * bilGrid[x][y][z][w + 1].second + 0.0625 * bilGrid[x][y][z][w + 2].second
						);
					}
				}
			}
		}
	}
	//gaussian convolution along w axis
	for (int x = 0; x <= gridSize_width; x++)
	{
		for (int y = 0; y <= gridSize_height; y++)
		{
			for (int w = 0; w <= gridSize_rangeR; w++)
			{
				for (int z = 0; z <= gridSize_rangeL; z++)
				{
					if (z == 0)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.6 * bilGrid[x][y][z][w].first
							+ 0.3 * bilGrid[x][y][z + 1][w].first + 0.1 * bilGrid[x][y][z + 2][w].first,
							0.6 * bilGrid[x][y][z][w].second
							+ 0.3 * bilGrid[x][y][z + 1][w].second + 0.1 * bilGrid[x][y][z + 2][w].second
						);
					}
					else if (z == 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.2 * bilGrid[x][y][z - 1][w].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x][y][z + 1][w].first + 0.1 * bilGrid[x][y][z + 2][w].first,
							0.2 * bilGrid[x][y][z - 1][w].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x][y][z + 1][w].second + 0.1 * bilGrid[x][y][z + 2][w].second
						);
					}
					else if (z == gridSize_rangeL - 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x][y][z - 2][w].first
							+ 0.2 * bilGrid[x][y][z - 1][w].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x][y][z + 1][w].first,
							0.1 * bilGrid[x][y][z - 2][w].second
							+ 0.2 * bilGrid[x][y][z - 1][w].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x][y][z + 1][w].second
						);
					}
					else if (z == gridSize_rangeL)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x][y][z - 2][w].first
							+ 0.3 * bilGrid[x][y][z - 1][w].first + 0.6 * bilGrid[x][y][z][w].first,
							0.1 * bilGrid[x][y][z - 2][w].second
							+ 0.3 * bilGrid[x][y][z - 1][w].second + 0.6 * bilGrid[x][y][z][w].second
						);
					}
					else
					{
						bilGrid[x][y][z][w] = make_pair(
							0.0625 * bilGrid[x][y][z - 2][w].first
							+ 0.25 * bilGrid[x][y][z - 1][w].first + 0.375 * bilGrid[x][y][z][w].first
							+ 0.25 * bilGrid[x][y][z + 1][w].first + 0.0625 * bilGrid[x][y][z + 2][w].first,
							0.0625 * bilGrid[x][y][z - 2][w].second
							+ 0.25 * bilGrid[x][y][z - 1][w].second + 0.375 * bilGrid[x][y][z][w].second
							+ 0.25 * bilGrid[x][y][z + 1][w].second + 0.0625 * bilGrid[x][y][z + 2][w].second
						);
					}
				}
			}
		}
	}

	//gaussian convolution along y axis
	for (int w = 0; w <= gridSize_rangeR; w++)
	{
		for (int z = 0; z <= gridSize_rangeL; z++)
		{
			for (int x = 0; x <= gridSize_width; x++)
			{
				for (int y = 0; y <= gridSize_height; y++)
				{
					if (y == 0)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.6 * bilGrid[x][y][z][w].first
							+ 0.3 * bilGrid[x][y + 1][z][w].first + 0.1 * bilGrid[x][y + 2][z][w].first,
							0.6 * bilGrid[x][y][z][w].second
							+ 0.3 * bilGrid[x][y + 1][z][w].second + 0.1 * bilGrid[x][y + 2][z][w].second
						);
					}
					else if (y == 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.2 * bilGrid[x][y - 1][z][w].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x][y + 1][z][w].first + 0.1 * bilGrid[x][y + 2][z][w].first,
							0.2 * bilGrid[x][y - 1][z][w].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x][y + 1][z][w].second + 0.1 * bilGrid[x][y + 2][z][w].second
						);
					}
					else if (y == gridSize_height - 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x][y - 2][z][w].first
							+ 0.2 * bilGrid[x][y - 1][z][w].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x][y + 1][z][w].first,
							0.1 * bilGrid[x][y - 2][z][w].second
							+ 0.2 * bilGrid[x][y - 1][z][w].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x][y + 1][z][w].second
						);
					}
					else if (y == gridSize_height)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x][y - 2][z][w].first
							+ 0.3 * bilGrid[x][y - 1][z][w].first + 0.6 * bilGrid[x][y][z][w].first,
							0.1 * bilGrid[x][y - 2][z][w].second
							+ 0.3 * bilGrid[x][y - 1][z][w].second + 0.6 * bilGrid[x][y][z][w].second
						);
					}
					else
					{
						bilGrid[x][y][z][w] = make_pair(
							0.0625 * bilGrid[x][y - 2][z][w].first
							+ 0.25 * bilGrid[x][y - 1][z][w].first + 0.375 * bilGrid[x][y][z][w].first
							+ 0.25 * bilGrid[x][y + 1][z][w].first + 0.0625 * bilGrid[x][y + 2][z][w].first,
							0.0625 * bilGrid[x][y - 2][z][w].second
							+ 0.25 * bilGrid[x][y - 1][z][w].second + 0.375 * bilGrid[x][y][z][w].second
							+ 0.25 * bilGrid[x][y + 1][z][w].second + 0.0625 * bilGrid[x][y + 2][z][w].second
						);
					}
				}
			}
		}
	}
	//gaussian convolution along x axis
	for (int y = 0; y <= gridSize_height; y++)
	{
		for (int z = 0; z <= gridSize_rangeL; z++)
		{
			for (int w = 0; w <= gridSize_rangeR; w++)
			{
				for (int x = 0; x <= gridSize_width; x++)
				{
					if (x == 0)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.6 * bilGrid[x][y][z][w].first
							+ 0.3 * bilGrid[x + 1][y][z][w].first + 0.1 * bilGrid[x + 2][y][z][w].first,
							0.6 * bilGrid[x][y][z][w].second
							+ 0.3 * bilGrid[x + 1][y][z][w].second + 0.1 * bilGrid[x + 2][y][z][w].second
						);
					}
					else if (x == 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.2 * bilGrid[x - 1][y][z][w].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x + 1][y][z][w].first + 0.1 * bilGrid[x + 2][y][z][w].first,
							0.2 * bilGrid[x - 1][y][z][w].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x + 1][y][z][w].second + 0.1 * bilGrid[x + 2][y][z][w].second
						);
					}
					else if (x == gridSize_width - 1)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x - 2][y][z][w].first
							+ 0.2 * bilGrid[x - 1][y][z][w].first + 0.5 * bilGrid[x][y][z][w].first
							+ 0.2 * bilGrid[x + 1][y][z][w].first,
							0.1 * bilGrid[x - 2][y][z][w].second
							+ 0.2 * bilGrid[x - 1][y][z][w].second + 0.5 * bilGrid[x][y][z][w].second
							+ 0.2 * bilGrid[x + 1][y][z][w].second
						);
					}
					else if (x == gridSize_width)
					{
						bilGrid[x][y][z][w] = make_pair(
							0.1 * bilGrid[x - 2][y][z][w].first
							+ 0.3 * bilGrid[x - 1][y][z][w].first + 0.6 * bilGrid[x][y][z][w].first,
							0.1 * bilGrid[x - 2][y][z][w].second
							+ 0.3 * bilGrid[x - 1][y][z][w].second + 0.6 * bilGrid[x][y][z][w].second
						);
					}
					else
					{
						bilGrid[x][y][z][w] = make_pair(
							0.0625 * bilGrid[x - 2][y][z][w].first
							+ 0.25 * bilGrid[x - 1][y][z][w].first + 0.375 * bilGrid[x][y][z][w].first
							+ 0.25 * bilGrid[x + 1][y][z][w].first + 0.0625 * bilGrid[x + 2][y][z][w].first,
							0.0625 * bilGrid[x - 2][y][z][w].second
							+ 0.25 * bilGrid[x - 1][y][z][w].second + 0.375 * bilGrid[x][y][z][w].second
							+ 0.25 * bilGrid[x + 1][y][z][w].second + 0.0625 * bilGrid[x + 2][y][z][w].second
						);
					}
				}
			}
		}
	}

}

/**
 * \brief 
 * \param axis_diff_xyz	:	[0]x+ - x	[1]y+ - y	[2]z+ - z	
 * \param neighbor_xyz_02 :	[0](x-,y+,z-)	[1](x-,y-,z-)
 *							[2](x+,y+,z-)	[3](x+,y-,z-)
 *							[4](x-,y+,z+)	[5](x-,y-,z+)
 *							[6](x+,y+,z+)	[7](x+,y-,z+)
 * \return 
 */
double stereomatch_1::trilinear_3d(std::vector<double> axis_diff_xyz, std::vector<double> neighbor_xyz_02)
{
	if (axis_diff_xyz.size() != 3 || neighbor_xyz_02.size() != 8)
		return -1;

	double a1 = axis_diff_xyz[1] * neighbor_xyz_02[0] + (1 - axis_diff_xyz[1]) * neighbor_xyz_02[1];
	double a2 = axis_diff_xyz[1] * neighbor_xyz_02[2] + (1 - axis_diff_xyz[1]) * neighbor_xyz_02[3];

	double b1 = axis_diff_xyz[1] * neighbor_xyz_02[4] + (1 - axis_diff_xyz[1]) * neighbor_xyz_02[5];
	double b2 = axis_diff_xyz[1] * neighbor_xyz_02[6] + (1 - axis_diff_xyz[1]) * neighbor_xyz_02[7];

	double c1 = axis_diff_xyz[0] * a2 + (1 - axis_diff_xyz[0]) * a1;
	double c2 = axis_diff_xyz[0] * b2 + (1 - axis_diff_xyz[0]) * b1;

	return axis_diff_xyz[2] * c2 + (1 - axis_diff_xyz[2]) * c1;
}

/**
 * \brief  
 * \param axis_diff_xyzw :		[0]x+ - x	[1]y+ - y	[2]z+ - z		[3]w+ - w
 * \param neighbor_xyzw_02 :	[0](x-,y-,z-,w-)	[1](x-,y-,z-,w+)
 *								[2](x-,y-,z+,w-)	[3](x-,y-,z+,w+)
 *								[4](x-,y+,z-,w-)	[5](x-,y+,z-,w+)
 *								[6](x-,y+,z+,w-)	[7](x-,y+,z+,w+)
 *								
 *								[8](x+,y-,z-,w-)	[9](x+,y-,z-,w+)
 *								[10](x+,y-,z+,w-)	[11](x+,y-,z+,w+)
 *								[12](x+,y+,z-,w-)	[13](x+,y+,z-,w+)
 *								[14](x+,y+,z+,w-)	[15](x+,y+,z+,w+)
 * \return 
 */
double stereomatch_1::quadrlinear_blGrid(std::vector<double> axis_diff_xyzw, std::vector<double> neighbor_xyzw_02)
{
	if(axis_diff_xyzw.size() != 4 || neighbor_xyzw_02.size() != 16)
	{
		return -1;
	}
	double a1 = neighbor_xyzw_02[0] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[1] * axis_diff_xyzw[3];
	double a2 = neighbor_xyzw_02[2] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[3] * axis_diff_xyzw[3];
	double a3 = neighbor_xyzw_02[4] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[5] * axis_diff_xyzw[3];
	double a4 = neighbor_xyzw_02[6] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[7] * axis_diff_xyzw[3];
	double a5 = neighbor_xyzw_02[8] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[9] * axis_diff_xyzw[3];
	double a6 = neighbor_xyzw_02[10] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[11] * axis_diff_xyzw[3];
	double a7 = neighbor_xyzw_02[12] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[13] * axis_diff_xyzw[3];
	double a8 = neighbor_xyzw_02[14] * (1 - axis_diff_xyzw[3]) + neighbor_xyzw_02[15] * axis_diff_xyzw[3];

	double b1 = a1 * (1 - axis_diff_xyzw[2]) + a2 * axis_diff_xyzw[2];
	double b2 = a3 * (1 - axis_diff_xyzw[2]) + a4 * axis_diff_xyzw[2];
	double b3 = a5 * (1 - axis_diff_xyzw[2]) + a6 * axis_diff_xyzw[2];
	double b4 = a7 * (1 - axis_diff_xyzw[2]) + a8 * axis_diff_xyzw[2];

	double c1 = b1 * (1 - axis_diff_xyzw[1]) + b2 * axis_diff_xyzw[1];
	double c2 = b3 * (1 - axis_diff_xyzw[1]) + b4 * axis_diff_xyzw[1];

	return c1 * (1 - axis_diff_xyzw[0]) + c2 * axis_diff_xyzw[0];
}

cv::Mat stereomatch_1::computeAdaptiveWeight_bilateralGrid(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType,
	double sampleRateS, double sampleRateR, int minDisparity, int numDisparity)
{	
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity;
	int min_offset = minDisparity;

	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value

	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max());
		min_asw.push_back(tmp);
	}

	if (leftImg.channels() == 3)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
	}
	if (rightImg.channels() == 3)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
	}

	for (int offset = min_offset; offset <= max_offset; offset++)
	{
		std::map<int, std::map<int, std::map<int, std::map<int, std::pair<double, int> > > > > bilaterGrid_stereo;
		createBilGrid(leftImg, rightImg, bilaterGrid_stereo, offset, dispType, sampleRateS, sampleRateR);
		// calculate each pixel's ASW value
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (dispType == DISPARITY_LEFT)
				{
					double x_ = x / sampleRateS;
					double y_ = y / sampleRateS;
					double color_L_ = leftImg.at<uchar>(y, x) / sampleRateR;
					double color_R_ = rightImg.at<uchar>(y, max(0, x - offset)) / sampleRateR;

					int x__ = cvCeil(x_);
					int y__ = cvCeil(y_);
					int color_L__ = cvCeil(color_L_);
					int color_R__ = cvCeil(color_R_);

					vector<double> xyzw_diff;
					double x_d = x__ - x_;
					double y_d = y__ - y_;
					double color_L_d = color_L__ - color_L_;
					double color_R_d = color_R__ - color_R_;
					xyzw_diff.push_back(x_d);
					xyzw_diff.push_back(y_d);
					xyzw_diff.push_back(color_L_d);
					xyzw_diff.push_back(color_R_d);

					vector<double> neighbor_xyzw;
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ + 1].first);

					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ + 1].first);

					vector<double> neighbor_xyzw_count;
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ + 1].second);

					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ + 1].second);

  					double curCost_ = quadrlinear_blGrid(xyzw_diff, neighbor_xyzw) / quadrlinear_blGrid(xyzw_diff, neighbor_xyzw_count);
					if (curCost_ < min_asw[y][x])
					{
						min_asw[y][x] = curCost_;
						// for better visualization
						depth.at<float>(y, x) = (float)offset;
					}
				}
				else
				{
					double x_ = x / sampleRateS;
					double y_ = y / sampleRateS;
					double color_L_ = leftImg.at<uchar>(y,min(width, x + offset)) / sampleRateR;
					double color_R_ = rightImg.at<uchar>(y, x) / sampleRateR;

					int x__ = cvCeil(x_);
					int y__ = cvCeil(y_);
					int color_L__ = cvCeil(color_L_);
					int color_R__ = cvCeil(color_R_);

					vector<double> xyzw_diff;
					double x_d = x__ - x_;
					double y_d = y__ - y_;
					double color_L_d = color_L__ - color_L_;
					double color_R_d = color_R__ - color_R_;
					xyzw_diff.push_back(x_d);
					xyzw_diff.push_back(y_d);
					xyzw_diff.push_back(color_L_d);
					xyzw_diff.push_back(color_R_d);

					vector<double> neighbor_xyzw;
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ + 1].first);

					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ + 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ - 1].first);
					neighbor_xyzw.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ + 1].first);

					vector<double> neighbor_xyzw_count;
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ - 1][color_L__ + 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ - 1][y__ + 1][color_L__ + 1][color_R__ + 1].second);

					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ - 1][color_L__ + 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ - 1][color_R__ + 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ - 1].second);
					neighbor_xyzw_count.push_back(bilaterGrid_stereo[x__ + 1][y__ + 1][color_L__ + 1][color_R__ + 1].second);


					double curCost_ = quadrlinear_blGrid(xyzw_diff, neighbor_xyzw) / quadrlinear_blGrid(xyzw_diff, neighbor_xyzw_count);
					if (curCost_ < min_asw[y][x])
					{
						min_asw[y][x] = curCost_;
						// for better visualization
						depth.at<float>(y, x) = (float)offset;
					}
				}
			}
		}
	}

	return depth;
}

/**
 * \brief	if \dispType == \DISPARITY_LEFT , then rightImg is bordered image;
 *			if \dispType == \DISPARITY_RIGHT , then leftImg is bordered image;
 * \param leftImg 
 * \param rightImg 
 * \param disparity 
 * \param dispType 
 * \param winSize 
 * \return 
 */
cv::Mat stereomatch_1::getCostSAD_d(cv::Mat leftImg, cv::Mat rightImg, int disparity, DisparityType dispType, int winSize)
{
	double start = static_cast<double>(getTickCount());

	if (leftImg.channels() != 1 || leftImg.depth() == CV_32S)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
		leftImg.convertTo(leftImg, CV_8UC1);
	}

	if (rightImg.channels() != 1 || rightImg.depth() == CV_32S)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
		rightImg.convertTo(rightImg, CV_8UC1);
	}

	if (winSize % 2 == 0)
	{
		std::cout << "winsize must be odd" << std::endl;
		return Mat();
	}

	//optimization:using integral image
	Mat differ_ranges(leftImg.size(), CV_8U, Scalar::all(0));
	Mat differ_BF;
	if (dispType == DISPARITY_LEFT)
	{
		int imgHeight = leftImg.rows;
		int imgWidth = leftImg.cols;

		if(rightImg.cols <= imgWidth)
		{
			return Mat();
		}

		absdiff(leftImg, rightImg(Rect(rightImg.cols - imgWidth - disparity, 0, imgWidth, imgHeight)), differ_ranges);
		differ_ranges.convertTo(differ_ranges, CV_32FC1);

		boxFilter(differ_ranges, differ_BF, -1, Size(winSize, winSize), Point(-1, -1), true);
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		int imgHeight = rightImg.rows;
		int imgWidth = rightImg.cols;

		if (leftImg.cols <= imgWidth)
		{
			return Mat();
		}

		absdiff(leftImg(Rect(disparity, 0, imgWidth, imgHeight)), rightImg, differ_ranges);
		differ_ranges.convertTo(differ_ranges, CV_32FC1);

		boxFilter(differ_ranges, differ_BF, -1, Size(winSize, winSize), Point(-1, -1), true);
	}

	if(differ_BF.type() != CV_32FC1)
	{
		differ_BF.convertTo(differ_BF, CV_32FC1);
	}
	return differ_BF;
}

cv::Mat stereomatch_1::computeAdaptiveWeight_BLO1(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType,
	double sampleRateR, int winSize, int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity - 1;
	int min_offset = minDisparity;


	if (leftImg.channels() == 3)
	{
		cvtColor(leftImg, leftImg, COLOR_BGR2GRAY);
	}
	if (rightImg.channels() == 3)
	{
		cvtColor(rightImg, rightImg, COLOR_BGR2GRAY);
	}

	Mat  leftImg_border, rightImg_border;
	copyMakeBorder(leftImg, leftImg_border, 0, 0, 0, max_offset, BORDER_REFLECT);
	copyMakeBorder(rightImg, rightImg_border, 0, 0, max_offset, 0, BORDER_REFLECT);

	//get costs
	vector<Mat> costs_ds;
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = min_offset; i <= max_offset; i++)
		{
			Mat cost_d(height, width, CV_32FC1);
			cost_d = getCostSAD_d(leftImg, rightImg_border, i, DISPARITY_LEFT, winSize);
			costs_ds.push_back(cost_d);
		}
	}
	else if (dispType == DISPARITY_RIGHT)
	{
		for (int i = min_offset; i <= max_offset; i++)
		{
			Mat cost_d(height, width, CV_32FC1);
			cost_d = getCostSAD_d(leftImg_border, rightImg, i, DISPARITY_RIGHT, winSize);
			costs_ds.push_back(cost_d);
		}
	}

	//get discretized intensity range
	int downSamStep = 256 * sampleRateR;
	vector<int> discretInten;
	for (int i = 0; i < 256; )
	{
		discretInten.push_back(i);
		i += downSamStep;
	}
	if (find(discretInten.begin(), discretInten.end(), 255) == discretInten.end())
	{
		discretInten.push_back(255);
	}
	//get aggregated costs
	//W_k_y
	std::map<int, vector<Mat> > setsJB_ks_ds_x;

	Mat M_k_y_l(height, width, CV_32FC1);
	Mat M_k_y_r(height, width, CV_32FC1);
	if (dispType == DISPARITY_LEFT)
	{
		for (vector<int>::iterator itor = discretInten.begin(); itor != discretInten.end();
			itor++)
		{
			M_k_y_l = abs(leftImg - *itor);
			M_k_y_l.convertTo(M_k_y_l, CV_32FC1);

			Mat M_ki_kr_y;
			vector<Mat> setsJ_k_ds_y;
			for (int i = 0; i < numDisparity; i++)
			{
				M_k_y_r = abs(rightImg_border(Rect(max_offset - i,0, width, height)) - *itor);
				M_k_y_r.convertTo(M_k_y_r, CV_32FC1);
				M_ki_kr_y = M_k_y_r.mul(M_k_y_l);

				Mat J_k_y(height, width, CV_32FC1);
				J_k_y = M_ki_kr_y.mul(costs_ds[i]);
				boxFilter(J_k_y, J_k_y, -1, Size(winSize, winSize));
				setsJ_k_ds_y.push_back(J_k_y);
			}

			boxFilter(M_ki_kr_y, M_ki_kr_y, -1, Size(winSize, winSize));

			vector<Mat> setsJB_k_ds_x;
			for (int i = 0; i < numDisparity; i++)
			{
				Mat setJB_k_d_x(height, width, CV_32FC1);
				setJB_k_d_x = setsJ_k_ds_y[i] / M_ki_kr_y;
				setsJB_k_ds_x.push_back(setJB_k_d_x);
			}
			setsJB_ks_ds_x[*itor] = setsJB_k_ds_x;
		}
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		for (vector<int>::iterator itor = discretInten.begin(); itor != discretInten.end();
			itor++)
		{
			M_k_y_r = abs(rightImg - *itor);
			M_k_y_r.convertTo(M_k_y_r, CV_32FC1);

			Mat M_ki_kr_y;
			vector<Mat> setsJ_k_ds_y;
			for (int i = 0; i < numDisparity; i++)
			{
				M_k_y_l = abs(leftImg_border(Rect(i, 0, width, height)) - *itor);
				M_k_y_l .convertTo(M_k_y_l, CV_32FC1);
				M_ki_kr_y = M_k_y_r.mul(M_k_y_l);

				Mat J_k_y(height, width, CV_32FC1);
				J_k_y = M_ki_kr_y.mul(costs_ds[i]);
				boxFilter(J_k_y, J_k_y, -1, Size(winSize, winSize));
				setsJ_k_ds_y.push_back(J_k_y);
			}

			boxFilter(M_ki_kr_y, M_ki_kr_y, -1, Size(winSize, winSize));

			vector<Mat> setsJB_k_ds_x;
			for (int i = 0; i < numDisparity; i++)
			{
				Mat setJB_k_d_x(height, width, CV_32FC1);
				setJB_k_d_x = setsJ_k_ds_y[i] / M_ki_kr_y;
				setsJB_k_ds_x.push_back(setJB_k_d_x);
			}
			setsJB_ks_ds_x[*itor] = setsJB_k_ds_x;
		}
	}

	costs_ds.clear();
	discretInten.clear();

	// calculate each pixel's ASW value
	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value
	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max());
		min_asw.push_back(tmp);
	}

	if (dispType == DISPARITY_LEFT)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				for (int offset = min_offset; offset <= max_offset; offset++)
				{
					double curCost_ = 0.0;
					int curIntensity = leftImg.at<uchar>(y, x);
					if(setsJB_ks_ds_x.count(curIntensity) == 0)
					{
						int lowerKey = curIntensity / downSamStep * downSamStep;
						int upperKey = lowerKey + downSamStep;
						if(upperKey > 255)
						{
							upperKey = 255;
						}

						curCost_ = (curIntensity - lowerKey) * setsJB_ks_ds_x[lowerKey][offset].at<float>(y, x)
							+ (upperKey - curIntensity) * setsJB_ks_ds_x[upperKey][offset].at<float>(y, x);
					}
					else 
					{
						Mat img__ = setsJB_ks_ds_x[curIntensity][offset];
						curCost_ = setsJB_ks_ds_x[curIntensity][offset].at<float>(y, x);
					}

					if (curCost_ < min_asw[y][x])
					{
						min_asw[y][x] = curCost_;
						// for better visualization
						depth.at<float>(y, x) = (float)offset;
					}
				}
			}
		}
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				for (int offset = min_offset; offset <= max_offset; offset++)
				{
					double curCost_ = numeric_limits<double>::max();
					int curIntensity = rightImg.at<uchar>(y, x);
					if (setsJB_ks_ds_x.count(curIntensity) == 0)
					{
						int lowerKey = curIntensity / downSamStep * downSamStep;
						int upperKey = lowerKey + downSamStep;
						if (upperKey > 255)
						{
							upperKey = 255;
						}

						curCost_ = (curIntensity - lowerKey) * setsJB_ks_ds_x[lowerKey][offset].at<float>(y, x)
							+ (upperKey - curIntensity)* setsJB_ks_ds_x[upperKey][offset].at<float>(y, x);
					}
					else
					{
						curCost_ = setsJB_ks_ds_x[curIntensity][offset].at<float>(y, x);
					}

					if (curCost_ < min_asw[y][x])
					{
						min_asw[y][x] = curCost_;
						// for better visualization
						depth.at<float>(y, x) = (float)offset;
					}
				}
			}
		}

	}

	return depth;
}

cv::Mat stereomatch_1::multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg)
{
	if (firstImg.size != secondImg.size)
		return Mat();

	if (firstImg.depth() == CV_32F && secondImg.depth() == CV_32F
		&& ((firstImg.channels() == 3 && secondImg.channels() == 3)
			|| (firstImg.channels() == 6 && secondImg.channels() == 6)))
	{
		int width = firstImg.cols;
		int height = firstImg.rows;

		Mat res(height, width, CV_32FC1);

		if (firstImg.channels() == 3)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<Vec3f>(j, i) * secondImg.at<Vec3f>(j, i);
				}
			}
		}
		else if (firstImg.channels() == 6)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<Vec6f>(j, i) * secondImg.at<Vec6f>(j, i);
				}
			}
		}
		return res;
	}
	return Mat();
}

cv::Mat stereomatch_1::getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
{
	if (guidedImg.size() != inputP.size())
		return Mat();

	int width = guidedImg.cols;
	int height = guidedImg.rows;

	normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	vector<Mat> guidedImg_split;
	cv::split(guidedImg, guidedImg_split);

	vector<Mat> corrGuid_split;
	Mat corrGuidP;
	for(int i = 0; i < guidedImg_split.size(); i++)
	{
		Mat corrGuid_channel;
		boxFilter(guidedImg_split[i].mul(inputP), corrGuid_channel, CV_32F, Size(r, r));
		corrGuid_split.push_back(corrGuid_channel);
	}
	merge(corrGuid_split, corrGuidP);

	Mat corrGuid;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, Size(r, r));

	Mat varGuid;
	varGuid = corrGuid - meanGuid.mul(meanGuid);

	vector<Mat> meanGrid_split;
	cv::split(meanGuid, meanGrid_split);

	vector<Mat> guidmul_split;
	Mat meanGuidmulP;
	for(int i = 0; i < meanGrid_split.size(); i++)
	{
		Mat guidmul_channel;
		guidmul_channel = meanGrid_split[i].mul(meanP);
		guidmul_split.push_back(guidmul_channel);
	}
	merge(guidmul_split, meanGuidmulP);

	Mat covGuidP;
	covGuidP = corrGuidP - meanGuidmulP;

	Mat a = covGuidP / (varGuid + (varGuid / varGuid) * eps);

	Mat b = meanP - multiChl_to_oneChl_mul(a, meanGuid);

	boxFilter(a, a, CV_32F, Size(r, r));
	boxFilter(b, b, CV_32F, Size(r, r));

	Mat filteredImg = multiChl_to_oneChl_mul(a, guidedImg) + b;
	return filteredImg;
}

/**
 * \brief ƥ����ۼ����㷨��SAD
 * \param leftImg 
 * \param rightImg 
 * \param dispType 
 * \param eps 
 * \param winSize 
 * \param minDisparity 
 * \param numDisparity 
 * \return 
 */
cv::Mat stereomatch_1::computeAdaptiveWeight_GuidedF(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType, double eps,
	int winSize, int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity - 1;
	int min_offset = minDisparity;

	Mat  leftImg_border, rightImg_border;
	copyMakeBorder(leftImg, leftImg_border, 0, 0, 0, max_offset, BORDER_REFLECT);
	copyMakeBorder(rightImg, rightImg_border, 0, 0, max_offset, 0, BORDER_REFLECT);

	//get costs
	vector<Mat> costs_ds;
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = min_offset; i <= max_offset; i++)
		{
			Mat cost_d(height, width, CV_32FC1);
			cost_d = getCostSAD_d(leftImg, rightImg_border, i, DISPARITY_LEFT, winSize);
			costs_ds.push_back(cost_d);
		}
	}
	else if (dispType == DISPARITY_RIGHT)
	{
		for (int i = min_offset; i <= max_offset; i++)
		{
			Mat cost_d(height, width, CV_32FC1);
			cost_d = getCostSAD_d(leftImg_border, rightImg, i, DISPARITY_RIGHT, winSize);
			costs_ds.push_back(cost_d);
		}
	}

	vector<Mat> adw_costsMat;
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = 0; i < numDisparity; i++)
		{
			//��������ͼ�Ĳ�ɫͼ��ϲ�Ϊ6 * width * height��ͼ����Ϊ����ͼ�����ڵ����˲�
			vector<Mat> left_cns, right_cns;
			split(leftImg, left_cns);
			split(rightImg_border(Rect(numDisparity - i - 1, 0, width, height)), right_cns);
			left_cns.insert(left_cns.end(), right_cns.begin(), right_cns.end());
			Mat guidedImg;
			merge(left_cns, guidedImg);

			//���ڵ����˲���ƥ����۾ۺ�
			Mat adw_costs_d = getGuidedFilter(guidedImg, costs_ds[i], winSize, eps);
			adw_costsMat.push_back(adw_costs_d);
		}
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		for (int i = 0; i < numDisparity; i++)
		{
			//��������ͼ�Ĳ�ɫͼ��ϲ�Ϊ6 * width * height��ͼ����Ϊ����ͼ�����ڵ����˲�
			vector<Mat> left_cns, right_cns;
			split(leftImg_border(Rect(i + minDisparity, 0, width, height)), left_cns);
			split(rightImg, right_cns);
			left_cns.insert(left_cns.end(), right_cns.begin(), right_cns.end());
			Mat guidedImg;
			merge(left_cns, guidedImg);

			//���ڵ����˲���ƥ����۾ۺ�
			Mat adw_costs_d = getGuidedFilter(guidedImg, costs_ds[i], winSize, eps);
			adw_costsMat.push_back(adw_costs_d);
		}
	}

	// calculate each pixel's ASW value
	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value
	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max()); //tmp��ʾͼ���һ������
		min_asw.push_back(tmp);
	}
	for(int x = 0; x < width; x++)
	{
		for(int y = 0; y < height; y++)
		{
			for(int disp = 0; disp < numDisparity; disp++)
			{
				double curCost_ = adw_costsMat[disp].at<float>(y, x);

				if (curCost_ < min_asw[y][x])	//ѡȡ ƥ����۾ۺ�ֵ��С���Ӳ�ֵ ��ΪĿ�����ص���Ӳ�ֵ��WTA
				{
					min_asw[y][x] = curCost_;
					// for better visualization
					depth.at<float>(y, x) = (float)(disp + minDisparity);
				}
			}
		}
	}
	return depth;
}

/**
 * \brief ƥ����ۼ����㷨�����ضԼ����ɫ���ݶ�������
 * \param leftImg 
 * \param rightImg 
 * \param dispType 
 * \param eps 
 * \param winSize 
 * \param minDisparity 
 * \param numDisparity 
 * \return 
 */
cv::Mat stereomatch_1::computeAdaptiveWeight_GuidedF_2(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType, double eps,
	int winSize, int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity - 1;
	int min_offset = minDisparity;

	Mat  leftImg_border, rightImg_border;
	copyMakeBorder(leftImg, leftImg_border, 0, 0, 0, max_offset, BORDER_REFLECT);
	copyMakeBorder(rightImg, rightImg_border, 0, 0, max_offset, 0, BORDER_REFLECT);

	//get costs
	vector<Mat> costs_ds;
	computeSimilarity(leftImg, rightImg, costs_ds, 0.4, 10, 50, dispType, minDisparity, numDisparity);

	vector<Mat> adw_costsMat;
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = 0; i < numDisparity; i++)
		{
			//vector<Mat> left_cns, right_cns;
			//split(leftImg, left_cns);
			//split(rightImg_border(Rect(numDisparity - i - 1, 0, width, height)), right_cns);
			//left_cns.insert(left_cns.end(), right_cns.begin(), right_cns.end());
			//Mat guidedImg;
			//merge(left_cns, guidedImg);

			Mat adw_costs_d = getGuidedFilter(leftImg, costs_ds[i], winSize, eps);
			adw_costsMat.push_back(adw_costs_d);
		}
	}
	else if (dispType == DISPARITY_RIGHT)
	{
		for (int i = 0; i < numDisparity; i++)
		{
			//vector<Mat> left_cns, right_cns;
			//split(leftImg_border(Rect(i + minDisparity, 0, width, height)), left_cns);
			//split(rightImg, right_cns);
			//left_cns.insert(left_cns.end(), right_cns.begin(), right_cns.end());
			//Mat guidedImg;
			//merge(left_cns, guidedImg);

			Mat adw_costs_d = getGuidedFilter(rightImg, costs_ds[i], winSize, eps);
			adw_costsMat.push_back(adw_costs_d);
		}
	}

	// calculate each pixel's ASW value
	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value
	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max()); //tmp��ʾͼ���һ������
		min_asw.push_back(tmp);
	}
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int disp = 0; disp < numDisparity; disp++)
			{
				double curCost_ = adw_costsMat[disp].at<float>(y, x);

				if (curCost_ < min_asw[y][x])
				{
					min_asw[y][x] = curCost_;
					// for better visualization
					depth.at<float>(y, x) = (float)(disp + minDisparity);
				}
			}
		}
	}
	return depth;
}

/**
 * \brief ƥ����ۼ����㷨��NCC
 * \param leftImg 
 * \param rightImg 
 * \param dispType 
 * \param eps 
 * \param winSize 
 * \param minDisparity 
 * \param numDisparity 
 * \return 
 */
cv::Mat stereomatch_1::computeAdaptiveWeight_GuidedF_3(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType, double eps,
	int winSize, int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity - 1;
	int min_offset = minDisparity;

	Mat  leftImg_border, rightImg_border;
	copyMakeBorder(leftImg, leftImg_border, 0, 0, 0, max_offset, BORDER_REFLECT);
	copyMakeBorder(rightImg, rightImg_border, 0, 0, max_offset, 0, BORDER_REFLECT);

	//get costs
	vector<Mat> costs_ds;
	computeNCC(leftImg, rightImg, costs_ds, dispType, winSize, minDisparity, numDisparity);

	vector<Mat> adw_costsMat;
	if (dispType == DISPARITY_LEFT)
	{
		for (int i = 0; i < numDisparity; i++)
		{
			vector<Mat> left_cns, right_cns;
			split(leftImg, left_cns);
			split(rightImg_border(Rect(numDisparity - i - 1, 0, width, height)), right_cns);
			left_cns.insert(left_cns.end(), right_cns.begin(), right_cns.end());
			Mat guidedImg;
			merge(left_cns, guidedImg);

			Mat adw_costs_d = getGuidedFilter(guidedImg, costs_ds[i], winSize, eps);
			adw_costsMat.push_back(adw_costs_d);
		}
	}
	else if (dispType == DISPARITY_RIGHT)
	{
		for (int i = 0; i < numDisparity; i++)
		{
			vector<Mat> left_cns, right_cns;
			split(leftImg_border(Rect(i + minDisparity, 0, width, height)), left_cns);
			split(rightImg, right_cns);
			left_cns.insert(left_cns.end(), right_cns.begin(), right_cns.end());
			Mat guidedImg;
			merge(left_cns, guidedImg);

			Mat adw_costs_d = getGuidedFilter(rightImg, costs_ds[i], winSize, eps);
			adw_costsMat.push_back(adw_costs_d);
		}
	}

	// calculate each pixel's ASW value
	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > max_asw; // store min ASW value
	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max()); //tmp��ʾͼ���һ������
		max_asw.push_back(tmp);
	}
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int disp = 0; disp < numDisparity; disp++)
			{
				double curCost_ = adw_costsMat[disp].at<float>(y, x);

				if (curCost_ < max_asw[y][x])
				{
					max_asw[y][x] = curCost_;
					// for better visualization
					depth.at<float>(y, x) = (float)(disp + minDisparity);
				}
			}
		}
	}
	return depth;
}

void stereomatch_1::computeColorWeightGau(cv::Mat src, std::vector< std::vector<cv::Mat> >& resWins, double rateR, int winSize)
{
	int width = src.cols;
	int height = src.rows;

	if (!resWins.empty())
	{
		resWins.clear();
	}

	if(winSize % 2 == 0)
	{
		return;
	}
	int halfWin = winSize / 2;

	Mat src_border;
	copyMakeBorder(src, src_border, halfWin, halfWin, halfWin, halfWin, BORDER_REFLECT);

	if (src.channels() == 3)
	{
		for (int y = 0; y < height; y++)
		{
			vector<Mat> rowWeights;
			for (int x = 0; x < width; x++)
			{
				Mat winImg = src_border(Rect(x, y, winSize, winSize));

				vector<Mat> winImg_cns;
				split(winImg, winImg_cns);
				Mat winDiff_0, winDiff_1, winDiff_2;
				absdiff(winImg_cns[0], winImg_cns[0].at<uchar>(halfWin, halfWin), winDiff_0);
				absdiff(winImg_cns[1], winImg_cns[1].at<uchar>(halfWin, halfWin), winDiff_1);
				absdiff(winImg_cns[2], winImg_cns[2].at<uchar>(halfWin, halfWin), winDiff_2);
				winDiff_0.convertTo(winDiff_0, CV_32FC1);
				winDiff_1.convertTo(winDiff_1, CV_32FC1);
				winDiff_2.convertTo(winDiff_2, CV_32FC1);

				Mat rangeDiff = (winDiff_0 + winDiff_1 + winDiff_2) / rateR * (-1);
				Mat dstWeight;
				exp(rangeDiff, dstWeight);
				rowWeights.push_back(dstWeight);
			}
			resWins.push_back(rowWeights);
		}
	}
	else if(src.channels() == 1)
	{
		for (int y = 0; y < height; y++)
		{
			vector<Mat> rowWeights;
			for (int x = 0; x < width; x++)
			{
				Mat winImg = src_border(Rect(x, y, winSize, winSize));
				Mat winDiff;
				absdiff(winImg, winImg.at<uchar>(halfWin, halfWin), winDiff);
				winDiff.convertTo(winDiff, CV_32FC1);

				Mat rangeDiff = winDiff / rateR * (-1);
				Mat dstWeight;
				exp(rangeDiff, dstWeight);
				rowWeights.push_back(dstWeight);
			}
			resWins.push_back(rowWeights);
		}
	}
}

void stereomatch_1::computeSpaceWeightGau(cv::Mat& dstKernel, int winSize, double rateS)
{
	dstKernel = cv::Mat::zeros(winSize, winSize, CV_32FC1);
	if(winSize % 2 == 0)
	{
		return;
	}

	int halfWin = winSize / 2;

	for(int y = 0; y < winSize; y++)
	{
		float yDist = (y - halfWin)*(y - halfWin);
		for (int x = 0; x < winSize; x++)
		{
			dstKernel.at<float>(x, y) = (x - halfWin)*(x - halfWin) + yDist;
		}
	}
	exp(dstKernel / rateS * (-1), dstKernel);
}

cv::Mat stereomatch_1::computeAdaptiveWeight_WeightedMedian(cv::Mat leftImg, cv::Mat rightImg,
	DisparityType dispType, int winSize,
	double sampleRateS, double sampleRateR,
	int minDisparity, int numDisparity)
{
	int width = leftImg.size().width;
	int height = leftImg.size().height;
	int max_offset = minDisparity + numDisparity - 1;
	int min_offset = minDisparity;

	if(winSize % 2 == 0 || leftImg.size != rightImg.size)
	{
		return Mat();
	}
	int halfWinSize = winSize / 2;

	Mat  leftImg_border, rightImg_border;
	copyMakeBorder(leftImg, leftImg_border, winSize, 0, 0, max_offset, BORDER_REFLECT);
	copyMakeBorder(rightImg, rightImg_border, 0, 0, max_offset, 0, BORDER_REFLECT);

	//get costs
	vector<Mat> costs_ds;
	computeSimilarity(leftImg, rightImg, costs_ds, 0.4, 10, 50, dispType, winSize, minDisparity, numDisparity);

	//get weights:W_BL

	Mat weightDist;
	computeSpaceWeightGau(weightDist, winSize, sampleRateS);

	vector<Mat> wmCost_ds;
	if (dispType == DISPARITY_LEFT)
	{
		//color-weight
		vector<vector<Mat> > weightWinsL, weightWinsR;
		computeColorWeightGau(leftImg, weightWinsL, sampleRateR, winSize);
		computeColorWeightGau(rightImg_border, weightWinsR, sampleRateR, winSize);
		for (int offset = 0; offset < numDisparity; offset++)
		{
			Mat wmCost(leftImg.size(), CV_32FC1);

			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					//get the cost and weights of each pixel in the support window of anchor pixel
					Mat cost_img_win = costs_ds[offset](Rect(x, y, winSize, winSize));
					Mat weight_img_win = weightWinsL[y][x].mul(weightDist).mul(weightWinsR[y][x - offset + numDisparity - 1]);

					std::multimap<float, float> cost_weight_s;
					for (int winY = 0; winY < winSize; winY++)
					{
						for (int winX = 0; winX < winSize; winX++)
						{
							cost_weight_s.insert(pair<float, float>(cost_img_win.at<float>(winY, winX), weight_img_win.at<float>(winY, winX)));
						}
					}
					double halfSumWeight = sum(weight_img_win)[0] / 2;

					//get the weighted median cost
					double partialSum = 0.0;
					for(std::multimap<float, float>::iterator itor = cost_weight_s.begin(); itor != cost_weight_s.end(); itor++)
					{
						partialSum += (*itor).second;
						if(partialSum > halfSumWeight)
						{
							if(itor == cost_weight_s.begin())
							{
								wmCost.at<float>(y, x) = (*itor).first;
							}
							else
							{
								itor--;
								wmCost.at<float>(y, x) = (*itor).first;		//weighted mean support								
							}
							break;
						}
					}
				}
			}
			wmCost_ds.push_back(wmCost);
		}
	}
	else if(dispType == DISPARITY_RIGHT)
	{
		//color-weight
		vector<vector<Mat> > weightWinsL, weightWinsR;
		computeColorWeightGau(leftImg_border, weightWinsL, sampleRateR, winSize);
		computeColorWeightGau(rightImg, weightWinsR, sampleRateR, winSize);
		for (int offset = 0; offset < numDisparity; offset++)
		{
			Mat wmCost(rightImg.size(), CV_32FC1);

			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					Mat cost_img_win = costs_ds[offset](Rect(x, y, winSize, winSize));
					Mat weight_img_win = weightWinsR[y][x].mul(weightDist).mul(weightWinsL[y][x + offset + min_offset]);

					std::multimap<float, float> cost_weight_s;
					for (int winY = 0; winY < winSize; winY++)
					{
						for (int winX = 0; winX < winSize; winX++)
						{
							cost_weight_s.insert(pair<float, float>(cost_img_win.at<float>(winY, winX), weight_img_win.at<float>(winY, winX)));
						}
					}
					double halfSumWeight = sum(weight_img_win)[0] / 2;

					//get the weighted median cost
					double partialSum = 0.0;
					for (std::multimap<float, float>::iterator itor = cost_weight_s.begin(); itor != cost_weight_s.end(); itor++)
					{
						partialSum += (*itor).second;
						if (partialSum > halfSumWeight)
						{
							itor--;
							wmCost.at<float>(y, x) = (*itor).first;		//weighted mean support
							break;
						}
					}
				}
			}
			wmCost_ds.push_back(wmCost);
		}
	}



	// calculate each pixel's ASW value
	Mat depth(height, width, CV_32FC1);
	vector< vector<double> > min_asw; // store min ASW value
	for (int i = 0; i < height; ++i)
	{
		vector<double> tmp(width, numeric_limits<double>::max()); //tmp��ʾͼ���һ������
		min_asw.push_back(tmp);
	}
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int disp = 0; disp < numDisparity; disp++)
			{
				double curCost_ = wmCost_ds[disp].at<float>(y, x);

				if (curCost_ < min_asw[y][x])
				{
					min_asw[y][x] = curCost_;
					// for better visualization
					depth.at<float>(y, x) = (float)(disp + minDisparity);
				}
			}
		}
	}
	return depth;
}