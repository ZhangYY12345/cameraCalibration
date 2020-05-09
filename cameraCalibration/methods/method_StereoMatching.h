#pragma once
#include <opencv2/opencv.hpp>
#include "parametersStereo.h"
#include <boost/fusion/container/vector/vector.hpp>
#include <boost/unordered_map.hpp>

	struct MY_COMP_Point2i {
		bool operator()(const cv::Point& left, const cv::Point& right) const
		{
			if (left.x < right.x)
			{
				return true;
			}
			if (left.x == right.x && left.y < right.y)
			{
				return true;
			}

			return false;
		}
	};

	struct MY_COMP_Point3i {
		bool operator()(const cv::Point3i& left, const cv::Point3i& right) const
		{
			if (left.x < right.x)
			{
				return true;
			}
			if (left.x == right.x && left.y < right.y)
			{
				return true;
			}
			if (left.x == right.x && left.y == right.y && left.z == right.z)
			{
				return true;
			}

			return false;
		}
	};

	struct MY_COMP_vec3i
	{
		bool operator()(const cv::Vec3i& left, const cv::Vec3i& right) const
		{
			if (left[0] < right[0])
			{
				return true;
			}
			if (left[0] == right[0] && left[1] < right[1])
			{
				return true;
			}
			if (left[0] == right[0] && left[1] == right[1] && left[2] < right[2])
			{
				return true;
			}

			return false;
		}
	};

	struct MY_COMP_vec4i
	{
		bool operator()(const cv::Vec4i& left, const cv::Vec4i& right) const
		{
			if (left[0] < right[0])
			{
				return true;
			}
			if (left[0] == right[0] && left[1] < right[1])
			{
				return true;
			}
			if (left[0] == right[0] && left[1] == right[1] && left[2] < right[2])
			{
				return true;
			}
			if (left[0] == right[0] && left[1] == right[1] && left[2] == right[2] && left[3] < right[3])
			{
				return true;
			}

			return false;
		}
	};

namespace stereomatch_1 {
	//stereo matching four steps:1.2.3.4.
	//1.match cost computation
	//2.coat aggregation
	//absolute differences
	cv::Mat computeAD(cv::Mat leftImg, cv::Mat rightImg, int minDisparity = 0, int numDisparity = 30);
	//sum of absolute differences
	cv::Mat computeSAD(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int minDisparity = 0, int numDisparity = 30);
	cv::Mat computeSAD_inteOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int minDisparity = 0, int numDisparity = 30);
	cv::Mat computeSAD_BFOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int minDisparity = 0, int numDisparity = 30);

	//truncated absolute differences
	cv::Mat computeTAD(cv::Mat leftImg, cv::Mat rightImg, int threshold_T = 30, int minDisparity = 0, int numDisparity = 30);
	//sum of truncated absolute differences
	cv::Mat computeSTAD(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int threshold_T = 30, int minDisparity = 0, int numDisparity = 30);
	cv::Mat computeSTAD_inteOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int threshold_T = 30, int minDisparity = 0, int numDisparity = 30);
	cv::Mat computeSTAD_BFOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int threshold_T = 30, int minDisparity = 0, int numDisparity = 30);

	//squared differences
	cv::Mat computeSD(cv::Mat leftImg, cv::Mat rightImg, int minDisparity = 0, int numDisparity = 30);
	//sum of squared differences
	cv::Mat computeSSD(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int minDisparity = 0, int numDisparity = 30);
	cv::Mat computeSSD_inteOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int minDisparity = 0, int numDisparity = 30);
	cv::Mat computeSSD_BFOpti(cv::Mat leftImg, cv::Mat rightImg, int winSize = 7, int minDisparity = 0, int numDisparity = 30);

	//normalized cross correlation,ncc
	void getInputImgNCC(cv::Mat src, std::vector<std::vector<cv::Mat> >& dst, int winSize);
	cv::Mat computeNCC(cv::Mat leftImg, cv::Mat rightImg, DisparityType dispType = DISPARITY_LEFT,
		int winSize = 7, int minDisparity = 0, int numDisparity = 30);
	void computeNCC(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_ds,
		DisparityType dispType = DISPARITY_LEFT, int winSize = 7,
		int minDisparity = 0, int numDisparity = 30);
	cv::Mat ncc(cv::Mat in1, cv::Mat in2, std::string type, bool add_constant);

	//cost computation using pixels' color and gradient similarity 
	void computeSimilarity(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_d_imgs,
		double regularity, double thresC, double thresG, DisparityType dispType,
		int minDisparity, int numDisparity);
	void computeSimilarity(cv::Mat leftImg, cv::Mat rightImg, std::vector<cv::Mat>& cost_d_imgs,
		double regularity, double thresC, double thresG, DisparityType dispType,
		int winSize, int minDisparity, int numDisparity);

	//shiftable windows
	cv::Mat computeShiftableWin(cv::Mat leftImg, cv::Mat rightImg, int winSize = 3, int minDisparity = 0, int numDisparity = 30);

	//multiple windows
	cv::Mat computeMultiWin(cv::Mat leftImg, cv::Mat rightImg, int winSize = 3, int winNum = 9, int minDisparity = 0, int numDisparity = 30);
	//3.disparity computation/optimization: integral image or box-filtering


	//4.disparity refinement


	//adaptive support windows
	cv::Mat asw(cv::Mat in1, cv::Mat in2, std::string type);
	cv::Mat computeAdaptiveWeight(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, int winSize = 7, int minDisparity = 186, int numDisparity = 144);
	cv::Mat computeAdaptiveWeight_direct8(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, int winSize = 7, int minDisparity = 186, int numDisparity = 144);


	float getColorDist(cv::Vec3b pointA, cv::Vec3b pointB);
	void getWinGeoDist(cv::Mat originImg, cv::Mat& winDistImg, int winSize = 15, int iterTime = 3);
	void getGeodesicDist(cv::Mat originImg, std::map<cv::Point, cv::Mat, MY_COMP_Point2i>& weightGeoDist, int winSize = 15, int iterTime = 3);
	cv::Mat computeAdaptiveWeight_geodesic(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, int winSize = 7, int minDisparity = 186, int numDisparity = 144);


	void createBilGrid(cv::Mat image, std::map<cv::Vec3i, std::pair<double, int>, MY_COMP_vec3i>& bilGrid,
		double sampleRateS = 16, double sampleRateR = 0.07);
	void createBilGrid(cv::Mat imageL, cv::Mat imageR, std::map<cv::Vec4i, std::pair<double, int>, MY_COMP_vec4i>& bilGrid,
		int disparity, DisparityType dispType = DISPARITY_LEFT, double sampleRateS = 16, double sampleRateR = 0.07);
	void createBilGrid(cv::Mat imageL, cv::Mat imageR, std::map<int, std::map<int, std::map<int, std::map<int, std::pair<double, int> > > > >& bilGrid,
		int disparity, DisparityType dispType = DISPARITY_LEFT, double sampleRateS = 16, double sampleRateR = 0.07);

	double trilinear_3d(std::vector<double> axis_diff_xyz, std::vector<double> neighbor_xyz_02);
	double quadrlinear_blGrid(std::vector<double> axis_diff_xyzw, std::vector<double> neighbor_xyzw_02);
	cv::Mat computeAdaptiveWeight_bilateralGrid(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, double sampleRateS = 10, double sampleRateR = 10,
		int minDisparity = 186, int numDisparity = 144);


	cv::Mat getCostSAD_d(cv::Mat leftImg, cv::Mat rightImg, int disparity, DisparityType dispType = DISPARITY_LEFT, int winSize = 35);
	cv::Mat computeAdaptiveWeight_BLO1(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, double sampleRateR = 10, int winSize = 35,
		int minDisparity = 186, int numDisparity = 144);


	cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg);
	cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);
	cv::Mat computeAdaptiveWeight_GuidedF(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, double eps = 0.01, int winSize = 35,
		int minDisparity = 186, int numDisparity = 144);
	cv::Mat computeAdaptiveWeight_GuidedF_2(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, double eps = 0.01, int winSize = 35,
		int minDisparity = 186, int numDisparity = 144);
	cv::Mat computeAdaptiveWeight_GuidedF_3(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, double eps = 1e-6, int winSize = 35,
		int minDisparity = 186, int numDisparity = 144);


	void computeColorWeightGau(cv::Mat src, std::vector< std::vector<cv::Mat> >& resWins, double rateR, int winSize);
	void computeSpaceWeightGau(cv::Mat& dstKernel, int winSize, double rateS);
	cv::Mat computeAdaptiveWeight_WeightedMedian(cv::Mat leftImg, cv::Mat rightImg,
		DisparityType dispType = DISPARITY_LEFT, int winSize = 35,
		double sampleRateS = 10, double sampleRateR = 10,
		int minDisparity = 186, int numDisparity = 144);
}