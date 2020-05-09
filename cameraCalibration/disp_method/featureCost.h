#pragma once
#include <opencv2/opencv.hpp>
#include "methods_disp.h"

enum FEATURE_DESCRIPTOR
{
	FEATURE_SURF = 1,
	FEATURE_SIFT = 2,
	FEATURE_ORB = 3,
};

void getPts(std::vector<std::vector<cv::KeyPoint> >& pts_out, cv::Size imgSize, int winSize);
void computeDescript(cv::Mat src, std::vector<std::vector<cv::KeyPoint> >& pts, 
	std::vector<cv::Mat>& descriptors, int winSize, FEATURE_DESCRIPTOR feature_des);
void computeDisp_Feature(StereoMatchParam param, cv::Mat& disp, FEATURE_DESCRIPTOR feature_des);
void computeDisp_Feature(StereoMatchParam param, std::vector<cv::Mat>& cost_d, FEATURE_DESCRIPTOR feature_des);