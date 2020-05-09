#include "featureCost.h"
#include "nonfree.hpp"
#include <future>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;

void getPts(std::vector<std::vector<cv::KeyPoint> >& pts_out, Size imgSize, int winSize)
{
	for(int h = 0; h < imgSize.height; h++)
	{
		std::vector<cv::KeyPoint> oneRowPt;
		for(int w = 0; w < imgSize.width; w++)
		{
			KeyPoint pt(w, h, winSize);
			oneRowPt.push_back(pt);
		}
		pts_out.push_back(oneRowPt);
	}
}

void computeDescript(cv::Mat src, std::vector<std::vector<cv::KeyPoint> >& pts, std::vector<cv::Mat>& descriptors, int winSize, FEATURE_DESCRIPTOR feature_des)
{
	std::vector<std::vector<cv::KeyPoint> > prePts;
	getPts(prePts, src.size(), winSize);

	switch(feature_des)
	{
	case FEATURE_SURF:
	{
		Ptr<DescriptorExtractor> descriptor = xfeatures2d::SURF::create();
		for (int i = 0; i < src.rows; i++)
		{
			std::vector<KeyPoint> keyPointOneRow = prePts[i];
			Mat descriptorOneRow;
			descriptor->compute(src, keyPointOneRow, descriptorOneRow);
			if(!descriptorOneRow.empty())
			{
				pts.push_back(keyPointOneRow);
				descriptors.push_back(descriptorOneRow);
			}
		}
	}
	break;
	case FEATURE_SIFT:
	{
		Ptr<DescriptorExtractor> descriptor = xfeatures2d::SIFT::create();
		for (int i = 0; i < src.rows; i++)
		{
			std::vector<KeyPoint> keyPointOneRow = prePts[i];
			Mat descriptorOneRow;
			descriptor->compute(src, keyPointOneRow, descriptorOneRow);
			if (!descriptorOneRow.empty())
			{
				pts.push_back(keyPointOneRow);
				descriptors.push_back(descriptorOneRow);
			}
		}
	}
	break;
	case FEATURE_ORB:
	{
		Ptr<DescriptorExtractor> descriptor = xfeatures2d::BriefDescriptorExtractor::create(32, true);
		for (int i = 0; i < src.rows; i++)
		{
			std::vector<KeyPoint> keyPointOneRow = prePts[i];
			Mat descriptorOneRow;
			descriptor->compute(src, keyPointOneRow, descriptorOneRow);
			if (!descriptorOneRow.empty())
			{
				pts.push_back(keyPointOneRow);
				descriptors.push_back(descriptorOneRow);
			}
		}
	}
	break;
	}
}

void computeDisp_Feature(StereoMatchParam param, Mat& disp, FEATURE_DESCRIPTOR feature_des)
{
	int width = param.imgLeft.cols;
	int height = param.imgLeft.rows;

	int min_offset = param.minDisparity;
	int max_offset = param.maxDisparity;

	if (param.isDispLeft)
	{
		Mat right_border;
		copyMakeBorder(param.imgRight, right_border, 0, 0, max_offset, 0, BORDER_REFLECT);

		std::vector<std::vector<KeyPoint> > keyPointL, keyPointR;
		std::vector<Mat> descriptorsL, descriptorsR;
		std::future<void> ft1 = std::async(std::launch::async, [&]
		{
			computeDescript(param.imgLeft, keyPointL, descriptorsL, param.winSize, feature_des);
		});
		std::future<void> ft2 = std::async(std::launch::async, [&]
		{
			computeDescript(right_border, keyPointR, descriptorsR, param.winSize, feature_des);
		});
		ft1.wait();
		ft2.wait();

		Mat disparityMap(height, width, CV_64F, Scalar::all(0));
#pragma omp parallel for
		for (int h = 0; h < height; h++)
		{
			for(int w = 0; w < width; w++)
			{
				Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
				std::vector<DMatch> matches;
				cv::Mat descriptor_L_wh = descriptorsL[h].row(w);
				cv::Mat descriptor_R_0_d = descriptorsR[h](Rect(0, w, descriptorsR[h].cols, max_offset));
				matcher->match(descriptor_L_wh, descriptor_R_0_d, matches);
				if(matches.size() == 1)
				{
					disparityMap.at<double>(h, w) = max_offset - matches[0].trainIdx;
				}
			}
		}
		disp = disparityMap.clone();

		Mat disparityMap_dst;
		cv::normalize(disparityMap, disparityMap_dst, 0, 255, NORM_MINMAX);
		disparityMap_dst.convertTo(disparityMap_dst, CV_8U);
		imwrite("disparityL.jpg", disparityMap_dst);
	}
	else
	{
		Mat left_border;
		copyMakeBorder(param.imgLeft, left_border, 0, 0, 0, max_offset, BORDER_REFLECT);

		std::vector<std::vector<KeyPoint> > keyPointL, keyPointR;
		std::vector<Mat> descriptorsL, descriptorsR;
		std::future<void> ft1 = std::async(std::launch::async, [&]
		{
			computeDescript(left_border, keyPointL, descriptorsL, param.winSize, feature_des);
		});
		std::future<void> ft2 = std::async(std::launch::async, [&]
		{
			computeDescript(param.imgRight, keyPointR, descriptorsR, param.winSize, feature_des);
		});
		ft1.wait();
		ft2.wait();

		Mat disparityMap(height, width, CV_64F, Scalar::all(0));
#pragma omp parallel for
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");//使用L2范数表示距离
				std::vector<DMatch> matches;
				cv::Mat descriptor_R_wh = descriptorsR[h].row(w);
				cv::Mat descriptor_L_0_d = descriptorsL[h](Rect(0, w, descriptorsL[h].cols, max_offset));
				matcher->match(descriptor_R_wh, descriptor_L_0_d, matches);
				if (matches.size() == 1)
				{
					disparityMap.at<double>(h, w) = matches[0].trainIdx;
				}
			}
		}
		disp = disparityMap.clone();

		Mat disparityMap_dst;
		cv::normalize(disparityMap, disparityMap_dst, 0, 255, NORM_MINMAX);
		disparityMap_dst.convertTo(disparityMap_dst, CV_8U);
		imwrite("disparityR.jpg", disparityMap_dst);
	}
}

void computeDisp_Feature(StereoMatchParam param, std::vector<cv::Mat>& cost_d,
	FEATURE_DESCRIPTOR feature_des)
{
	int width = param.imgLeft.cols;
	int height = param.imgLeft.rows;

	int min_offset = param.minDisparity;
	int max_offset = param.maxDisparity;
	int numDisp = max_offset - min_offset + 1;

	int featureWinSize = param.winSize;
	if(feature_des == FEATURE_SIFT)
	{
		featureWinSize = 3;
	}
	if (param.isDispLeft)
	{
		Mat right_border;
		copyMakeBorder(param.imgRight, right_border, 0, 0, max_offset, 0, BORDER_REFLECT);

		std::vector<std::vector<KeyPoint> > keyPointL, keyPointR;
		std::vector<Mat> descriptorsL, descriptorsR;
		std::future<void> ft1 = std::async(std::launch::async, [&]
		{
			computeDescript(param.imgLeft, keyPointL, descriptorsL, featureWinSize, feature_des);
		});
		std::future<void> ft2 = std::async(std::launch::async, [&]
		{
			computeDescript(right_border, keyPointR, descriptorsR, featureWinSize, feature_des);
		});
		ft1.wait();
		ft2.wait();

		NormTypes norm_type;
		if(feature_des == FEATURE_SURF || feature_des == FEATURE_SIFT)
		{
			norm_type = NORM_L2;
		}
		else if(feature_des == FEATURE_ORB)
		{
			norm_type = NORM_HAMMING;
		}

		if(!cost_d.empty())
		{
			cost_d.clear();
		}
		std::vector<cv::Mat> cost_d_temp(numDisp);
		for(int i = 0; i < numDisp; i++)
		{
			cost_d_temp[i] = cv::Mat::zeros(height, width, CV_32F);
		}
#pragma omp parallel for
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				cv::Mat descriptor_L_wh = descriptorsL[h].row(w);
				for(int i = 0; i < numDisp; i++)
				{
					cv::Mat descriptor_R_0_d = descriptorsR[h].row(w + max_offset -( i + min_offset));
					cost_d_temp[i].at<float>(h, w) = cv::norm(descriptor_L_wh, descriptor_R_0_d, norm_type);//L2范数距离
				}
			}
		}
		cost_d.swap(cost_d_temp);
	}
	else
	{
		Mat left_border;
		copyMakeBorder(param.imgLeft, left_border, 0, 0, 0, max_offset, BORDER_REFLECT);

		std::vector<std::vector<KeyPoint> > keyPointL, keyPointR;
		std::vector<Mat> descriptorsL, descriptorsR;
		std::future<void> ft1 = std::async(std::launch::async, [&]
		{
			computeDescript(left_border, keyPointL, descriptorsL, featureWinSize, feature_des);
		});
		std::future<void> ft2 = std::async(std::launch::async, [&]
		{
			computeDescript(param.imgRight, keyPointR, descriptorsR, featureWinSize, feature_des);
		});
		ft1.wait();
		ft2.wait();

		NormTypes norm_type;
		if (feature_des == FEATURE_SURF || feature_des == FEATURE_SIFT)
		{
			norm_type = NORM_L2;
		}
		else if (feature_des == FEATURE_ORB)
		{
			norm_type = NORM_HAMMING;
		}

		if (!cost_d.empty())
		{
			cost_d.clear();
		}
		std::vector<cv::Mat> cost_d_temp(numDisp);
		for (int i = 0; i < numDisp; i++)
		{
			cost_d_temp[i] = cv::Mat::zeros(height, width, CV_32F);
		}
#pragma omp parallel for
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				cv::Mat descriptor_R_wh = descriptorsR[h].row(w);
				for (int i = 0; i < numDisp; i++)
				{
					cv::Mat descriptor_L_0_d = descriptorsL[h].row(w+ i + min_offset);
					cost_d_temp[i].at<float>(h, w) = cv::norm(descriptor_R_wh, descriptor_L_0_d, norm_type);
				}
			}
		}
		cost_d.swap(cost_d_temp);
	}
}
