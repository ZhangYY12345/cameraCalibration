#include "gifs.h"
#include "steerFilter.h"
#include <pcl/filters/conditional_removal.h>

using namespace cv;
using namespace std;

cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg)
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
					res.at<float>(j, i) = firstImg.at<Vec3f>(j, i).dot(secondImg.at<Vec3f>(j, i));
				}
			}
		}
		else if (firstImg.channels() == 6)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<Vec6f>(j, i).dot(secondImg.at<Vec6f>(j, i));
				}
			}
		}
		return res;
	}
	return firstImg.mul(secondImg);
}

cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
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

	std::vector<Mat> guidedImg_split;
	cv::split(guidedImg, guidedImg_split);

	std::vector<Mat> corrGuid_split;
	Mat corrGuidP;
	for (int i = 0; i < guidedImg_split.size(); i++)
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

	std::vector<Mat> meanGrid_split;
	cv::split(meanGuid, meanGrid_split);

	std::vector<Mat> guidmul_split;
	Mat meanGuidmulP;
	for (int i = 0; i < meanGrid_split.size(); i++)
	{
		Mat guidmul_channel;
		guidmul_channel = meanGrid_split[i].mul(meanP);
		guidmul_split.push_back(guidmul_channel);
	}
	merge(guidmul_split, meanGuidmulP);

	Mat covGuidP;
	covGuidP = corrGuidP - meanGuidmulP;

	//create image mask for matrix adding integer
	Mat onesMat = Mat::ones(varGuid.size(), varGuid.depth());
	Mat mergeOnes;
	if (varGuid.channels() == 1)
	{
		mergeOnes = onesMat;
	}
	else if (varGuid.channels() == 3)
	{
		std::vector<Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}
	else if (varGuid.channels() == 6)
	{
		std::vector<Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}

	Mat a = covGuidP / (varGuid + mergeOnes * eps);
	Mat b = meanP - multiChl_to_oneChl_mul(a, meanGuid);

	boxFilter(a, a, CV_32F, Size(r, r));
	boxFilter(b, b, CV_32F, Size(r, r));

	Mat filteredImg = multiChl_to_oneChl_mul(a, guidedImg) + b;
	return filteredImg;
}

void getGIF(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& filteredImg, cv::Mat& a, cv::Mat& b)
{
	if (guidedImg.size() != inputP.size())
		return;

	int width = guidedImg.cols;
	int height = guidedImg.rows;

	normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	std::vector<Mat> guidedImg_split;
	cv::split(guidedImg, guidedImg_split);

	std::vector<Mat> corrGuid_split;
	Mat corrGuidP;
	for (int i = 0; i < guidedImg_split.size(); i++)
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

	std::vector<Mat> meanGrid_split;
	cv::split(meanGuid, meanGrid_split);

	std::vector<Mat> guidmul_split;
	Mat meanGuidmulP;
	for (int i = 0; i < meanGrid_split.size(); i++)
	{
		Mat guidmul_channel;
		guidmul_channel = meanGrid_split[i].mul(meanP);
		guidmul_split.push_back(guidmul_channel);
	}
	merge(guidmul_split, meanGuidmulP);

	Mat covGuidP;
	covGuidP = corrGuidP - meanGuidmulP;

	//create image mask for matrix adding integer
	Mat onesMat = Mat::ones(varGuid.size(), varGuid.depth());
	Mat mergeOnes;
	if (varGuid.channels() == 1)
	{
		mergeOnes = onesMat;
	}
	else if (varGuid.channels() == 3)
	{
		std::vector<Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}
	else if (varGuid.channels() == 6)
	{
		std::vector<Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}

	a = covGuidP / (varGuid + mergeOnes * eps);
	b = meanP - multiChl_to_oneChl_mul(a, meanGuid);

	boxFilter(a, a, CV_32F, Size(r, r));
	boxFilter(b, b, CV_32F, Size(r, r));

	filteredImg = multiChl_to_oneChl_mul(a, guidedImg) + b;
}

//guided image is always one-channel in following functions
cv::Mat getGuidedFilter_egif(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& beta_)
{
	double eps0 = 0.001 * 0.001;

	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	double mean2VarGG = sum(varGuidGuid)[0] / (varGuidGuid.cols * varGuidGuid.rows);
	Mat a = covGuidP / (varGuidGuid + mean2VarGG * eps + eps0);
	Mat b = meanP - a.mul(meanGuid);

	boxFilter(a, a, CV_32F, Size(r, r));
	boxFilter(b, b, CV_32F, Size(r, r));

	double gamma = 1.0;
	cv::Mat beta(a.size(), CV_32FC1);
	beta = a / (1 - a);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			beta.at<float>(i, j) = pow(beta.at<float>(i, j), gamma);
		}
	}
	beta_ = beta.clone();

	Mat filteredImg = a.mul(guidedImg) + b;
	return filteredImg;
}

cv::Mat getGuidedFilter_egif_3cn(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
{
	vector<cv::Mat> gImgs, pImgs;
	split(guidedImg, gImgs);
	split(inputP, pImgs);

	vector<cv::Mat> filteredImg(3);
	for (int i = 0; i < gImgs.size(); i++)
	{
		cv::Mat beta;
		cv::Mat filtered;
		filtered = getGuidedFilter_egif(gImgs[i], pImgs[i], r, eps, beta);

		cv::Mat temp = gImgs[i] - pImgs[i];
		temp.convertTo(temp, beta.type());
		filteredImg[i] = temp.mul(beta) + filtered;
	}

	cv::Mat dst;
	merge(filteredImg, dst);
	return dst;
}

void getEGIF(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& beta_, cv::Mat& filteredImg, cv::Mat& a,
	cv::Mat& b)
{
	double eps0 = 0.001 * 0.001;

	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	double mean2VarGG = sum(varGuidGuid)[0] / (varGuidGuid.cols * varGuidGuid.rows);
	a = covGuidP / (varGuidGuid + mean2VarGG * eps + eps0);
	b = meanP - a.mul(meanGuid);

	boxFilter(a, a, CV_32F, Size(r, r));
	boxFilter(b, b, CV_32F, Size(r, r));

	double gamma = 1.0;
	cv::Mat beta(a.size(), CV_32FC1);
	beta = a / (1 - a);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			beta.at<float>(i, j) = pow(beta.at<float>(i, j), gamma);
		}
	}
	beta_ = beta.clone();

	filteredImg = a.mul(guidedImg) + b;

}

cv::Mat edgeAwareWeight(cv::Mat guidedImg)
{
	double minVal, maxVal;
	cv::minMaxLoc(guidedImg, &minVal, &maxVal, NULL, NULL);
	double L = maxVal - minVal;
	double eps = (0.001 * L) * (0.001 * L);

	int r = 1;

	Size imgSize = guidedImg.size();
	cv::Mat meanG;
	boxFilter(guidedImg, meanG, CV_32F, Size(r, r));
	cv::Mat meanGG;
	boxFilter(guidedImg.mul(guidedImg), meanGG, CV_32F, Size(r, r));

	cv::Mat varG;
	varG = meanGG - meanG.mul(meanG);

	cv::Mat varG1 = varG + eps;
	double varG1_sum = sum(1.0/varG1)[0];

	double val_ = varG1_sum / (imgSize.width * imgSize.height);
	cv::Mat gamma0 = varG1 * val_;

	cv::Mat gamma;
	GaussianBlur(gamma0, gamma, Size(3, 3), 2);
	return gamma;
}

cv::Mat getGuidedFilter_wgif(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
{
	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	cv::Mat gamma = edgeAwareWeight(guidedImg);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));
	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat a = covGuidP / (varGuidGuid + eps / gamma);
	Mat b = meanP - a.mul(meanGuid);

	cv::Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_32F, Size(r, r));
	boxFilter(b, mean_b, CV_32F, Size(r, r));
	
	Mat filteredImg = mean_a.mul(guidedImg) + mean_b;
	return filteredImg;
}

cv::Mat getGuidedFilter_wgif_3cn(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
{
	vector<cv::Mat> gImgs, pImgs;
	split(guidedImg, gImgs);
	split(inputP, pImgs);

	vector<cv::Mat> filteredImg(3);
	for (int i = 0; i < gImgs.size(); i++)
	{
		filteredImg[i] = getGuidedFilter_wgif(gImgs[i], pImgs[i], r, eps);
	}

	cv::Mat dst;
	merge(filteredImg, dst);
	return dst;
}

void getWGIF(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& filteredImg, cv::Mat& a, cv::Mat& b)
{
	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	cv::Mat gamma = edgeAwareWeight(guidedImg);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));
	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	a = covGuidP / (varGuidGuid + eps / gamma);
	b = meanP - a.mul(meanGuid);

	cv::Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_32F, Size(r, r));
	boxFilter(b, mean_b, CV_32F, Size(r, r));

	filteredImg = mean_a.mul(guidedImg) + mean_b;
}

cv::Mat getGuidedFilter_skwgif(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h)
{
	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid, meanGuid2;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	boxFilter(guidedImg, meanGuid2, CV_32F, Size(r2, r2));

	Mat meanP, meanP2;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));
	boxFilter(inputP, meanP2, CV_32F, Size(r2, r2));

	cv::Mat corrGuid, corrGuid2;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, Size(r, r));
	boxFilter(guidedImg.mul(guidedImg), corrGuid2, CV_32F, Size(r2, r2));

	cv::Mat corrGuidP;
	boxFilter(guidedImg.mul(inputP), corrGuidP, CV_32F, Size(r, r));

	cv::Mat varGuid, varGuid2;
	varGuid = corrGuid - meanGuid.mul(meanGuid);
	varGuid2 = corrGuid2 - meanGuid2.mul(meanGuid2);

	cv::Mat temp = varGuid2 + namuda;
	cv::Mat gammaGuid = temp * sum(1.0 / temp)[0] / (temp.rows * temp.cols);

	cv::Mat covGuidP;
	covGuidP = corrGuidP - meanGuid.mul(meanP);

	cv::Mat a = covGuidP / (varGuid + eps / gammaGuid);
	cv::Mat b = meanP - a.mul(meanGuid);

	cv::Mat steerFilter_;
	steerFilter(guidedImg, steerFilter_);
	steerFilter_.convertTo(steerFilter_, CV_32F);

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);


	for (int i = 0; i < steerFilter_.cols; i++)
	{
		for (int j = 0; j < steerFilter_.rows; j++)
		{
			cv::Mat temp_ = steerFilter_(Rect(cv::Point(max(0, i - r2 / 2), max(0, j - r2 / 2)),
				cv::Point(min(steerFilter_.cols - 1, i + r2 / 2), min(steerFilter_.rows - 1, j + r2 / 2))));
			int num_ = temp_.cols * temp_.rows;

			//
			cv::Mat subSrc_a = a(Rect(cv::Point(max(0, i - r2 / 2), max(0, j - r2 / 2)),
				cv::Point(min(steerFilter_.cols - 1, i + r2 / 2), min(steerFilter_.rows - 1, j + r2 / 2))));

			cv::Mat subres_a = temp_.mul(subSrc_a);
			double val_a = sum(subres_a)[0] * 1.0 / sum(temp_)[0];
			mean_a.at<float>(j, i) = val_a;

			//
			cv::Mat subSrc_b = b(Rect(cv::Point(max(0, i - r2 / 2), max(0, j - r2 / 2)),
				cv::Point(min(steerFilter_.cols - 1, i + r2 / 2), min(steerFilter_.rows - 1, j + r2 / 2))));

			cv::Mat subres_b = temp_.mul(subSrc_b);
			double val_b = sum(sum(subres_b))[0] * 1.0 / sum(temp_)[0];
			mean_b.at<float>(j, i) = val_b;
		}
	}

	//todo

	Mat filteredImg = mean_a.mul(guidedImg) + mean_b;
	return filteredImg;
}

cv::Mat getGuidedFilter_skwgif2(cv::Mat guidedImg, cv::Mat inputP, cv::Mat steerFilterW, int r, int r2, double eps,
	double namuda, double h)
{
	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid, meanGuid2;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	boxFilter(guidedImg, meanGuid2, CV_32F, Size(r2, r2));

	Mat meanP, meanP2;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));
	boxFilter(inputP, meanP2, CV_32F, Size(r2, r2));

	cv::Mat corrGuid, corrGuid2;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, Size(r, r));
	boxFilter(guidedImg.mul(guidedImg), corrGuid2, CV_32F, Size(r2, r2));

	cv::Mat corrGuidP;
	boxFilter(guidedImg.mul(inputP), corrGuidP, CV_32F, Size(r, r));

	cv::Mat varGuid, varGuid2;
	varGuid = corrGuid - meanGuid.mul(meanGuid);
	varGuid2 = corrGuid2 - meanGuid2.mul(meanGuid2);

	cv::Mat temp = varGuid2 + namuda;
	cv::Mat gammaGuid = temp * sum(1.0 / temp)[0] / (temp.rows * temp.cols);

	cv::Mat covGuidP;
	covGuidP = corrGuidP - meanGuid.mul(meanP);

	cv::Mat a = covGuidP / (varGuid + eps / gammaGuid);
	cv::Mat b = meanP - a.mul(meanGuid);

	cv::Mat steerFilter_;
	if(steerFilterW.type() != CV_32F)
	{
		steerFilterW.convertTo(steerFilter_, CV_32F);
	}
	else
	{
		steerFilter_ = steerFilterW.clone();
	}

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);


	for (int i = 0; i < steerFilter_.cols; i++)
	{
		for (int j = 0; j < steerFilter_.rows; j++)
		{
			cv::Mat temp_ = steerFilter_(Rect(cv::Point(max(0, i - r2 / 2), max(0, j - r2 / 2)),
				cv::Point(min(steerFilter_.cols - 1, i + r2 / 2), min(steerFilter_.rows - 1, j + r2 / 2))));
			int num_ = temp_.cols * temp_.rows;

			//
			cv::Mat subSrc_a = a(Rect(cv::Point(max(0, i - r2 / 2), max(0, j - r2 / 2)),
				cv::Point(min(steerFilter_.cols - 1, i + r2 / 2), min(steerFilter_.rows - 1, j + r2 / 2))));

			cv::Mat subres_a = temp_.mul(subSrc_a);
			double val_a = sum(subres_a)[0] * 1.0 / sum(temp_)[0];
			mean_a.at<float>(j, i) = val_a;

			//
			cv::Mat subSrc_b = b(Rect(cv::Point(max(0, i - r2 / 2), max(0, j - r2 / 2)),
				cv::Point(min(steerFilter_.cols - 1, i + r2 / 2), min(steerFilter_.rows - 1, j + r2 / 2))));

			cv::Mat subres_b = temp_.mul(subSrc_b);
			double val_b = sum(sum(subres_b))[0] * 1.0 / sum(temp_)[0];
			mean_b.at<float>(j, i) = val_b;
		}
	}

	//todo

	Mat filteredImg = mean_a.mul(guidedImg) + mean_b;
	return filteredImg;
}

//wgif + bil for a,b
cv::Mat getGuidedFilter_ours_gif(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h)
{
	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid, meanGuid2;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	boxFilter(guidedImg, meanGuid2, CV_32F, Size(r2, r2));

	Mat meanP, meanP2;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));
	boxFilter(inputP, meanP2, CV_32F, Size(r2, r2));

	cv::Mat corrGuid, corrGuid2;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, Size(r, r));
	boxFilter(guidedImg.mul(guidedImg), corrGuid2, CV_32F, Size(r2, r2));

	cv::Mat corrGuidP;
	boxFilter(guidedImg.mul(inputP), corrGuidP, CV_32F, Size(r, r));

	cv::Mat varGuid, varGuid2;
	varGuid = corrGuid - meanGuid.mul(meanGuid);
	varGuid2 = corrGuid2 - meanGuid2.mul(meanGuid2);

	cv::Mat temp = varGuid2 + namuda;
	cv::Mat gammaGuid = temp * sum(1.0 / temp)[0] / (temp.rows * temp.cols);

	cv::Mat covGuidP;
	covGuidP = corrGuidP - meanGuid.mul(meanP);

	cv::Mat a = covGuidP / (varGuid + eps / gammaGuid);
	cv::Mat b = meanP - a.mul(meanGuid);


	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);
	//mean_a = getGuidedFilter(guidedImg, a, r2, eps);
	//mean_b = getGuidedFilter(guidedImg, b, r2, eps);
	{
		bilateralFilter(a, mean_a, r2, 2, 3);
		bilateralFilter(b, mean_b, r2, 2, 3);
		a = mean_a;
		b = mean_b;
	}

	Mat filteredImg = mean_a.mul(guidedImg) + mean_b;
	return filteredImg;
}

//gif + bil for a,b
cv::Mat getGuidedFilter_ours_gif2(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h)
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

	std::vector<Mat> guidedImg_split;
	cv::split(guidedImg, guidedImg_split);

	Mat corrGuidP;
	boxFilter(guidedImg.mul(inputP), corrGuidP, CV_32F, Size(r, r));

	Mat corrGuid;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, Size(r, r));

	Mat varGuid;
	varGuid = corrGuid - meanGuid.mul(meanGuid);

	Mat meanGuidmulP;
	meanGuidmulP = meanGuid.mul(meanP);

	Mat covGuidP;
	covGuidP = corrGuidP - meanGuidmulP;

	Mat a = covGuidP / (varGuid + eps);
	Mat b = meanP - a.mul(meanGuid);

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);
	//{
	//	cv::Mat beta_a, beta_b;
	//	mean_a = getGuidedFilter_egif(guidedImg, a, r, eps, beta_a);
	//	mean_b = getGuidedFilter_egif(guidedImg, b, r, eps, beta_b);
	//}
	{
		bilateralFilter(a, mean_a, r2, 2, 3);
		bilateralFilter(b, mean_b, r2, 2, 3);
		//a = mean_a;
		//b = mean_b;
	}

	Mat filteredImg = mean_a.mul(guidedImg) + mean_b;
	return filteredImg;
}

//egif+bil
cv::Mat getGuidedFilter_ours_gif3(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h, cv::Mat& beta_)
{
	double eps0 = 0.001 * 0.001;

	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	double mean2VarGG = sum(varGuidGuid)[0] / (varGuidGuid.cols * varGuidGuid.rows);
	Mat a = covGuidP / (varGuidGuid + mean2VarGG * eps + eps0);
	Mat b = meanP - a.mul(meanGuid);

	//boxFilter(a, a, CV_32F, Size(r, r));
	//boxFilter(b, b, CV_32F, Size(r, r));

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);
	//{
	//	cv::Mat beta_a, beta_b;
	//	//mean_a = getGuidedFilter_egif(guidedImg, a, r, eps, beta_a);
	//	//mean_b = getGuidedFilter_egif(guidedImg, b, r, eps, beta_b);
	//	mean_a = getGuidedFilter_egif(a, a, r, eps, beta_a);
	//	mean_b = getGuidedFilter_egif(b, b, r, eps, beta_b);

	//	cv::Mat temp_a = guidedImg - a;
	//	temp_a.convertTo(temp_a, beta_a.type());
	//	a = temp_a.mul(beta_a) + mean_a;

	//	cv::Mat temp_b = guidedImg - b;
	//	temp_b.convertTo(temp_b, beta_b.type());
	//	b = temp_b.mul(beta_b) + mean_b;
	//}

	//{
	//	mean_a = getGuidedFilter(a, a, r, eps);
	//	mean_b = getGuidedFilter(b, b, r, eps);
	//	a = mean_a;
	//	b = mean_b;
	//}
	{
		bilateralFilter(a, mean_a, r, 2, 3);
		bilateralFilter(b, mean_b, r, 2, 3);
		a = mean_a;
		b = mean_b;
	}
	double gamma = 1.0;
	cv::Mat beta(a.size(), CV_32FC1);
	beta = a / (1 - a);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			beta.at<float>(i, j) = pow(beta.at<float>(i, j), gamma);
		}
	}
	beta_ = beta.clone();

	Mat filteredImg = a.mul(guidedImg) + b;
	return filteredImg;
}

//GIF_EGIF
cv::Mat getGuidedFilter_ours_gif4(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h,
	cv::Mat& beta_)
{
	double eps0 = 0.001 * 0.001;

	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	double mean2VarGG = sum(varGuidGuid)[0] / (varGuidGuid.cols * varGuidGuid.rows);
	Mat a = covGuidP / (varGuidGuid + mean2VarGG * eps + eps0);
	Mat b = meanP - a.mul(meanGuid);

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);
	{
		bilateralFilter(a, mean_a, r, 2, 3);
		bilateralFilter(b, mean_b, r, 2, 3);
	
		double minVal_a, minVal_b, maxVal_a, maxVal_b;
		minMaxLoc(mean_a, &minVal_a, &maxVal_a, NULL, NULL);
		minMaxLoc(mean_b, &minVal_b, &maxVal_b, NULL, NULL);
	
		mean_a = getGuidedFilter(a, a, r, eps);
		mean_b = getGuidedFilter(b, b, r, eps);
		normalize(mean_a, a, minVal_a, maxVal_a, NORM_MINMAX);
		normalize(mean_b, b, minVal_b, maxVal_b, NORM_MINMAX);
	
	}

	//{
	//	cv::Mat a_a, a_b;
	//	getGIF(a, a, r, eps, mean_a, a_a, a_b);
	//	cv::Mat b_a, b_b;
	//	getGIF(b, b, r, eps, mean_b, b_a, b_b);
	//
	//	a = mean_a;
	//	b = a_b + b_b;
	//}

	double gamma = 1.0;
	cv::Mat beta(a.size(), CV_32FC1);
	beta = a / (1 - a);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			beta.at<float>(i, j) = pow(beta.at<float>(i, j), gamma);
		}
	}
	beta_ = beta.clone();

	Mat filteredImg = a.mul(guidedImg) + b;
	return filteredImg;
}

//WGIF_EGIF
cv::Mat getGuidedFilter_ours_gif5(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h,
	cv::Mat& beta_)
{
	double eps0 = 0.001 * 0.001;

	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	double mean2VarGG = sum(varGuidGuid)[0] / (varGuidGuid.cols * varGuidGuid.rows);
	Mat a = covGuidP / (varGuidGuid + mean2VarGG * eps + eps0);
	Mat b = meanP - a.mul(meanGuid);

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);
	{
		bilateralFilter(a, mean_a, r, 2, 3);
		bilateralFilter(b, mean_b, r, 2, 3);
	
		double minVal_a, minVal_b, maxVal_a, maxVal_b;
		minMaxLoc(mean_a, &minVal_a, &maxVal_a, NULL, NULL);
		minMaxLoc(mean_b, &minVal_b, &maxVal_b, NULL, NULL);
	
		mean_a = getGuidedFilter_wgif(a, a, r, eps);
		mean_b = getGuidedFilter_wgif(b, b, r, eps);
		normalize(mean_a, a, minVal_a, maxVal_a, NORM_MINMAX);
		normalize(mean_b, b, minVal_b, maxVal_b, NORM_MINMAX);
	}

	//{
	//	cv::Mat a_a, a_b;
	//	getWGIF(a, a, r, eps, mean_a, a_a, a_b);
	//	cv::Mat b_a, b_b;
	//	getWGIF(b, b, r, eps, mean_b, b_a, b_b);
	//
	//	a = mean_a;
	//	b = a_b + b_b;
	//}

	double gamma = 1.0;
	cv::Mat beta(a.size(), CV_32FC1);
	beta = a / (1 - a);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			beta.at<float>(i, j) = pow(beta.at<float>(i, j), gamma);
		}
	}
	beta_ = beta.clone();

	Mat filteredImg = a.mul(guidedImg) + b;
	return filteredImg;
}


//EGIF_EGIF
cv::Mat getGuidedFilter_ours_gif6(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h,
	cv::Mat& beta_)
{
	double eps0 = 0.001 * 0.001;

	Size imgSize = guidedImg.size();

	cv::normalize(guidedImg, guidedImg, 0, 1, NORM_MINMAX, CV_32F);
	cv::normalize(inputP, inputP, 0, 1, NORM_MINMAX, CV_32F);

	Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, Size(r, r));
	Mat meanP;
	boxFilter(inputP, meanP, CV_32F, Size(r, r));

	Mat meanGuidP;
	boxFilter(guidedImg.mul(inputP), meanGuidP, CV_32F, Size(r, r));

	Mat covGuidP;
	covGuidP = meanGuidP - meanGuid.mul(meanP);

	Mat meanGuidGuid;
	boxFilter(guidedImg.mul(guidedImg), meanGuidGuid, CV_32F, Size(r, r));
	Mat varGuidGuid;
	varGuidGuid = meanGuidGuid - meanGuid.mul(meanGuid);

	double mean2VarGG = sum(varGuidGuid)[0] / (varGuidGuid.cols * varGuidGuid.rows);
	Mat a = covGuidP / (varGuidGuid + mean2VarGG * eps + eps0);
	Mat b = meanP - a.mul(meanGuid);

	cv::Mat mean_a(a.size(), CV_32FC1);
	cv::Mat mean_b(b.size(), CV_32FC1);
	{
		bilateralFilter(a, mean_a, r, 2, 3);
		bilateralFilter(b, mean_b, r, 2, 3);
	
		double minVal_a, minVal_b, maxVal_a, maxVal_b;
		minMaxLoc(mean_a, &minVal_a, &maxVal_a, NULL, NULL);
		minMaxLoc(mean_b, &minVal_b, &maxVal_b, NULL, NULL);
	
		cv::Mat beta_a, beta_b;
		mean_a = getGuidedFilter_egif(a, a, r, eps, beta_a);
		mean_b = getGuidedFilter_egif(b, b, r, eps, beta_b);
		normalize(mean_a, a, minVal_a, maxVal_a, NORM_MINMAX);
		normalize(mean_b, b, minVal_b, maxVal_b, NORM_MINMAX);
	}

	//{
	//	cv::Mat beta_a, beta_b;
	//	cv::Mat a_a, a_b;
	//	getEGIF(a, a, r, eps, beta_a, mean_a, a_a, a_b);
	//	cv::Mat b_a, b_b;
	//	getEGIF(b, b, r, eps, beta_b, mean_b, b_a, b_b);
	//
	//	a = mean_a;
	//	b = a_b + b_b;
	//}

	double gamma = 1.0;
	cv::Mat beta(a.size(), CV_32FC1);
	beta = a / (1 - a);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			beta.at<float>(i, j) = pow(beta.at<float>(i, j), gamma);
		}
	}
	beta_ = beta.clone();

	Mat filteredImg = a.mul(guidedImg) + b;
	return filteredImg;
}

void test_gifs(cv::Mat src)
{
	double r = 35;
	double lambda = 0.01;
	double r2 = 15;
	double h = 0.5;
	double eps = 0.01;

	vector<cv::Mat> Img_cns_dst(3), Img_cns_src(3);

	if (!src.empty())
	{
		split(src, Img_cns_src);
	}
	else
	{
		cout << "Source Image Error" << endl;
		return;
	}
	vector<double> minVal(3),  maxVal(3);
	minMaxLoc(Img_cns_src[0], &minVal[0], &maxVal[0], NULL, NULL);
	minMaxLoc(Img_cns_src[1], &minVal[1], &maxVal[1], NULL, NULL);
	minMaxLoc(Img_cns_src[2], &minVal[2], &maxVal[2], NULL, NULL);

	vector<int> minVal_(3), maxVal_(3);
	minVal_[0] = (int)minVal[0]; maxVal_[0] = (int)maxVal[0];
	minVal_[1] = (int)minVal[1]; maxVal_[1] = (int)maxVal[1];
	minVal_[2] = (int)minVal[2]; maxVal_[2] = (int)maxVal[2];


	//gif_egif
	cv::Mat z_gif;
	Img_cns_dst[0] = getGuidedFilter(Img_cns_src[0], Img_cns_src[0], r, lambda);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter(Img_cns_src[1], Img_cns_src[1], r, lambda);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter(Img_cns_src[2], Img_cns_src[2], r, lambda);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_gif);
	z_gif.convertTo(z_gif, CV_8UC3);

	//wgif
	cv::Mat z_wgif;
	Img_cns_dst[0] = getGuidedFilter_wgif(Img_cns_src[0], Img_cns_src[0], r, lambda);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_wgif(Img_cns_src[1], Img_cns_src[1], r, lambda);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_wgif(Img_cns_src[2], Img_cns_src[2], r, lambda);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_wgif);
	z_wgif.convertTo(z_wgif, CV_8UC3);

	//egif
	cv::Mat z_egif;
	cv::Mat beta;
	Img_cns_dst[0] = getGuidedFilter_egif(Img_cns_src[0], Img_cns_src[0], r, lambda, beta);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_egif(Img_cns_src[1], Img_cns_src[1], r, lambda, beta);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_egif(Img_cns_src[2], Img_cns_src[2], r, lambda, beta);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_egif);
	z_egif.convertTo(z_egif, CV_8UC3);

	//skwgif
	cv::Mat z_skwgif;
	cv::Mat steerFilterW;
	steerFilter(Img_cns_src[0], steerFilterW);
	Img_cns_dst[0] = getGuidedFilter_skwgif2(Img_cns_src[0], Img_cns_src[0], steerFilterW, r, r2, eps, lambda, h);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_skwgif2(Img_cns_src[1], Img_cns_src[1], steerFilterW, r, r2, eps, lambda, h);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_skwgif2(Img_cns_src[2], Img_cns_src[2], steerFilterW, r, r2, eps, lambda, h);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_skwgif);
	z_skwgif.convertTo(z_skwgif, CV_8UC3);

	//ours
	//ours gif
	cv::Mat z_ours_gif1;
	Img_cns_dst[0] = getGuidedFilter_ours_gif(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif1);
	z_ours_gif1.convertTo(z_ours_gif1, CV_8UC3);

	//ours_wgif
	cv::Mat z_ours_gif2;
	Img_cns_dst[0] = getGuidedFilter_ours_gif2(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif2(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif2(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif2);
	//normalize(z_ours_gif2, z_ours_gif2, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif2.convertTo(z_ours_gif2, CV_8UC3);

	//ours_egif//BF_EGIF
	cv::Mat z_ours_gif3;
	cv::Mat beta_;
	Img_cns_dst[0] = getGuidedFilter_ours_gif3(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif3(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif3(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif3);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif3.convertTo(z_ours_gif3, CV_8UC3);

	//ours_egif//GIF_EGIF
	cv::Mat z_ours_gif4;
	cv::Mat beta_2;
	Img_cns_dst[0] = getGuidedFilter_ours_gif4(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_2);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif4(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_2);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif4(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_2);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif4);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif4.convertTo(z_ours_gif4, CV_8UC3);

	//ours_egif//WGIF_EGIF
	cv::Mat z_ours_gif5;
	cv::Mat beta_3;
	Img_cns_dst[0] = getGuidedFilter_ours_gif5(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_3);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif5(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_3);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif5(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_3);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif5);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif5.convertTo(z_ours_gif5, CV_8UC3);

	//ours_egif//EGIF_EGIF
	cv::Mat z_ours_gif6;
	cv::Mat beta_4;
	Img_cns_dst[0] = getGuidedFilter_ours_gif6(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_4);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif6(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_4);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif6(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_4);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif6);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif6.convertTo(z_ours_gif6, CV_8UC3);


	////tulips
	//Rect roi_(cv::Point(606, 445), cv::Point(794, 677));
	//Rect roi_2(cv::Point(662, 453), cv::Point(781, 571));
	//Rect roi_3(cv::Point(580, 445), cv::Point(698, 600));
	////Rect roi_(cv::Point(606, 436), cv::Point(639, 492));
	////Rect roi_2(cv::Point(176, 106), cv::Point(280, 190));
	////Rect roi_3(cv::Point(511, 959), cv::Point(554, 1052));

	////cat
	//Rect roi_(cv::Point(106, 46), cv::Point(154, 61));
	//Rect roi_2(cv::Point(32, 173), cv::Point(66, 217));
	//Rect roi_3(cv::Point(59, 240), cv::Point(85, 266));

	//toy
	Rect roi_(cv::Point(258, 286), cv::Point(322, 352));
	Rect roi_2(cv::Point(377, 326), cv::Point(430, 368));
	Rect roi_3(cv::Point(464, 441), cv::Point(511, 498));

	vector<std::string> names(11);
	names[0] = "gif";
	names[1] = "wgif";
	names[2] = "egif";
	names[3] = "skwgif";
	names[4] = "ours_gif1";
	names[5] = "ours_gif2";
	names[6] = "ours_gif3";
	names[7] = "ours_gif4";
	names[8] = "ours_gif5";
	names[9] = "ours_gif6";
	names[10] = "src";

	vector<cv::Mat> imgs(11);
	imgs[0] = z_gif;
	imgs[1] = z_wgif;
	imgs[2] = z_egif;
	imgs[3] = z_skwgif;
	imgs[4] = z_ours_gif1;
	imgs[5] = z_ours_gif2;
	imgs[6] = z_ours_gif3;
	imgs[7] = z_ours_gif4;
	imgs[8] = z_ours_gif5;
	imgs[9] = z_ours_gif6;
	imgs[10] = src;

	for(int i = 0; i < 11; i++)
	{
		cv::imwrite("test/test_" + names[i] + ".jpg", imgs[i]);
		cv::imwrite("test/test_roi_" + names[i] + ".jpg", imgs[i](roi_));
		cv::imwrite("test/test_roi2_" + names[i] + ".jpg", imgs[i](roi_2));
		cv::imwrite("test/test_roi3_" + names[i] + ".jpg", imgs[i](roi_3));
		{
			cv::Mat img_ = imgs[i].clone();
			rectangle(img_, roi_, Scalar(0, 0, 255));
			cv::imwrite("test/test_full_roi_" + names[i] + ".jpg", img_);
		}
		{
			cv::Mat img_ = imgs[i].clone();
			rectangle(img_, roi_2, Scalar(0, 0, 255));
			cv::imwrite("test/test_full_roi2_" + names[i] + ".jpg", img_);
		}
		{
			cv::Mat img_ = imgs[i].clone();
			rectangle(img_, roi_3, Scalar(0, 0, 255));
			cv::imwrite("test/test_full_roi3_" + names[i] + ".jpg", img_);
		}

	}
}

void test_our_egifs(cv::Mat src)
{
	double r = 15;
	double lambda = 0.01;
	double r2 = 15;
	double h = 0.5;
	double eps = 0.01;

	vector<cv::Mat> Img_cns_dst(3), Img_cns_src(3);

	if (!src.empty())
	{
		split(src, Img_cns_src);
	}
	else
	{
		cout << "Source Image Error" << endl;
		return;
	}
	vector<double> minVal(3), maxVal(3);
	minMaxLoc(Img_cns_src[0], &minVal[0], &maxVal[0], NULL, NULL);
	minMaxLoc(Img_cns_src[1], &minVal[1], &maxVal[1], NULL, NULL);
	minMaxLoc(Img_cns_src[2], &minVal[2], &maxVal[2], NULL, NULL);

	vector<int> minVal_(3), maxVal_(3);
	minVal_[0] = (int)minVal[0]; maxVal_[0] = (int)maxVal[0];
	minVal_[1] = (int)minVal[1]; maxVal_[1] = (int)maxVal[1];
	minVal_[2] = (int)minVal[2]; maxVal_[2] = (int)maxVal[2];

	//ours_egif//BF_EGIF
	cv::Mat z_ours_gif3;
	cv::Mat beta_;
	Img_cns_dst[0] = getGuidedFilter_ours_gif3(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif3(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif3(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif3);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif3.convertTo(z_ours_gif3, CV_8UC3);

	//ours_egif//GIF_EGIF
	cv::Mat z_ours_gif4;
	cv::Mat beta_2;
	Img_cns_dst[0] = getGuidedFilter_ours_gif4(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_2);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif4(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_2);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif4(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_2);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif4);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif4.convertTo(z_ours_gif4, CV_8UC3);

	//ours_egif//WGIF_EGIF
	cv::Mat z_ours_gif5;
	cv::Mat beta_3;
	Img_cns_dst[0] = getGuidedFilter_ours_gif5(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_3);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif5(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_3);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif5(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_3);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif5);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif5.convertTo(z_ours_gif5, CV_8UC3);

	//ours_egif//EGIF_EGIF
	cv::Mat z_ours_gif6;
	cv::Mat beta_4;
	Img_cns_dst[0] = getGuidedFilter_ours_gif6(Img_cns_src[0], Img_cns_src[0], r, r2, eps, lambda, h, beta_4);
	normalize(Img_cns_dst[0], Img_cns_dst[0], minVal_[0], maxVal_[0], NORM_MINMAX);
	Img_cns_dst[1] = getGuidedFilter_ours_gif6(Img_cns_src[1], Img_cns_src[1], r, r2, eps, lambda, h, beta_4);
	normalize(Img_cns_dst[1], Img_cns_dst[1], minVal_[1], maxVal_[1], NORM_MINMAX);
	Img_cns_dst[2] = getGuidedFilter_ours_gif6(Img_cns_src[2], Img_cns_src[2], r, r2, eps, lambda, h, beta_4);
	normalize(Img_cns_dst[2], Img_cns_dst[2], minVal_[2], maxVal_[2], NORM_MINMAX);
	merge(Img_cns_dst, z_ours_gif6);
	//normalize(z_ours_gif3, z_ours_gif3, minVal_, maxVal_, NORM_MINMAX);
	z_ours_gif6.convertTo(z_ours_gif6, CV_8UC3);


	////tulips
	//Rect roi_(cv::Point(606, 445), cv::Point(794, 677));
	//Rect roi_2(cv::Point(662, 453), cv::Point(781, 571));
	//Rect roi_3(cv::Point(580, 445), cv::Point(698, 600));
	Rect roi_(cv::Point(606, 436), cv::Point(639, 492));
	Rect roi_2(cv::Point(176, 106), cv::Point(280, 190));
	Rect roi_3(cv::Point(511, 959), cv::Point(554, 1052));

	////cat
	//Rect roi_(cv::Point(106, 46), cv::Point(154, 61));
	//Rect roi_2(cv::Point(32, 173), cv::Point(66, 217));
	//Rect roi_3(cv::Point(59, 240), cv::Point(85, 266));

	////toy
	//Rect roi_(cv::Point(258, 286), cv::Point(322, 352));
	//Rect roi_2(cv::Point(377, 326), cv::Point(430, 368));
	//Rect roi_3(cv::Point(464, 441), cv::Point(511, 498));

	vector<std::string> names(5);
	names[0] = "BF_EGIF";
	names[1] = "GIF_EGIF";
	names[2] = "WGIF_EGIF";
	names[3] = "EGIF_EGIF";
	names[4] = "src";

	vector<cv::Mat> imgs(5);
	imgs[0] = z_ours_gif3;
	imgs[1] = z_ours_gif4;
	imgs[2] = z_ours_gif5;
	imgs[3] = z_ours_gif6;
	imgs[4] = src;

	for (int i = 0; i < 5; i++)
	{
		cv::imwrite("test/test_" + names[i] + ".jpg", imgs[i]);
		cv::imwrite("test/test_roi_" + names[i] + ".jpg", imgs[i](roi_));
		cv::imwrite("test/test_roi2_" + names[i] + ".jpg", imgs[i](roi_2));
		cv::imwrite("test/test_roi3_" + names[i] + ".jpg", imgs[i](roi_3));
		{
			cv::Mat img_ = imgs[i].clone();
			rectangle(img_, roi_, Scalar(0, 0, 255));
			cv::imwrite("test/test_full_roi_" + names[i] + ".jpg", img_);
		}
		{
			cv::Mat img_ = imgs[i].clone();
			rectangle(img_, roi_2, Scalar(0, 0, 255));
			cv::imwrite("test/test_full_roi2_" + names[i] + ".jpg", img_);
		}
		{
			cv::Mat img_ = imgs[i].clone();
			rectangle(img_, roi_3, Scalar(0, 0, 255));
			cv::imwrite("test/test_full_roi3_" + names[i] + ".jpg", img_);
		}

	}
}

cv::Mat addSaltNoise(cv::Mat src, int n)
{
	Mat result = src.clone();
	for (int k = 0; k < n; k++)
	{
		//随机选取行列值
		int i = rand() % result.cols;
		int j = rand() % result.rows;
		if (result.channels() == 1)
		{
			result.at<uchar>(j, i) = 255;
		}
		else
		{
			result.at<Vec3b>(j, i)[0] = 255;
			result.at<Vec3b>(j, i)[1] = 255;
			result.at<Vec3b>(j, i)[2] = 255;
		}

	}
	return result;
}

double generateGaussianNoise(double mu, double sigma)
{
	//定义最小值
	double epsilon = numeric_limits<double>::min();
	double z0 = 0, z1 = 0;
	bool flag = false;
	flag = !flag;
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	do
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

cv::Mat addGaussianNoise(cv::Mat src)
{
	Mat result = src.clone();
	int channels = result.channels();
	int nRows = result.rows;
	int nCols = result.cols*channels;
	if (result.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int val = result.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			result.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return result;
}


