#pragma once
#include <opencv2/opencv.hpp>

enum GIF_TYPE
{
	GIF,
	WGIF,
	EGIF,
	SKWGIF,
	OURS_GIF,
	OURS_GIF2,
	OURS_GIF3,
};

//gif
cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg);
cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);
void getGIF(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& filteredImg, cv::Mat& a, cv::Mat& b);

// egif
cv::Mat getGuidedFilter_egif(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& beta_);
cv::Mat getGuidedFilter_egif_3cn(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);
void getEGIF(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& beta_, cv::Mat& filteredImg, cv::Mat& a, cv::Mat& b);

//wgif
cv::Mat edgeAwareWeight(cv::Mat guidedImg);
cv::Mat getGuidedFilter_wgif(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);
cv::Mat getGuidedFilter_wgif_3cn(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);
void getWGIF(cv::Mat guidedImg, cv::Mat inputP, int r, double eps, cv::Mat& filteredImg, cv::Mat& a, cv::Mat& b);

//skWGIF
cv::Mat getGuidedFilter_skwgif(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h);
cv::Mat getGuidedFilter_skwgif2(cv::Mat guidedImg, cv::Mat inputP, cv::Mat steerFilterW,int r, int r2, double eps, double namuda, double h);

//ours
cv::Mat getGuidedFilter_ours_gif(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h);
cv::Mat getGuidedFilter_ours_gif2(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h);
cv::Mat getGuidedFilter_ours_gif3(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h, cv::Mat& beta_);

//¼¶ÁªEGIFs
cv::Mat getGuidedFilter_ours_gif4(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h, cv::Mat& beta_);
cv::Mat getGuidedFilter_ours_gif5(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h, cv::Mat& beta_);
cv::Mat getGuidedFilter_ours_gif6(cv::Mat guidedImg, cv::Mat inputP, int r, int r2, double eps, double namuda, double h, cv::Mat& beta_);


//
void test_gifs(cv::Mat src);
void test_our_egifs(cv::Mat src);
cv::Mat addSaltNoise(cv::Mat src, int n);
double generateGaussianNoise(double mu, double sigma);
cv::Mat addGaussianNoise(cv::Mat src);