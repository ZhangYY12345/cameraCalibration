#pragma once
#include <opencv2/opencv.hpp>
#include <CImg.h>


#define cimg_display_type 0

typedef cimg_library::CImg<unsigned char> Image;
typedef cimg_library::CImg<int> IImage;
typedef cimg_library::CImg<double> DImage;

struct convdata
{
	double G2a;
	double G2b;
	double G2c;
	double H2a;
	double H2b;
	double H2c;
	double H2d;
};
typedef cimg_library::CImg < convdata > ConvImage;

template<class T> struct Matrix : cimg_library::CImg<T>
{
	Matrix() : cimg_library::CImg<T>() {}
	Matrix(int dimx, int dimy, int dimz = 1, int dimv = 1) : cimg_library::CImg<T>(dimx, dimy, dimz, dimv) {}
};


void convertImg2Mat(Image& src, cv::Mat& dst);
void convertIImg2Mat(IImage& src, cv::Mat& dst);
void convertDImg2Mat(DImage& src, cv::Mat& dst);

void convertMat2Img(cv::Mat& src, Image& dst);
void convertMat2IImg(cv::Mat& src, IImage& dst);
void convertMat2DImg(cv::Mat& src, DImage& dst);

void steerFilter(cv::Mat src, cv::Mat& dst);

//void SteerImage(cv::Mat& src, cv::Mat& steered, cv::Mat& Ostrength);


///------------------------based on matlab code 

struct twodouble {
	twodouble(double a, double b)
	{
		this->begin = a;
		this->end = b;
	};
	double begin;
	double end;
};
void meshgrid(const struct twodouble &_Input2X, const struct twodouble &_Input2Y, float _size, cv::Mat &_OutputMatX, cv::Mat &_OutputMatY);
cv::Mat fspecialLoG(int WinSize, int sigma);
void downsample2(cv::Mat src, int factor, cv::Mat& dst);
void ckr2_regular(cv::Mat& z, cv::Mat& zx1, cv::Mat& zx2, cv::Mat srcImg, double h, int upS, int winSize);
void steering(std::vector<std::vector<cv::Mat> >& C, cv::Mat zx, cv::Mat zy, cv::Mat I, int winSize, double lambda, double alpha);