#include "steerFilter.h"

double G2afunc(int x, int y)
{
	double xs, ys;

	xs = (double)(x * x);
	ys = (double)(y * y);
	ys = exp(-(xs + ys));

	return 0.9213 * ((2 * xs) - 1) * ys;

}

double G2bfunc(int x, int y)
{
	double xs, ys;

	xs = (double)(x * x);
	ys = (double)(y * y);

	ys = exp(-(xs + ys));

	return 1.843 * x * y * ys;
}

double H2afunc(int x, int y)
{
	double xs, ys;

	xs = (double)(x * x);
	ys = (double)(y * y);

	ys = exp(-(xs + ys));

	return 0.9780 * (-2.254 * x + x * xs) * ys;
}

double H2bfunc(int x, int y)
{
	double xs, ys;

	xs = (double)(x * x);
	ys = (double)(y * y);

	ys = exp(-(xs + ys));

	return 0.9780 * (-0.7515 + xs) * y * ys;
}

double H2cfunc(int x, int y)
{
	double xs, ys;

	xs = (double)(x * x);
	ys = (double)(y * y);

	ys = exp(-(xs + ys));

	return 0.9780 * (-0.7515 + y * y) * x * ys;
}

double G2cfunc(int x, int y)
{

	double xs, ys;

	xs = (double)(x * x);
	ys = (double)(y * y);

	xs = exp(-(xs + ys));

	return 0.9213 * ((2 * ys) - 1) * xs;
}

double H2dfunc(int x, int y)
{

	double xs, ys;

	xs = (double)pow(x, 2);
	ys = (double)pow(y, 2);

	xs = exp(-(xs + ys));

	return 0.9780 * (-2.254 * y + y * ys) * xs;
}


static ConvImage conv;
static DImage C2, C3;

#define NUM_ANGLES	32
double angles[NUM_ANGLES] = { 0,
	31.0 / 16.0 * cimg_library::cimg::PI, 30.0 / 16.0 * cimg_library::cimg::PI, 29.0 / 16.0 * cimg_library::cimg::PI,
	28.0 / 16.0 * cimg_library::cimg::PI, 27.0 / 16.0 * cimg_library::cimg::PI,
	26.0 / 16.0 * cimg_library::cimg::PI, 25.0 / 16.0 * cimg_library::cimg::PI, 24.0 / 16.0 * cimg_library::cimg::PI,
	23.0 / 16.0 * cimg_library::cimg::PI, 22.0 / 16.0 * cimg_library::cimg::PI,
	21.0 / 16.0 * cimg_library::cimg::PI, 20.0 / 16.0 * cimg_library::cimg::PI, 19.0 / 16.0 * cimg_library::cimg::PI,
	18.0 / 16.0 * cimg_library::cimg::PI, 17.0 / 16.0 * cimg_library::cimg::PI,
	16.0 / 16.0 * cimg_library::cimg::PI, 15.0 / 16.0 * cimg_library::cimg::PI, 14.0 / 16.0 * cimg_library::cimg::PI,
	13.0 / 16.0 * cimg_library::cimg::PI, 12.0 / 16.0 * cimg_library::cimg::PI,
	11.0 / 16.0 * cimg_library::cimg::PI, 10.0 / 16.0 * cimg_library::cimg::PI, 9.0 / 16.0 * cimg_library::cimg::PI,
	8.0 / 16.0 * cimg_library::cimg::PI, 7.0 / 16.0 * cimg_library::cimg::PI,
	6.0 / 16.0 * cimg_library::cimg::PI, 5.0 / 16.0 * cimg_library::cimg::PI, 4.0 / 16.0 * cimg_library::cimg::PI,
	3.0 / 16.0 * cimg_library::cimg::PI,
	2.0 / 16.0 * cimg_library::cimg::PI,
	1.0 / 16.0 * cimg_library::cimg::PI
};

static convdata prefixes[NUM_ANGLES];

void initsteer()
{
	for (int i = 0; i < NUM_ANGLES; i++) {
		double cosa = cos(angles[i]);
		double sina = sin(angles[i]);

		prefixes[i].G2a = cosa * cosa;
		prefixes[i].G2b = -2.0 * cosa * sina;
		prefixes[i].G2c = sina * sina;
		prefixes[i].H2a = cosa * cosa * cosa;
		prefixes[i].H2b = -3.0 * cosa * cosa * sina;
		prefixes[i].H2c = 3.0 * cosa * sina * sina;
		prefixes[i].H2d = -sina * sina * sina;
	}
}

double steer(int ai, int x, int y)
{
	double H2, G2, G2a, G2b, G2c, H2a, H2b, H2c, H2d;

	G2a = prefixes[ai].G2a * conv(x, y).G2a;
	G2b = prefixes[ai].G2b * conv(x, y).G2b;
	G2c = prefixes[ai].G2c * conv(x, y).G2c;
	G2 = G2a + G2b + G2c;
	H2a = prefixes[ai].H2a * conv(x, y).H2a;
	H2b = prefixes[ai].H2b * conv(x, y).H2b;
	H2c = prefixes[ai].H2c * conv(x, y).H2c;
	H2d = prefixes[ai].H2d * conv(x, y).H2d;
	H2 = H2a + H2b + H2c + H2d;

	return G2 * G2 + H2 * H2;
}

void convertImg2Mat(Image& src, cv::Mat& dst)
{
	cv::Mat temp(src._height, src._width, CV_8UC1);
	for (int y = 0; y < temp.rows; y++) {
		for (int x = 0; x < temp.cols; x++) {
			temp.at<uchar>(y, x) = src(x, y);
		}
	}
	dst = temp.clone();
}

void convertIImg2Mat(IImage& src, cv::Mat& dst)
{
	cv::Mat temp(src._height, src._width, CV_32SC1);
	for (int y = 0; y < temp.rows; y++) {
		for (int x = 0; x < temp.cols; x++) {
			temp.at<int>(y, x) = src(x, y);
		}
	}

	dst = temp.clone();
}

void convertDImg2Mat(DImage& src, cv::Mat& dst)
{
	cv::Mat temp(src._height, src._width, CV_64FC1);
	for (int y = 0; y < temp.rows; y++) {
		for (int x = 0; x < temp.cols; x++) {
			temp.at<double>(y, x) = src(x, y);
		}
	}

	dst = temp.clone();
}

void convertMat2Img(cv::Mat& src, Image& dst)
{
	if (src.type() != CV_8U)
	{
		src.convertTo(src, CV_8U);
	}

	dst._width = src.cols;
	dst._height = src.rows;
	dst._depth = 1;
	dst._spectrum = src.channels();
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst(x, y) = src.at<uchar>(y, x);
		}
	}

}

void convertMat2IImg(cv::Mat& src, IImage& dst)
{
	if (src.type() != CV_32S)
	{
		src.convertTo(src, CV_32S);
	}

	dst._width = src.cols;
	dst._height = src.rows;
	dst._depth = 1;
	dst._spectrum = src.channels();

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst(x, y) = src.at<int>(y, x);
		}
	}

}

void convertMat2DImg(cv::Mat& src, DImage& dst)
{
	if (src.type() != CV_64F)
	{
		src.convertTo(src, CV_64F);
	}
	dst._width = src.cols;
	dst._height = src.rows;
	dst._depth = 1;
	dst._spectrum = src.channels();
	dst._data = (double*)malloc(dst._width * dst._height * sizeof(double));

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst(x, y, 0) = src.at<double>(y, x);
		}
	}
}

void steerFilter(cv::Mat src, cv::Mat& dst)
{
	cv::Mat tempSrc;
	cv::copyMakeBorder(src, tempSrc, 6, 6, 6, 6, cv::BORDER_REFLECT);

	DImage logdata, steered, Ostrength;
	DImage image;
	convertMat2DImg(tempSrc, image);

	int y, x, j, k;
	double
		G2akernel[11][11],
		G2bkernel[11][11],
		H2akernel[11][11],
		H2bkernel[11][11],
		G2ckernel[11][11],
		H2ckernel[11][11],
		H2dkernel[11][11], max, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

	/*allocate memory */

	conv = ConvImage(image._width, image._height);
	C2 = DImage(image._width, image._height);
	C3 = DImage(image._width, image._height);
	logdata = DImage(image._width, image._height);
	steered = DImage(image._width, image._height, 32);
	Ostrength = DImage(image._width, image._height);

	printf("point 1\n");

	/* put in gray form */
	for (y = 0; y < image._height; y++) {
		for (x = 0; x < image._width; x++) {
			//if (image(x, y, 0) == 0)
			//logdata(x, y) = log(0.1);
			//else
			//logdata(x, y) = log((double) image(x, y, 0));
			logdata(x, y) = image(x, y, 0);
		}
	}

	printf("point 2\n");

	/*make the kernels */
	for (y = -5; y <= 5; y++) {
		for (x = -5; x <= 5; x++) {
			G2akernel[y + 5][x + 5] = G2afunc(x, y);
			H2dkernel[y + 5][x + 5] = H2dfunc(x, y);
			G2bkernel[y + 5][x + 5] = G2bfunc(x, y);
			H2akernel[y + 5][x + 5] = H2afunc(x, y);
			H2bkernel[y + 5][x + 5] = H2bfunc(x, y);
			G2ckernel[y + 5][x + 5] = G2cfunc(x, y);
			H2ckernel[y + 5][x + 5] = H2cfunc(x, y);
		}
	}

	printf("point 4\n");

	/*convolve image with kernels */
	for (y = 6; y <= image._height - 6; y++) {
		for (x = 6; x <= image._width - 6; x++) {
			sum1 = 0.0;
			sum2 = 0.0;
			sum3 = 0.0;
			sum4 = 0.0;
			sum5 = 0.0;
			sum6 = 0.0;
			sum7 = 0.0;
			for (k = -5; k <= 5; k++) {
				for (j = -5; j <= 5; j++) {
					sum1 =
						sum1 + G2akernel[k + 5][j + 5] * logdata(x + j,
							y + k);
					sum2 =
						sum2 + G2bkernel[k + 5][j + 5] * logdata(x + j,
							y + k);
					sum3 =
						sum3 + H2akernel[k + 5][j + 5] * logdata(x + j,
							y + k);
					sum4 =
						sum4 + H2bkernel[k + 5][j + 5] * logdata(x + j,
							y + k);
					sum5 =
						sum5 + G2ckernel[k + 5][j + 5] * logdata(x + j,
							y + k);
					sum6 =
						sum6 + H2ckernel[k + 5][j + 5] * logdata(x + j,
							y + k);
					sum7 =
						sum7 + H2dkernel[k + 5][j + 5] * logdata(x + j,
							y + k);
				}
			}
			conv(x, y).G2a = sum1;
			conv(x, y).G2b = sum2;
			conv(x, y).G2c = sum5;
			conv(x, y).H2a = sum3;
			conv(x, y).H2b = sum4;
			conv(x, y).H2c = sum6;
			conv(x, y).H2d = sum7;
		}
	}

	printf("point 5\n");

	/*remember to take into acount image read upside down */

	initsteer();

	/*make arrays of responses steered to each angle */
	for (y = 6; y <= image._height - 7; y++) {
		for (x = 6; x <= image._width - 7; x++) {

			for (int i = 0; i < 32; i++) {
				steered(x, y, i) = steer(i, x, y);
			}
		}
	}

	printf("point 5.5\n");

	max = 0;
	for (y = 6; y <= steered._height - 7; y++) {
		for (x = 6; x <= steered._width - 7; x++) {
			for (j = 0; j < steered._depth; j++) {
				if (steered(x, y, j) > max)
					max = steered(x, y, j);
			}
		}
	}

	printf("point 6\n");

	/*to find areas with high contrast */
	/*compute C2 and C3 */
	for (y = 0; y < image._height; y++) {
		for (x = 0; x < image._width; x++) {
			C2(x, y) =
				0.5 * (pow(conv(x, y).G2a, 2) - pow(conv(x, y).G2c, 2)) +
				0.46875 * (pow(conv(x, y).H2a, 2) -
					pow(conv(x, y).H2d,
						2)) + 0.28125 * (pow(conv(x, y).H2b,
							2) - pow(conv(x,
								y).H2c,
								2)) +
				0.1875 * (conv(x, y).H2a * conv(x, y).H2c -
					conv(x, y).H2b * conv(x, y).H2d);

			C3(x, y) = -(conv(x, y).G2a * conv(x, y).G2b) -
				(conv(x, y).G2b * conv(x, y).G2c) -
				0.9375 * (conv(x, y).H2c * conv(x, y).H2d -
					conv(x, y).H2a * conv(x, y).H2b) -
				1.6875 * conv(x, y).H2b * conv(x, y).H2c -
				0.1875 * conv(x, y).H2a * conv(x, y).H2d;

		}
	}

	printf("point 7\n");

	/*Compute strength of dominant orientation at each pixel */
	max = 0;
	for (y = 6; y < image._height - 6; y++) {
		for (x = 6; x < image._width - 6; x++) {
			Ostrength(x, y) = sqrt(pow(C2(x, y), 2) + pow(C3(x, y), 2));
			if (Ostrength(x, y) > max)
				max = Ostrength(x, y);
		}
	}

	//make into grayscale
	for (y = 6; y < image._height - 6; y++) {
		for (x = 6; x < image._width - 6; x++) {
			image(x, y) = ((Ostrength(x, y) * 1.0 / max) * 255);
		}
	}
	//image of high contrast areas
	cv::Mat dst_temp;
	convertDImg2Mat(image, dst_temp);
	dst = dst_temp(cv::Rect(cv::Point(6, 6), cv::Point(src.cols + 6, src.rows + 6))).clone();

	//cv::Mat dst_e_4, dst_e2;
	//cv::Mat dst_thres_L = cv::Mat(dst.size(), dst.type(), cv::Scalar::all(1e-4));
	//cv::Mat dst_thres_G = cv::Mat(dst.size(), dst.type(), cv::Scalar::all(1e2));

	//cv::compare(dst, dst_thres_L, dst_e_4, cv::CMP_LE);
	//cv::compare(dst, dst_thres_G, dst_e2, cv::CMP_GE);

	//dst_thres_L.convertTo(dst_thres_L, CV_8UC1);
	//dst_thres_G.convertTo(dst_thres_G, CV_8UC1);

	for(int i_ = 0; i_ < dst.rows; i_++)
	{
		for(int j_ = 0; j_ < dst.cols; j_++)
		{
			if(dst.at<double>(i_, j_) > 120)
			{
				dst.at<double>(i_, j_) = 120;
			}
			if(dst.at<double>(i_, j_) < 1e-6)
			{
				dst.at<double>(i_, j_) = 1e-6;
			}
		}
	}
	//cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);

}
//void SteerImage(cv::Mat& src, cv::Mat& steered, cv::Mat& Ostrength)
//{
//	double pi_Gxx, pi_Gyy, pi_Gxy, t_Gxx, t_Gyy, t_Gxy, SUM;
//
//	theta = (-t * (pi / 180));
//
//
//	if(src.depth() != CV_32F)
//	{
//		src.convertTo(src, CV_32F);
//	}
//	cv::Mat img = src;
//	cv::Mat Gxx = cv::Mat(src.size(), CV_32FC1);
//	cv::Mat Gyy = cv::Mat(src.size(), CV_32FC1);
//	cv::Mat Gxy = cv::Mat(src.size(), CV_32FC1);
//	cvtColor(img, Gxx, cv::COLOR_BGR2GRAY);
//	cvtColor(img, Gyy, cv::COLOR_BGR2GRAY);
//	cvtColor(img, Gxy, cv::COLOR_BGR2GRAY);
//
//
//	cv::Mat Gxx_32F = cv::Mat(Gxx.size(), CV_32FC1);
//	Sobel(Gxx, Gxx_32F, 2, 0, 3);
//	//Should hav worked on 16S. change for loop from Gxx to Gxx_16S for 16S
//
//
//	cv::Mat Gyy_32F = cv::Mat(Gyy.size(), CV_32FC1);
//	Sobel(Gyy, Gyy_32F, 0, 2, 3);
//	//Should hav worked on 16S. change for loop from Gyy to Gyy_16S for 16S
//
//
//	cv::Mat Gxy_32F = cv::Mat(Gxy.size(), CV_32FC1); 
//	Sobel(Gxy, Gxy_32F, 2, 2, 3);
//	//Should hav worked on 16S. change for loop from Gxx to Gxy_16S for 16S
//
//	cv::Mat output_32F = cv::Mat(Gxy.size(), CV_32FC1);
//
//	for (int i = 0; i < Gxx.rows; i++)
//	{
//		for (int j = 0; j < Gxx.cols; j++)
//		{
//			pi_Gxx = Gxx.at<float>(i, j);
//			t_Gxx = pi_Gxx * cos(theta) * cos(theta);
//
//			pi_Gyy = Gyy.at<float>(i, j);
//			t_Gyy = pi_Gyy * sin(theta) * sin(theta);
//
//			pi_Gxy = Gxy.at<float>(i, j);
//			t_Gxy = pi_Gxy * cos(theta) * sin(theta);
//			SUM = abs(t_Gxx) + abs(t_Gyy) + abs(t_Gxy);
//
//			output_32F.at<float>(i, j) = SUM;
//
//			/*theta min and theta max calculation. to be done on 16S image.
//			process halted for the further clarification
//
//			float A = sqrt((pi_Gxx * pi_Gxx) - (2 *  pi_Gxx * pi_Gyy) + (pi_Gyy * pi_Gyy) + (4 * pi_Gxy));
//			double Tmin = atan((pi_Gxx-pi_Gyy-A)/(2*pi_Gxy));
//			printf("\nTmin = %f\n",Tmin);
//			double Tmin = atan((pi_Gxx-pi_Gyy+A)/(2*pi_Gxy));
//			printf("\nTmax = %f\n",Tmax);
//			*/
//		}
//	}
//}


//---------------------------------------------
//----------code from matlab
//---------------------------------------------
void meshgrid(const twodouble& _Input2X, const twodouble& _Input2Y, float _size, cv::Mat& _OutputMatX,
	cv::Mat& _OutputMatY)
{
	std::vector<float> t_x, t_y;
	double temp = 0.0f;

	// x
	if (_Input2X.begin == _Input2X.end) {
		for (double i = 0; i <= 1; i += _size) {
			t_x.push_back(_Input2X.begin);
		}
	}
	else if (_Input2X.begin < _Input2X.end) {
		for (double i = _Input2X.begin; i <= _Input2X.end;
			i += _size) {
			t_x.push_back(i);
		}
	}
	else {
		for (double i = _Input2X.begin; i >= _Input2X.end;
			i -= _size) {
			t_x.push_back(i);
		}
	}

	// y
	if (_Input2Y.begin == _Input2Y.end) {
		for (double j = 0; j <= 1; j += _size) {
			t_y.push_back(_Input2Y.begin);
		}
	}
	else if (_Input2Y.begin < _Input2Y.end) {
		for (double j = _Input2Y.begin; j <= _Input2Y.end;
			j += _size) {
			t_y.push_back(j);
		}
	}
	else {
		for (double j = _Input2Y.begin; j >= _Input2Y.end;
			j -= _size) {
			t_y.push_back(j);
		}
	}

	//	for (float j = ygv.start; j < ygv.end + (ygv.end - ygv.start)*0.1; j+= (ygv.end - ygv.start)*0.1) t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, _OutputMatX);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), _OutputMatY);
}

cv::Mat fspecialLoG(int WinSize, int sigma)
{

	// I wrote this only for square kernels as I have no need for kernels that aren't square
	int siz = (WinSize - 1) / 2;
	//	int std2 = std::pow(sigma, 2);
	int std2 = sigma * sigma;

	twodouble xx(siz * -1, siz);

	twodouble yy(siz * -1, siz);


	cv::Mat xmesh, ymesh;

	meshgrid(xx, yy, 1, xmesh, ymesh);

	float temp = 0.0;
	for (int i = 0; i < xmesh.cols; i++) {
		for (int j = 0; j < xmesh.rows; j++) {
			temp = xmesh.at<float>(i, j);
			xmesh.at<float>(i, j) = temp * temp;
		}
	}
	for (int i = 0; i < ymesh.cols; i++) {
		for (int j = 0; j < ymesh.rows; j++) {
			temp = ymesh.at<float>(i, j);
			ymesh.at<float>(i, j) = temp * temp;
		}
	}

	cv::Mat arg = -(xmesh + ymesh) / (2 * sigma);

	cv::Mat h(arg.size(), arg.type());

	for (int i = 0; i < arg.cols; i++) {
		for (int j = 0; j < arg.rows; j++) {
			h.at<float>(i, j) = std::exp(arg.at<float>(i, j));
		}
	}

	if (cv::sum(h)[0] != 0) { h = h / cv::sum(h)[0]; }

	cv::Mat temp_mat;
	temp_mat = xmesh + ymesh - 2 * std2;

	cv::Mat h1(h.size(), h.type());

	for (int i = 0; i < h.cols; i++) {
		for (int j = 0; j < h.rows; j++) {
			h1.at<float>(i, j) = h.at<float>(i, j) * temp_mat.at<float>(i, j);
		}
	}
	h1 = h1 / (pow(std2, 2));

	h = h1 - cv::sum(h1)[0] / (WinSize*WinSize);

	return h;
}

void downsample2(cv::Mat src, int factor, cv::Mat& dst)
{
	if (src.type() != CV_32FC1)
	{
		src.convertTo(src, CV_32FC1);
	}

	int width = src.cols;
	int height = src.rows;

	std::vector<std::vector<float> > data_;
	for (int j = 0; j < height;)
	{
		std::vector<float> oneRow;
		for (int i = 0; i < width; )
		{
			oneRow.push_back(src.at<float>(j, i));
			i += factor;
		}
		data_.push_back(oneRow);
		j += factor;
	}

	int newH = data_.size();
	int newW = data_[0].size();
	std::vector<float> matData;
	for (auto it = data_.begin(); it != data_.end(); it++)
	{
		matData.insert(matData.end(), it->begin(), it->end());
	}
	cv::Mat img = cv::Mat(matData);
	dst = img.reshape(1, newH).clone();
}


/**
 * \brief
 * //output parameters
 * \param z: the estimated image
 * \param zx1: the estimated gradient image along the x1 direction (vertical direction)
 * \param zx2: the estimated gradient image along the x2 direction (horizontal direction)
 * //input parameters
 * \param srcImg: the input image
 * \param h: the global smoothing parameter
 * \param upS: the upscaling factor ("r" must be an integer number)
 * \param winSize: the size of the kernel (ksize x ksize, and "ksize" must be an odd number)
 */
void ckr2_regular(cv::Mat& z, cv::Mat& zx1, cv::Mat& zx2, cv::Mat srcImg, double h, int upS, int winSize)
{
	cv::Size imgSize = srcImg.size();

	//Initialize the return parameters
	z.create(imgSize* upS, CV_32FC1);
	zx1.create(imgSize* upS, CV_32FC1);
	zx2.create(imgSize* upS, CV_32FC1);

	//Create the equivalent kernels
	int radius = (winSize - 1) / 2;
	cv::Mat x1, x2;
	meshgrid(twodouble(-radius - (upS - 1), radius), twodouble(-radius - (upS - 1), radius), 1.0 / upS, x1, x2);
	std::vector<std::vector<cv::Mat> > A;
	for (int i = 0; i < upS; i++)
	{
		std::vector<cv::Mat> A_row;
		for (int j = 0; j < upS; j++)
		{
			cv::Mat xx1, xx2;
			downsample2(x1(cv::Rect(cv::Point(upS - j + 1, upS - i + 1), cv::Point(x1.cols - 1, x1.rows - 1))), upS, xx1);
			downsample2(x2(cv::Rect(cv::Point(upS - j + 1, upS - i + 1), cv::Point(x2.cols - 1, x2.rows - 1))), upS, xx2);

			//The feture matrix
			cv::Mat Xx(winSize*winSize, 6, CV_32F);
			Xx.col(0) = cv::Mat::ones(winSize*winSize, 1, CV_32F);
			Xx.col(1) = xx1.reshape(xx1.channels(), winSize*winSize);
			Xx.col(2) = xx2.reshape(xx1.channels(), winSize*winSize);
			Xx.col(3) = Xx.col(1).mul(Xx.col(1));
			Xx.col(4) = Xx.col(1).mul(Xx.col(2));
			Xx.col(5) = Xx.col(2).mul(Xx.col(2));

			//The weight matrix (Gaussian kernel function)
			cv::Mat tt = xx1.mul(xx1) + xx2.mul(xx2);
			cv::Mat W;
			double temp = 0.5 / (h * h);
			exp(-temp * tt, W);
			cv::Mat W_ = W.reshape(1, winSize * winSize);

			//Equivalent kernel
			cv::Mat Xw(Xx.size(), CV_32F);
			Xw.col(0) = Xx.col(0).mul(W_);
			Xw.col(1) = Xx.col(1).mul(W_);
			Xw.col(2) = Xx.col(2).mul(W_);
			Xw.col(3) = Xx.col(3).mul(W_);
			Xw.col(4) = Xx.col(4).mul(W_);
			Xw.col(5) = Xx.col(5).mul(W_);

			cv::Mat A__ = (Xx.t() * Xw).inv(cv::DECOMP_SVD) * Xw.t();
			A_row.push_back(A__);
		}
		A.push_back(A_row);
	}

	//Mirroring the input image
	cv::Mat srcImg_border;
	cv::copyMakeBorder(srcImg, srcImg_border, winSize, winSize, winSize, winSize, cv::BORDER_REFLECT);

	//Estimate an image and its first gradients with pixel-by-pixel
	for (int h_ = 0; h_ < imgSize.height; h_++)
	{
		for (int w_ = 0; w_ < imgSize.width; w_++)
		{
			//Neighboring samples to be taken account into the estimation
			cv::Mat yp = srcImg_border(cv::Range(h_, h_ + winSize), cv::Range(w_, w_ + winSize));

			for (int i = 0; i < winSize; i++)
			{
				int nn = h_ * winSize + i;
				for (int j = 0; j < winSize; j++)
				{
					int mm = w_ * winSize + j;
					z.at<float>(nn, mm) = (A[i][j].row(0).dot(yp.reshape(1, A[j][i].cols)));
					zx1.at<float>(nn, mm) = A[i][j].row(1).dot(yp.reshape(1, A[j][i].cols));
					zx2.at<float>(nn, mm) = A[i][j].row(2).dot(yp.reshape(1, A[j][i].cols));
				}
			}
		}
	}

}

/**
 * \brief Compute steering matrices
 * \param C : steering matrices
 * \param zx : image gradients along x and y directions
 * \param zy
 * \param I : sampling positions
 * \param winSize : size of an analysis window
 * \param lambda : regularization parameter
 * \param alpha : structure sensitive parameter
 */
void steering(std::vector<std::vector<cv::Mat> >& C, cv::Mat zx, cv::Mat zy, cv::Mat I, int winSize, double lambda, double alpha)
{
	cv::Size imgSize = zx.size();

	if (!C.empty())
	{
		C.clear();
	}
	for (int i = 0; i < 2; i++)
	{
		std::vector<cv::Mat> c_;
		for (int j = 0; j < 2; j++)
		{
			cv::Mat tmp = cv::Mat::zeros(imgSize, CV_32FC1);
			c_.push_back(tmp);
		}
		C.push_back(c_);
	}

	if (winSize % 2 == 0)
	{
		winSize += 1;
	}
	int win = winSize / 2;


}
