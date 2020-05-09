#pragma once
#include <opencv2/opencv.hpp>
#include "parametersStereo.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>

//single camera calibreation
void myCameraCalibration(std::string cameraParaPath);
void myCameraCalibration(std::string imgFilePath, std::string cameraParaPath);
void myCameraUndistort(std::string cameraParaPath);
void myCameraUndistort(std::string imgFilePath, std::string cameraParaPath);

//two camera calibration and stereo vision
void twoCamerasCalibration(std::string cameraParaPath);
void twoCamerasCalibration(std::string imgFilePath, std::string cameraParaPath);
void twoCamerasCalibration(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath);

cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r);
void stereoFisheyeCamCalib(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath);
void stereoFisheyCamCalibRecti(std::string imgFilePathL, std::string cameraParaPath);


void stereoCameraUndistort(std::string cameraParaPath);
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(cv::Mat imgLeft, cv::Mat imgRight, std::string cameraParaPath, 
	cv::Mat& rectifiedLeft, cv::Mat& rectifiedRight);

void stereoMatching(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap, 
	StereoMatchingAlgorithms algorithmType);
void getDisparity_BM(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap);
void getDisparity_SGBM(cv::Mat srcLeft, cv::Mat srcRight, cv::Mat& disparityMap);

//          
void cvtGreyToBGR(cv::Mat greySrc, cv::Mat& dstBGR);
  
//get 3-d point cloud with the disparity map
void viewerOneOff(pcl::visualization::PCLVisualizer& viewer);
void viewerPsycho(pcl::visualization::PCLVisualizer& viewer);
void showPointCloud(cv::Mat originBGRMap, cv::Mat disparityMap, 
	std::string stereoParamPath = "");
void showPointCloudVisual(cv::Mat originBGRMap, cv::Mat disparityMap, 
	std::string cameraPairParaPath = "");
void showPointCloudVisual_my2(cv::Mat originBGRMap, cv::Mat disparityMap,
	std::string cameraPairParaPath = "", std::string resPCLPath = "", bool isLeft = true);
void showPointCloudVisual_my22(cv::Mat originBGRMap, cv::Mat disparityMap,
	std::string cameraPairParaPath = "", std::string resPCLPath = "", bool isLeft = true);

void showPointCloudVisual_my3(cv::Mat originBGRMap, cv::Mat disparityMap,
	std::string cameraPairParaPath = "", std::string resPCLPath = "", bool isLeft = true);
void showPointCloudVisual_my33(cv::Mat originBGRMap, cv::Mat disparityMap,
	std::string cameraPairParaPath = "", std::string resPCLPath = "", bool isLeft = true);

void pclFilter_my2(std::string pclPath, std::string dstPCL_path);

void pclPoint_xyzrgb2xyz(
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_origin,
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_dst);

void pclMesh_OrganizedFastMesh(
	boost::shared_ptr<pcl::RangeImagePlanar> range_image_origin, 
	pcl::PolygonMesh& triangle_mesh_dst);

double computeCloudResolution(
	const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud);
void pcl_CorrespGroup(
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> > cloud_model,
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> > cloud_scene,
	std::vector<Eigen::Matrix4f, 
				Eigen::aligned_allocator<Eigen::Matrix4f> >& rototranslations);