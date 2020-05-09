#pragma once
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>

void pclFeaturePt_NARF(pcl::RangeImage& rangeImg, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPts, pcl::PointCloud<int>::Ptr keypoint_indices);
void pclFeaturePt_SIFT(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPts);
void pclFeaturePt_HARRIS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZI >::Ptr keyPts);

void pclFeatureDesp_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal);
void pclFeatureDesp_normal_inte(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal);
void pclFeatureDesp_PFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PFHSignature125>::Ptr cloud_feature);
void pclFeatureDesp_FPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_feature);
void pclFeatureDesp_FPFH_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_feature);
void pclFeaureDesp_VFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::VFHSignature308>::Ptr cloud_feature);
void pclFeatureDesp_NARF(pcl::RangeImage& rangeImg,
	pcl::PointCloud<pcl::Narf36>::Ptr cloud_feature);
void pclFeatureDesp_RoPS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::Histogram <135>>::Ptr cloud_feature);
void pclFeatureDesp_MomentOfInertial(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin);
void pclFeatureDesp_BounderEst(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary);
