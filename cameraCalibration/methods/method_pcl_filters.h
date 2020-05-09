#pragma once
#include "parametersStereo.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//pcl filters
void pointcloudFilter(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered, PCLFILTERS_ filterType);
void pointcloudFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered, PCLFILTERS_ filterType);

void pclFilter_PassThrough(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_PassThrough(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_VoxelGrid(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_VoxelGrid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_OutlierRemoval(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_OutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_ModelCoefficients(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_ModelCoefficients(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_ExtractIndices(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_ExtractIndices(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_CondidtionalRemoval(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_CondidtionalRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_RadiusOutlierRemoval(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_RadiusOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered);

void pclFilter_CropHull(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered);
void pclFilter_CropHull(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

