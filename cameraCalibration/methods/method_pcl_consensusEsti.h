#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

void pclSampleConsens_RANSAC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyPts);
