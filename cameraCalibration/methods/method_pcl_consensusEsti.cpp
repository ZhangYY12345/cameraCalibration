#include "method_pcl_consensusEsti.h"
#include "parametersStereo.h"
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/io/io.h>


void pclSampleConsens_RANSAC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyPts, CONSENSUS_MODEL_TYPE_ type)
{
	std::vector<int> inliers;																//存储局内点集合的点的索引的向量

	// created RandomSampleConsensus object and compute the appropriated model
	pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr									//针对平面模型的对象
		model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>(cloud_origin));
	pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr									//针对球模型的对象
		model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>(cloud_origin));
	if (type == CONSENSUS_MODEL_PLANE_)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_p);
		ransac.setDistanceThreshold(.01);								//与平面距离小于0.01的点作为局内点考虑
		ransac.computeModel();											//执行随机参数估计
		ransac.getInliers(inliers);										//存储估计所得的局内点
	}
	else if (type == CONSENSUS_MODEL_SPHERE_)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_s);
		ransac.setDistanceThreshold(.01);								//与球面距离小于0.01的点作为局内点考虑
		ransac.computeModel();											//执行随机参数估计
		ransac.getInliers(inliers);										//存储估计所得的局内点
	}

	// copies all inliers of the model computed to another PointCloud
	pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud_origin, inliers, *keyPts);
}
