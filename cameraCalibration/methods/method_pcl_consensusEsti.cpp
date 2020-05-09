#include "method_pcl_consensusEsti.h"
#include "parametersStereo.h"
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/io/io.h>


void pclSampleConsens_RANSAC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyPts, CONSENSUS_MODEL_TYPE_ type)
{
	std::vector<int> inliers;																//�洢���ڵ㼯�ϵĵ������������

	// created RandomSampleConsensus object and compute the appropriated model
	pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr									//���ƽ��ģ�͵Ķ���
		model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>(cloud_origin));
	pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr									//�����ģ�͵Ķ���
		model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>(cloud_origin));
	if (type == CONSENSUS_MODEL_PLANE_)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_p);
		ransac.setDistanceThreshold(.01);								//��ƽ�����С��0.01�ĵ���Ϊ���ڵ㿼��
		ransac.computeModel();											//ִ�������������
		ransac.getInliers(inliers);										//�洢�������õľ��ڵ�
	}
	else if (type == CONSENSUS_MODEL_SPHERE_)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_s);
		ransac.setDistanceThreshold(.01);								//���������С��0.01�ĵ���Ϊ���ڵ㿼��
		ransac.computeModel();											//ִ�������������
		ransac.getInliers(inliers);										//�洢�������õľ��ڵ�
	}

	// copies all inliers of the model computed to another PointCloud
	pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud_origin, inliers, *keyPts);
}
