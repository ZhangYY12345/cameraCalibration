#pragma once
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/point_representation.h>

//��< x, y, z, curvature >��ʽ����һ���µĵ�
class MyPointRepresentation : public pcl::PointRepresentation <pcl::PointNormal>
{
	using PointRepresentation<pcl::PointNormal>::nr_dimensions_;
public:
	MyPointRepresentation()
	{
		//����ߴ�ֵ
		nr_dimensions_ = 4;
	}
	//����copyToFloatArray�������������ǵ�����ʸ��
	virtual void copyToFloatArray(const pcl::PointNormal &p, float * out) const
	{
		// < x, y, z, curvature >
		out[0] = p.x;
		out[1] = p.y;
		out[2] = p.z;
		out[3] = p.curvature;
	}
};

void pclRegister_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obj, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene);
void pclRegister_ICP_pairAlign(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr output, 
	Eigen::Matrix4f &final_transform, bool downsample = false);
