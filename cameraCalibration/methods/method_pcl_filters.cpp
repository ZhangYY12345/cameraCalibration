#include "method_pcl_filters.h"
#include "methods.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>

void pointcloudFilter(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered, PCLFILTERS_ filterType)
{
	switch (filterType)
	{
	case PASS_THROUGH:
		pclFilter_PassThrough(cloud2_origin, cloud2_filtered);
		break;
	case VOXEL_GRID:
		pclFilter_VoxelGrid(cloud2_origin, cloud2_filtered);
		break;
	case STATISTIC_OUTLIERS_REMOVE:
		pclFilter_OutlierRemoval(cloud2_origin, cloud2_filtered);
		break;
	case MODEL_COEFFICIENTS:
		pclFilter_ModelCoefficients(cloud2_origin, cloud2_filtered);
		break;
	case EXTRACT_INDICES:
		pclFilter_ExtractIndices(cloud2_origin, cloud2_filtered);
		break;

	}
}

void pointcloudFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered, PCLFILTERS_ filterType)
{
	switch (filterType)
	{
	case PASS_THROUGH:
		pclFilter_PassThrough(cloud_origin, cloud_filtered);
		break;
	case VOXEL_GRID:
		pclFilter_VoxelGrid(cloud_origin, cloud_filtered);
		break;
	case STATISTIC_OUTLIERS_REMOVE:
		pclFilter_OutlierRemoval(cloud_origin, cloud_filtered);
		break;
	case MODEL_COEFFICIENTS:
		pclFilter_ModelCoefficients(cloud_origin, cloud_filtered);
		break;
	case EXTRACT_INDICES:
		pclFilter_ExtractIndices(cloud_origin, cloud_filtered);
		break;
	}
}

void pclFilter_PassThrough(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_PassThrough(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);
}

void pclFilter_PassThrough(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud(cloud_origin);
	pass.setFilterFieldName("z");							//���ù���ʱ����Ҫ�������͵�z�ֶ�
	pass.setFilterLimits(0.0, 1.0);		//�����ڹ����ֶ��ϵķ�Χ
	//pass.setFilterLimitsNegative(true);						//���ñ�����Χ�ڵĻ��ǹ��˵���Χ�ڵ�
	pass.filter(*cloud_filtered);
}

void pclFilter_VoxelGrid(pcl::PCLPointCloud2::Ptr cloud2_origin, 
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud2_origin);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	sor.filter(*cloud2_filtered);
}

void pclFilter_VoxelGrid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	pcl::PCLPointCloud2::Ptr cloud2_origin(new pcl::PCLPointCloud2);
	pcl::PCLPointCloud2::Ptr cloud2_filtered(new pcl::PCLPointCloud2);
	pcl::toPCLPointCloud2(*cloud_origin, *cloud2_origin);
	pclFilter_VoxelGrid(cloud2_origin, cloud2_filtered);
	pcl::fromPCLPointCloud2(*cloud2_filtered, *cloud_filtered);
}

void pclFilter_OutlierRemoval(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_OutlierRemoval(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);
}

void pclFilter_OutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud_origin);
	sor.setMeanK(50);					//�����ڽ���ͳ��ʱ���ǲ�ѯ���ڽ�����
	sor.setStddevMulThresh(1.0);		//�����ж��Ƿ�Ϊ��Ⱥ�����ֵ
	sor.filter(*cloud_filtered);
}

void pclFilter_ModelCoefficients(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_ModelCoefficients(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);
}

void pclFilter_ModelCoefficients(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	//����ģ��ϵ�����󣬲�����Ӧ������
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	coefficients->values.resize(4);					//ʹ��ax + by + cz + d = 0��ƽ��ģ��
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	pcl::ProjectInliers<pcl::PointXYZRGB> proj;				//����ͶӰ�˲�����
	proj.setModelType(pcl::SACMODEL_PLANE);					//���ö����Ӧ��ͶӰģ�ͣ��˴�Ϊƽ��
	proj.setInputCloud(cloud_origin);
	proj.setModelCoefficients(coefficients);
	proj.filter(*cloud_filtered);
}

void pclFilter_ExtractIndices(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_ExtractIndices(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);
}

void pclFilter_ExtractIndices(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_blob(new pcl::PointCloud<pcl::PointXYZRGB>);
	pointcloudFilter(cloud_origin, cloud_blob, VOXEL_GRID);				//�����ݽ����²��������ٴ������

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;									//�����ָ����
	seg.setOptimizeCoefficients(true);											//���öԹ��Ƶ�ģ�Ͳ��������Ż�����
	seg.setModelType(pcl::SACMODEL_PLANE);										//���÷ָ�ģ�����
	seg.setModelType(pcl::SAC_RANSAC);											//�������ĸ�����������Ʒ���
	seg.setMaxIterations(100);													//��������������
	seg.setDistanceThreshold(0.01);												//�����ж��Ƿ�Ϊģ���ڵ�ľ�����ֵ

	seg.setInputCloud(cloud_blob);								
	seg.segment(*inliers, *coefficients);						

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;				//����������ȡ����
	extract.setInputCloud(cloud_blob);							
	extract.setIndices(inliers);								//���÷ָ����ڵ�Ϊ��Ҫ��ȡ�ĵ㼯
	extract.setNegative(false);									//������ȡ�ڵ�������
	extract.filter(*cloud_filtered);							//��ȡ����洢��cloud_filtered
}

void pclFilter_CondidtionalRemoval(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_CondidtionalRemoval(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);
}

void pclFilter_CondidtionalRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr range_cone(new pcl::ConditionAnd<pcl::PointXYZRGB>);		//���������������
	//�����z�ֶ��ϴ���0.0�ıȽ�����
	range_cone->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
		new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GT, 0.0)));
	//�����z�ֶ���С��0.8�ıȽ�����
	range_cone->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
		new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LT, 0.8)));
	//�����˲�������������������ʼ��
	pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem(range_cone);
	condrem.setInputCloud(cloud_origin);
	condrem.setKeepOrganized(true);				//���ñ��ֵ��ƵĽṹ
	condrem.filter(*cloud_filtered);
}

void pclFilter_RadiusOutlierRemoval(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_RadiusOutlierRemoval(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);
}

void pclFilter_RadiusOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered)
{
	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
	outrem.setInputCloud(cloud_origin);
	outrem.setRadiusSearch(0.8);							//������0.8�뾶�ķ�Χ�����ڽ���
	outrem.setMinNeighborsInRadius(5);						//���ò�ѯ����ڽ��㼯����С��5��ɾ��
	outrem.filter(*cloud_filtered);
}

void pclFilter_CropHull(pcl::PCLPointCloud2::Ptr cloud2_origin,
	pcl::PCLPointCloud2::Ptr cloud2_filtered)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(*cloud2_origin, *cloud_origin);
	pclFilter_CropHull(cloud_origin, cloud_filtered);
	pcl::toPCLPointCloud2(*cloud_filtered, *cloud2_filtered);

}

void pclFilter_CropHull(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr boundingbox_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	//����2Dƽ�����//�����
	boundingbox_ptr->push_back(pcl::PointXYZ(0.1, 0.1, 0.0));
	boundingbox_ptr->push_back(pcl::PointXYZ(0.1, -0.1, 0));
	boundingbox_ptr->push_back(pcl::PointXYZ(-0.1, 0.1, 0));
	boundingbox_ptr->push_back(pcl::PointXYZ(-0.1, -0.1, 0));

	pcl::ConvexHull<pcl::PointXYZ> hull;													//����͹������
	hull.setInputCloud(boundingbox_ptr);
	hull.setDimension(2);
	std::vector<pcl::Vertices> polygons;													//����pcl::Vertices���͵����������ڱ���͹������
	pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull(new pcl::PointCloud<pcl::PointXYZ>);	//�õ�����������͹����״
	hull.reconstruct(*surface_hull, polygons);												//����2D͹�����

	pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::CropHull<pcl::PointXYZ> bb_filter;
	bb_filter.setDim(2);
	bb_filter.setInputCloud(cloud_origin);
	bb_filter.setHullIndices(polygons);
	bb_filter.setHullCloud(surface_hull);
}

