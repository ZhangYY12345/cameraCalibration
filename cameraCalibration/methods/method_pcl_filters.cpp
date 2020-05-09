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
	pass.setFilterFieldName("z");							//设置过滤时所需要点云类型的z字段
	pass.setFilterLimits(0.0, 1.0);		//设置在过滤字段上的范围
	//pass.setFilterLimitsNegative(true);						//设置保留范围内的还是过滤掉范围内的
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
	sor.setMeanK(50);					//设置在进行统计时考虑查询点邻近点数
	sor.setStddevMulThresh(1.0);		//设置判断是否为离群点的阈值
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
	//定义模型系数对象，并填充对应的数据
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	coefficients->values.resize(4);					//使用ax + by + cz + d = 0的平面模型
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	pcl::ProjectInliers<pcl::PointXYZRGB> proj;				//创建投影滤波对象
	proj.setModelType(pcl::SACMODEL_PLANE);					//设置对象对应的投影模型，此处为平面
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
	pointcloudFilter(cloud_origin, cloud_blob, VOXEL_GRID);				//对数据进行下采样，加速处理过程

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;									//创建分割对象
	seg.setOptimizeCoefficients(true);											//设置对估计的模型参数进行优化处理
	seg.setModelType(pcl::SACMODEL_PLANE);										//设置分割模型类别
	seg.setModelType(pcl::SAC_RANSAC);											//设置用哪个随机参数估计方法
	seg.setMaxIterations(100);													//设置最大迭代次数
	seg.setDistanceThreshold(0.01);												//设置判断是否为模型内点的距离阈值

	seg.setInputCloud(cloud_blob);								
	seg.segment(*inliers, *coefficients);						

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;				//创建点云提取对象
	extract.setInputCloud(cloud_blob);							
	extract.setIndices(inliers);								//设置分割后的内点为需要提取的点集
	extract.setNegative(false);									//设置提取内点而非外点
	extract.filter(*cloud_filtered);							//提取输出存储到cloud_filtered
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
	pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr range_cone(new pcl::ConditionAnd<pcl::PointXYZRGB>);		//创建条件定义对象
	//添加在z字段上大于0.0的比较算子
	range_cone->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
		new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GT, 0.0)));
	//添加在z字段上小于0.8的比较算子
	range_cone->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
		new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LT, 0.8)));
	//创建滤波器并用条件定义对象初始化
	pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem(range_cone);
	condrem.setInputCloud(cloud_origin);
	condrem.setKeepOrganized(true);				//设置保持点云的结构
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
	outrem.setRadiusSearch(0.8);							//设置在0.8半径的范围内找邻近点
	outrem.setMinNeighborsInRadius(5);						//设置查询点的邻近点集点数小于5的删除
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
	//输入2D平面点云//多边形
	boundingbox_ptr->push_back(pcl::PointXYZ(0.1, 0.1, 0.0));
	boundingbox_ptr->push_back(pcl::PointXYZ(0.1, -0.1, 0));
	boundingbox_ptr->push_back(pcl::PointXYZ(-0.1, 0.1, 0));
	boundingbox_ptr->push_back(pcl::PointXYZ(-0.1, -0.1, 0));

	pcl::ConvexHull<pcl::PointXYZ> hull;													//创建凸包对象
	hull.setInputCloud(boundingbox_ptr);
	hull.setDimension(2);
	std::vector<pcl::Vertices> polygons;													//设置pcl::Vertices类型的向量，用于保存凸包顶点
	pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull(new pcl::PointCloud<pcl::PointXYZ>);	//该点云用于描述凸包形状
	hull.reconstruct(*surface_hull, polygons);												//计算2D凸包结果

	pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::CropHull<pcl::PointXYZ> bb_filter;
	bb_filter.setDim(2);
	bb_filter.setInputCloud(cloud_origin);
	bb_filter.setHullIndices(polygons);
	bb_filter.setHullCloud(surface_hull);
}

