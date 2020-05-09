#include "method_pcl_register.h"
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/pcl_visualizer.h>

/**
 * \brief 使用ICP算法（iterative closest point,迭代最近点法)进行点云配准；精配准
 * \param cloud_obj :目标对象的点云数据（模板，匹配目标）原点云；目标点云
 * \param cloud_scene :场景点云数据；在场景中寻找目标对象；源点云
 */
void pclRegister_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obj, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene)
{
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputCloud(cloud_scene);
	icp.setInputTarget(cloud_obj);
	pcl::PointCloud<pcl::PointXYZ> Final;						//存储经过配准变换源点云后的点云
	icp.align(Final);											//执行配准，存储变换后的源点云到Final
	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << std::endl;						//打印配准相关输入信息
	std::cout << icp.getFinalTransformation() << std::endl;		//打印输出最终估计的变换矩阵
}

/**
 * \brief 
 * \param cloud_src 
 * \param cloud_tgt 
 * \param output 
 * \param final_transform 
 * \param downsample 
 */
void pclRegister_ICP_pairAlign(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt, pcl::PointCloud<pcl::PointXYZ>::Ptr output,
	Eigen::Matrix4f& final_transform, bool downsample)
{
	//创建可视化工具
	pcl::visualization::PCLVisualizer *p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");;
	//定义左右视点
	int vp_1, vp_2;
	p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
	p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);

	//为了一致性和高速的下采样
	//注意：为了大数据集需要允许这项
	pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> grid;
	if (downsample)
	{
		grid.setLeafSize(0.05, 0.05, 0.05);
		grid.setInputCloud(cloud_src);
		grid.filter(*src);
		grid.setInputCloud(cloud_tgt);
		grid.filter(*tgt);
	}
	else
	{
		src = cloud_src;
		tgt = cloud_tgt;
	}
	//计算曲面法线和曲率
	pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt(new pcl::PointCloud<pcl::PointNormal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> norm_est;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	norm_est.setSearchMethod(tree);
	norm_est.setKSearch(30);
	norm_est.setInputCloud(src);
	norm_est.compute(*points_with_normals_src);
	pcl::copyPointCloud(*src, *points_with_normals_src);
	norm_est.setInputCloud(tgt);
	norm_est.compute(*points_with_normals_tgt);
	pcl::copyPointCloud(*tgt, *points_with_normals_tgt);
	
	//举例说明我们自定义点的表示（以上定义）
	MyPointRepresentation point_representation;
	//调整'curvature'尺寸权重以便使它和x, y, z平衡
	float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
	point_representation.setRescaleValues(alpha);

	// 配准
	pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
	reg.setTransformationEpsilon(1e-6);
	//将两个对应关系之间的(src<->tgt)最大距离设置为10厘米
	//注意：根据你的数据集大小来调整
	reg.setMaxCorrespondenceDistance(0.1);
	//设置点表示
	reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));
	reg.setInputCloud(points_with_normals_src);
	reg.setInputTarget(points_with_normals_tgt);
	//
	//在一个循环中运行相同的最优化并且使结果可视化
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
	pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = points_with_normals_src;
	reg.setMaximumIterations(2);
	for (int i = 0; i < 30; ++i)
	{
		PCL_INFO("Iteration Nr. %d.\n", i);
		//为了可视化的目的保存点云
		points_with_normals_src = reg_result;
		//估计
		reg.setInputCloud(points_with_normals_src);
		reg.align(*reg_result);
		//在每一个迭代之间累积转换
		Ti = reg.getFinalTransformation() * Ti;
		//如果这次转换和之前转换之间的差异小于阈值
		//则通过减小最大对应距离来改善程序
		if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
			reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);
		prev = reg.getLastIncrementalTransformation();

		//可视化当前状态
		p->removePointCloud("source");
		p->removePointCloud("target");
		pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> tgt_color_handler(points_with_normals_tgt, "curvature");
		if (!tgt_color_handler.isCapable())
			PCL_WARN("Cannot create curvature color handler!");
		pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> src_color_handler(points_with_normals_src, "curvature");
		if (!src_color_handler.isCapable())
			PCL_WARN("Cannot create curvature color handler!");
		p->addPointCloud(points_with_normals_tgt, tgt_color_handler, "target", vp_1);
		p->addPointCloud(points_with_normals_src, src_color_handler, "source", vp_1);
		p->spinOnce();
	}
	//
  // 得到目标点云到源点云的变换
	targetToSource = Ti.inverse();
	//
	//把目标点云转换回源框架
	pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);

	p->removePointCloud("source");
	p->removePointCloud("target");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_tgt_h(output, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_h(cloud_src, 255, 0, 0);
	p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
	p->addPointCloud(cloud_src, cloud_src_h, "source", vp_2);
	PCL_INFO("Press q to continue the registration.\n");
	p->spin();
	p->removePointCloud("source");
	p->removePointCloud("target");
	//添加源点云到转换目标
	*output += *cloud_src;
	final_transform = targetToSource;

}
