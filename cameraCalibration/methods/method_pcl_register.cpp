#include "method_pcl_register.h"
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/pcl_visualizer.h>

/**
 * \brief ʹ��ICP�㷨��iterative closest point,��������㷨)���е�����׼������׼
 * \param cloud_obj :Ŀ�����ĵ������ݣ�ģ�壬ƥ��Ŀ�꣩ԭ���ƣ�Ŀ�����
 * \param cloud_scene :�����������ݣ��ڳ�����Ѱ��Ŀ�����Դ����
 */
void pclRegister_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obj, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene)
{
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputCloud(cloud_scene);
	icp.setInputTarget(cloud_obj);
	pcl::PointCloud<pcl::PointXYZ> Final;						//�洢������׼�任Դ���ƺ�ĵ���
	icp.align(Final);											//ִ����׼���洢�任���Դ���Ƶ�Final
	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << std::endl;						//��ӡ��׼���������Ϣ
	std::cout << icp.getFinalTransformation() << std::endl;		//��ӡ������չ��Ƶı任����
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
	//�������ӻ�����
	pcl::visualization::PCLVisualizer *p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");;
	//���������ӵ�
	int vp_1, vp_2;
	p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
	p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);

	//Ϊ��һ���Ժ͸��ٵ��²���
	//ע�⣺Ϊ�˴����ݼ���Ҫ��������
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
	//�������淨�ߺ�����
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
	
	//����˵�������Զ����ı�ʾ�����϶��壩
	MyPointRepresentation point_representation;
	//����'curvature'�ߴ�Ȩ���Ա�ʹ����x, y, zƽ��
	float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
	point_representation.setRescaleValues(alpha);

	// ��׼
	pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
	reg.setTransformationEpsilon(1e-6);
	//��������Ӧ��ϵ֮���(src<->tgt)����������Ϊ10����
	//ע�⣺����������ݼ���С������
	reg.setMaxCorrespondenceDistance(0.1);
	//���õ��ʾ
	reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));
	reg.setInputCloud(points_with_normals_src);
	reg.setInputTarget(points_with_normals_tgt);
	//
	//��һ��ѭ����������ͬ�����Ż�����ʹ������ӻ�
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
	pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = points_with_normals_src;
	reg.setMaximumIterations(2);
	for (int i = 0; i < 30; ++i)
	{
		PCL_INFO("Iteration Nr. %d.\n", i);
		//Ϊ�˿��ӻ���Ŀ�ı������
		points_with_normals_src = reg_result;
		//����
		reg.setInputCloud(points_with_normals_src);
		reg.align(*reg_result);
		//��ÿһ������֮���ۻ�ת��
		Ti = reg.getFinalTransformation() * Ti;
		//������ת����֮ǰת��֮��Ĳ���С����ֵ
		//��ͨ����С����Ӧ���������Ƴ���
		if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
			reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);
		prev = reg.getLastIncrementalTransformation();

		//���ӻ���ǰ״̬
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
  // �õ�Ŀ����Ƶ�Դ���Ƶı任
	targetToSource = Ti.inverse();
	//
	//��Ŀ�����ת����Դ���
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
	//���Դ���Ƶ�ת��Ŀ��
	*output += *cloud_src;
	final_transform = targetToSource;

}
