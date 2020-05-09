#include "method_pcl_keypoint.h"
#include <pcl/range_image/range_image.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/features/narf.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/boundary.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3D.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

/**
 * \brief NARF: Normal Aligned Radial Feature
 * \param rangeImg 
 * \param keyPts 
 */
void pclFeaturePt_NARF(pcl::RangeImage& rangeImg, pcl::PointCloud<pcl::PointXYZ>::Ptr keyPts, pcl::PointCloud<int>::Ptr keypoint_indices)
{
	pcl::RangeImageBorderExtractor rangeImgBorderExtra;
	pcl::NarfKeypoint narfKeyptDete(&rangeImgBorderExtra);				//NARF首先需要探测深度图像的边缘
	narfKeyptDete.setRangeImage(&rangeImg);
	narfKeyptDete.getParameters().support_size = 5;
	//narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true;
	//narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;

	narfKeyptDete.compute(*keypoint_indices);
	std::cout << "Found " << keypoint_indices->points.size() << " key points.\n";

	//在距离图像显示组件内显示关键点
	//for (size_ti=0; i<keypoint_indices.points.size (); ++i)
	//range_image_widget.markPoint (keypoint_indices.points[i]%range_image.width,
	//keypoint_indices.points[i]/range_image.width);

	//关键点点云
	keyPts->points.resize(keypoint_indices->points.size());
	for (size_t i = 0; i < keypoint_indices->points.size(); ++i)
	{
		keyPts->points[i].getVector3fMap() = rangeImg.points[keypoint_indices->points[i]].getVector3fMap();
	}
}

/**
 * \brief SIFT:Scale-Invariant Feature Transform
 * \param rangeImg 
 * \param keyPts 
 */
void pclFeaturePt_SIFT(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_origin, pcl::PointCloud<pcl::PointXYZI>::Ptr keyPts)
{
	const float min_scale = 0.01;
	const int n_octaves = 6;
	const int n_scales_per_octave = 4;
	const float min_contrast = 0.01;

	pcl::SIFTKeypoint<pcl::PointXYZI, pcl::PointWithScale> sift;								//创建sift关键点检测对象
	pcl::PointCloud<pcl::PointWithScale> result;
	sift.setInputCloud(cloud_origin);															//设置输入点云
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
	sift.setSearchMethod(tree);																	//创建一个空的kd树对象tree，并把它传递给sift检测对象
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);									//指定搜索关键点的尺度范围
	sift.setMinimumContrast(min_contrast);														//设置限制关键点检测的阈值
	sift.compute(result);																		//执行sift关键点检测，保存结果在result
	
	copyPointCloud(result, *keyPts);															//将点类型pcl::PointWithScale的数据转换为点类型pcl::PointXYZ的数据
}

/**
 * \brief Harris:Harris算子是常见的特征检测算子，即可提取角点也可提取边缘点；
 *					2D Harris角点利用的是梯度信息，3D Harris角点检测利用的是点云法向量的信息
 * \param cloud_origin 
 * \param keyPts ：包含强度信息
 */
void pclFeaturePt_HARRIS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin, pcl::PointCloud<pcl::PointXYZI>::Ptr keyPts)
{
	float r_normal;
	float r_keypoint;

	r_normal = 0.1f;
	r_keypoint = 0.1f;

	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI, pcl::Normal>* harris_detector = new pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI, pcl::Normal>;

	//harris_detector->setNonMaxSupression(true);
	harris_detector->setRadius(r_normal);															//设置法向量估计的半径
	harris_detector->setRadiusSearch(r_keypoint);													//设置关键点估计的近邻搜索半径
	harris_detector->setInputCloud(cloud_origin);
	//harris_detector->setNormals(normal_source);
	//harris_detector->setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZRGB,pcl::PointXYZI>::LOWE);
	harris_detector->compute(*keyPts);

	pcl::PCDWriter writer;
	std::cout << "Harris_keypoints的大小是" << keyPts->size() << std::endl;
	writer.write<pcl::PointXYZI>("Harris_keypoints.pcd", *keyPts, false);
}

void pclFeatureDesp_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal)
{
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_origin);
	//创建一个空的kdtree对象，并把它传递给法线估计对象
	//基于给出的输入数据集，kdtree将被建立
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.03);			//使用半径在查询点周围3厘米范围内的所有邻元素
	ne.compute(*cloud_normal);			//计算特征值，输出点集

	////法线可视化
	//pcl::visualization::PCLVisualizer viewer("PCL Normal Viewer");
	//viewer.setBackgroundColor(0.0, 0.0, 0.0);
	//viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_origin, cloud_normal);
	//while (!viewer.wasStopped())
	//{
	//	viewer.spinOnce();
	//}
}

/**
 * \brief 使用积分图进行法线估计
 * \param cloud_origin 
 * \param cloud_normal 
 */
void pclFeatureDesp_normal_inte(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal)
{
	//估计法线
	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);			//设置估计方法
	ne.setMaxDepthChangeFactor(0.02f);								//最大深度变化系数
	ne.setNormalSmoothingSize(10.0f);								//优化法线方向时考虑邻域大小
	ne.setInputCloud(cloud_origin);									//输入点云，必须为有序点云
	ne.compute(*cloud_normal);										//执行法线估计，存储结果到cloud_normal

	////法线可视化
	//pcl::visualization::PCLVisualizer viewer("PCL Normal Viewer");
	//viewer.setBackgroundColor(0.0, 0.0, 0.5);
	//viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_origin, cloud_normal);
	//while (!viewer.wasStopped())
	//{
	//	viewer.spinOnce();
	//}
}

void pclFeatureDesp_PFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::PFHSignature125>::Ptr cloud_feature)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
	pclFeatureDesp_normal_inte(cloud_origin, cloud_normals);

	// Create the PFH estimation class, and pass the input dataset+normals to it
	pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
	pfh.setInputCloud(cloud_origin);
	pfh.setInputNormals(cloud_normals);
	// alternatively, if cloud is of type PointNormal, do pfh.setInputNormals (cloud);

	// Create an empty kdtree representation, and pass it to the PFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	//pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
	pfh.setSearchMethod(tree);

	// Use all neighbors in a sphere of radius 5cm
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	pfh.setRadiusSearch(0.05);			//使用半径在5厘米范围内的所有邻元素；
										//注意：此处使用的半径必须要大于估计表面法线时使用的半径

	// Compute the features
	pfh.compute(*cloud_feature);
}

void pclFeatureDesp_FPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_feature)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
	pclFeatureDesp_normal_inte(cloud_origin, cloud_normals);

	// Create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(cloud_origin);
	fpfh.setInputNormals(cloud_normals);
	// alternatively, if cloud is of type PointNormal, do fpfh.setInputNormals (cloud);

	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, 
	//based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	fpfh.setSearchMethod(tree);
	fpfh.setRadiusSearch(0.05);					//使用半径在5厘米范围内的所有邻元素；
												//注意：此处使用的半径必须要大于估计表面法线时使用的半径

	fpfh.compute(*cloud_feature);				// Compute the features
}

void pclFeatureDesp_FPFH_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_feature)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
	pclFeatureDesp_normal_inte(cloud_origin, cloud_normals);

	// Create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(cloud_origin);
	fpfh.setInputNormals(cloud_normals);
	// alternatively, if cloud is of type PointNormal, do fpfh.setInputNormals (cloud);

	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, 
	//based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	fpfh.setSearchMethod(tree);
	fpfh.setRadiusSearch(0.05);					//使用半径在5厘米范围内的所有邻元素；
												//注意：此处使用的半径必须要大于估计表面法线时使用的半径

	fpfh.compute(*cloud_feature);				// Compute the features

}

void pclFeaureDesp_VFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::VFHSignature308>::Ptr cloud_feature)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
	pclFeatureDesp_normal_inte(cloud_origin, cloud_normals);

	// Create the VFH estimation class, and pass the input dataset+normals to it
	pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(cloud_origin);
	vfh.setInputNormals(cloud_normals);
	// alternatively, if cloud is of type PointNormal, do vfh.setInputNormals (cloud);

	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	vfh.setSearchMethod(tree);

	// Compute the features
	vfh.compute(*cloud_feature);		//cloud_feature->points.size()的大小应该是1，即vfh描述子是针对全局的特征描述？？？
}

void pclFeatureDesp_NARF(pcl::RangeImage& rangeImg, pcl::PointCloud<pcl::Narf36>::Ptr cloud_feature)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypts(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<int>::Ptr keypoint_indices(new pcl::PointCloud<int>);
	pclFeaturePt_NARF(rangeImg, cloud_keypts, keypoint_indices);

	std::vector<int> keypoint_indices2;
	keypoint_indices2.resize(keypoint_indices->points.size());
	for (unsigned int i = 0; i < keypoint_indices->size(); ++i)
		keypoint_indices2[i] = keypoint_indices->points[i];					//要得到正确的向量类型，这一步是必要的

	pcl::NarfDescriptor narf_descriptor(&rangeImg, &keypoint_indices2);
	narf_descriptor.getParameters().support_size = 5;						//support size for the interest points (diameter of the used sphere default 5);与NARF特征点检测时的窗口大小一样
	narf_descriptor.getParameters().rotation_invariant = 0;					//switch rotational invariant version of the feature on/off:0/1
	narf_descriptor.compute(*cloud_feature);
	cout << "Extracted " << cloud_feature->size() << " descriptors for "
		<< keypoint_indices->points.size() << " keypoints.\n";
}

void pclFeatureDesp_RoPS(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin,
	pcl::PointCloud<pcl::Histogram<135>>::Ptr cloud_feature)
{
	float support_radius = 0.0285f;
	unsigned int number_of_partition_bins = 5;
	unsigned int number_of_rotations = 3;

	pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method(new pcl::search::KdTree<pcl::PointXYZ>);
	search_method->setInputCloud(cloud_origin);

	pcl::PointIndicesPtr indices = boost::shared_ptr <pcl::PointIndices>(new pcl::PointIndices());
	std::ifstream indices_file;
	indices_file.open("D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/methods/test/indices.txt", std::ifstream::in);
	for (std::string line; std::getline(indices_file, line);)
	{
		std::istringstream in(line);
		unsigned int index = 0;
		in >> index;
		indices->indices.push_back(index - 1);
	}
	indices_file.close();

	std::vector <pcl::Vertices> triangles;
	std::ifstream triangles_file;
	triangles_file.open("D:/studying/stereo vision/research code/cameraCalibration/cameraCalibration/methods/test/triangles.txt", std::ifstream::in);
	for (std::string line; std::getline(triangles_file, line);)
	{
		pcl::Vertices triangle;
		std::istringstream in(line);
		unsigned int vertex = 0;
		in >> vertex;
		triangle.vertices.push_back(vertex - 1);
		in >> vertex;
		triangle.vertices.push_back(vertex - 1);
		in >> vertex;
		triangle.vertices.push_back(vertex - 1);
		triangles.push_back(triangle);
	}


	pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram <135> > feature_estimator;
	feature_estimator.setSearchMethod(search_method);
	feature_estimator.setSearchSurface(cloud_origin);
	feature_estimator.setInputCloud(cloud_origin);
	feature_estimator.setIndices(indices);										//需要计算RoPS特征的点的标号，即关键点的标号
	feature_estimator.setTriangles(triangles);									//三角面元；RoPS算法并非针对点云数据，而需 求点之间拓扑信息
	feature_estimator.setRadiusSearch(support_radius);
	feature_estimator.setNumberOfPartitionBins(number_of_partition_bins);
	feature_estimator.setNumberOfRotations(number_of_rotations);
	feature_estimator.setSupportRadius(support_radius);
	feature_estimator.compute(*cloud_feature);

}

void pclFeatureDesp_MomentOfInertial(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin)
{
	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud_origin);
	feature_extractor.compute();

	//特征计算
	std::vector <float> moment_of_inertia;								//存储惯性矩的特征向量
	std::vector <float> eccentricity;									//存储偏心率的特征向量
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	feature_extractor.getMomentOfInertia(moment_of_inertia);										//惯性矩特征
	feature_extractor.getEccentricity(eccentricity);												//偏心率特征
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);										//AABB对应的左下角和右上角坐标；输入点云的AABB与全局坐标系对应
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);	//OBB对应的相关参数；输入点云的OBB其实就是以构造的局部坐标系对应的AABB
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);						//三个特征值
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);					//三个特征向量
	feature_extractor.getMassCenter(mass_center);													//点云中心坐标

	//特征可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("点云库PCL学习教程第二版-基于惯性矩与偏心率的描述子"));
	viewer->setBackgroundColor(1, 1, 1);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->addPointCloud<pcl::PointXYZ>(cloud_origin, 
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_origin, 0, 255, 0), "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
	viewer->addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, 
		max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, "AABB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, "AABB");
	Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
	std::cout << "position_OBB: " << position_OBB << endl;
	std::cout << "mass_center: " << mass_center << endl;
	Eigen::Quaternionf quat(rotational_matrix_OBB);
	viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, 
		max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, "OBB");
	viewer->setRepresentationToWireframeForAllActors();
	pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis(major_vector(0) + mass_center(0),
		major_vector(1) + mass_center(1), 
		major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), 
		middle_vector(1) + mass_center(1), 
		middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), 
		minor_vector(1) + mass_center(1), 
		minor_vector(2) + mass_center(2));
	viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
	viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");
	std::cout << "size of cloud :" << cloud_origin->points.size() << endl;
	std::cout << "moment_of_inertia :" << moment_of_inertia.size() << endl;
	std::cout << "eccentricity :" << eccentricity.size() << endl;
	//Eigen::Vector3f p1 (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
	//Eigen::Vector3f p2 (min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
	//Eigen::Vector3f p3 (max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
	//Eigen::Vector3f p4 (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
	//Eigen::Vector3f p5 (min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
	//Eigen::Vector3f p6 (min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
	//Eigen::Vector3f p7 (max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
	//Eigen::Vector3f p8 (max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

	//p1 = rotational_matrix_OBB * p1 + position;
	//p2 = rotational_matrix_OBB * p2 + position;
	//p3 = rotational_matrix_OBB * p3 + position;
	//p4 = rotational_matrix_OBB * p4 + position;
	//p5 = rotational_matrix_OBB * p5 + position;
	//p6 = rotational_matrix_OBB * p6 + position;
	//p7 = rotational_matrix_OBB * p7 + position;
	//p8 = rotational_matrix_OBB * p8 + position;

	//pcl::PointXYZ pt1 (p1 (0), p1 (1), p1 (2));
	//pcl::PointXYZ pt2 (p2 (0), p2 (1), p2 (2));
	//pcl::PointXYZ pt3 (p3 (0), p3 (1), p3 (2));
	//pcl::PointXYZ pt4 (p4 (0), p4 (1), p4 (2));
	//pcl::PointXYZ pt5 (p5 (0), p5 (1), p5 (2));
	//pcl::PointXYZ pt6 (p6 (0), p6 (1), p6 (2));
	//pcl::PointXYZ pt7 (p7 (0), p7 (1), p7 (2));
	//pcl::PointXYZ pt8 (p8 (0), p8 (1), p8 (2));

	//viewer->addLine (pt1, pt2, 1.0, 0.0, 0.0, "1 edge");
	//viewer->addLine (pt1, pt4, 1.0, 0.0, 0.0, "2 edge");
	//viewer->addLine (pt1, pt5, 1.0, 0.0, 0.0, "3 edge");
	//viewer->addLine (pt5, pt6, 1.0, 0.0, 0.0, "4 edge");
	//viewer->addLine (pt5, pt8, 1.0, 0.0, 0.0, "5 edge");
	//viewer->addLine (pt2, pt6, 1.0, 0.0, 0.0, "6 edge");
	//viewer->addLine (pt6, pt7, 1.0, 0.0, 0.0, "7 edge");
	//viewer->addLine (pt7, pt8, 1.0, 0.0, 0.0, "8 edge");
	//viewer->addLine (pt2, pt3, 1.0, 0.0, 0.0, "9 edge");
	//viewer->addLine (pt4, pt8, 1.0, 0.0, 0.0, "10 edge");
	//viewer->addLine (pt3, pt4, 1.0, 0.0, 0.0, "11 edge");
	//viewer->addLine (pt3, pt7, 1.0, 0.0, 0.0, "12 edge");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void pclFeatureDesp_BounderEst(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary)
{
	float re = 1.0;
	float reforn = 1.0;

	pcl::PointCloud<pcl::Boundary> boundaries;
	pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundEst;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	//法线估计
	normEst.setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(cloud_origin));
	normEst.setRadiusSearch(reforn);								//设置估计法线的半径；设置为分辨率的10倍时，效果较好，主要是对于法线估计
	normEst.compute(*normals);

	//边界估计
	boundEst.setInputCloud(cloud_origin);
	boundEst.setInputNormals(normals);
	boundEst.setRadiusSearch(re);									//设置为分辨率的10倍，太小则内部的很多点都当成边界点了
	boundEst.setAngleThreshold(M_PI / 4);							//边界判断是的角度阈值，默认值为PI/2
	boundEst.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
	boundEst.compute(boundaries);									//返回值赋给pcl::Boundary的是bool型，
																	//pcl::Boundary等于0时，表示判断为非边界；大于0时为边界

	for (int i = 0; i < cloud_origin->points.size(); i++)
	{

		if (boundaries[i].boundary_point > 0)
		{
			cloud_boundary->push_back(cloud_origin->points[i]);		//存储估计为边界的点云数据
		}
	}


	//可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer> MView(new pcl::visualization::PCLVisualizer("点云库PCL从入门到精通案例"));

	int v1(0);
	MView->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	MView->setBackgroundColor(0.3, 0.3, 0.3, v1);
	MView->addText("Raw point clouds", 10, 10, "v1_text", v1);
	int v2(0);
	MView->createViewPort(0.5, 0.0, 1, 1.0, v2);
	MView->setBackgroundColor(0.5, 0.5, 0.5, v2);
	MView->addText("Boudary point clouds", 10, 10, "v2_text", v2);

	MView->addPointCloud<pcl::PointXYZ>(cloud_origin, "sample cloud", v1);
	MView->addPointCloud<pcl::PointXYZ>(cloud_boundary, "cloud_boundary", v2);
	MView->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "sample cloud", v1);
	MView->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "cloud_boundary", v2);
	MView->addCoordinateSystem(1.0);
	MView->initCameraParameters();
	MView->spin();
}

