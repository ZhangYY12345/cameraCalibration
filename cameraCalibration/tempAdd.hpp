#pragma once
#include <core/types.hpp>

template<typename _Tp>
bool operator < (const cv::Point_<_Tp>& left, const cv::Point_<_Tp>& right)
{
	if (left.x < right.x)
	{
		return true;
	}
	if (left.x == right.x && left.y < right.y)
	{
		return true;
	}

	return false;
}
