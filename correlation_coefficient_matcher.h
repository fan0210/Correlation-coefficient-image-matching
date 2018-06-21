#ifndef CORRELATION_COEFFICIENT_MATCHER_H_
#define CORRELATION_COEFFICIENT_MATCHER_H_

#include <opencv.hpp>
#include <random>

class CorrelationCoefficientMatcher
{
public:
	//描述匹配的类型
	class Matcher
	{
	public:
		Matcher() = default;
		Matcher(const cv::Point &pt_l, const cv::Point &pt_r, double index) :m_ptl(pt_l), m_ptr(pt_r), correlationIndex(index) {}

		const cv::Point &getPt_l()const { return m_ptl; }
		const cv::Point &getPt_r()const { return m_ptr; }
		const double &getCorrelationIndex()const { return correlationIndex; }

	private:
		cv::Point m_ptl;   //左影像上的点
		cv::Point m_ptr;   //右影像上的点

		double correlationIndex = 0;     //相关系数
	};

	CorrelationCoefficientMatcher &setWinSize(int width, int height)    //设置窗口大小
	{ 
		m_winWidth = width % 2 == 0 ? width + 1 : width;
		m_winHeight = height % 2 == 0 ? height + 1 : height;

		return *this;
	}

	/*
	* 检测得到匹配结果
	* @ srcImg是左影像，已知点位于该影像内
	* @ searchImg是右影像，程序需要找出左影像中的已知点对应于右影像中的同名点
	* @ pts_l 是左影像中的已知点数组，程序需要找出左影像中每个已知点对应于右影像中的同名点
	*/
	const std::vector<Matcher> &detectMatches(const cv::Mat &srcImg, const cv::Mat &searchImg, const std::vector<cv::Point> &pts_l);
	
	//单个点做匹配
	//可用于多线程调用
	void singleThreadSearch(const cv::Mat &srcImg, const cv::Mat &searchImg, const cv::Point &pt_l);
	
	//获得匹配结果
	const std::vector<Matcher> &getMatchers()const { return m_matchers; }

	//显示匹配结果
	//两张图像与detectMatches中相同
	void drawMatchers(const cv::Mat &srcImg, const cv::Mat &searchImg)const;
private:
	int m_winWidth = 11, m_winHeight = 11;
	std::vector<Matcher> m_matchers;

	//计算相关系数
	double getCorrelationIndex(const cv::Mat &, const cv::Mat &, const cv::Rect &)const;

	//判断点是不是可以有足够的邻域点
	bool isPointLValid(const cv::Mat &, const cv::Point &)const;
};


#endif