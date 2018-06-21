#include <opencv.hpp>
#include <iostream>
#include <Windows.h>

#include "correlation_coefficient_matcher.h"


//#define BOOST_MULTITHREAD      //multithread demo, if boost is available.

#ifdef BOOST_MULTITHREAD
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
boost::mutex _mutex;
size_t threadOverNum = 0;
#endif

#ifdef BOOST_MULTITHREAD
void singleThread(CorrelationCoefficientMatcher *m, const cv::Mat &srcImg, const cv::Mat &searchImg, const cv::Point &pt_l)
{
	m->singleThreadSearch(srcImg, searchImg, pt_l);

	_mutex.lock();
	++threadOverNum;
	_mutex.unlock();
}
#endif

int main(int, char **)
{
	cv::Mat img_l = cv::imread("img_l.png");
	cv::Mat img_r = cv::imread("img_r.png");

	std::vector<cv::Point> pts{ cv::Point(112,48),cv::Point(258,17),cv::Point(347,151),cv::Point(428,251),cv::Point(255,333),cv::Point(454,299),cv::Point(531,349) };

	CorrelationCoefficientMatcher match;

	match.setWinSize(13, 13);       

	double t = cv::getTickCount();
#ifdef BOOST_MULTITHREAD
	for (auto it = pts.cbegin(); it != pts.cend(); ++it)
	{
		boost::thread t(&singleThread, &match, img_l, img_r, *it);
		t.detach();
	}
	while (threadOverNum != pts.size())Sleep(1);
#else
	match.detectMatches(img_l, img_r, pts);
#endif
	std::cout << "本次匹配共用时 " << (cv::getTickCount() - t) / cv::getTickFrequency() << " 秒" << std::endl;

	match.drawMatchers(img_l, img_r);

	system("pause");
	return 0;
}