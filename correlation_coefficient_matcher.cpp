#include "correlation_coefficient_matcher.h"

void CorrelationCoefficientMatcher::drawMatchers(const cv::Mat &srcImg, const cv::Mat &searchImg)const
{
	cv::Mat match;
    match.create(cv::Size(srcImg.cols + searchImg.cols, std::max(srcImg.rows,searchImg.rows)), CV_8UC3);

	for (auto i = 0; i < match.rows; ++i)
	{
		unsigned char *data = match.ptr<unsigned char>(i);
		for (auto j = 0; j < match.cols; ++j)
			for (int k = 0; k < 3; ++k)
				data[j * 3 + k] = 0;
	}

	cv::Mat srcImg_ = srcImg.clone();
	cv::Mat searchImg_ = searchImg.clone();
	if (srcImg.channels() == 1)
		cv::cvtColor(srcImg_, srcImg_, CV_GRAY2BGR);
	if (searchImg.channels() == 1)
		cv::cvtColor(searchImg_, searchImg_, CV_GRAY2BGR);

	for (auto i = 0; i < srcImg_.rows; ++i)
	{
		const unsigned char *data = srcImg_.ptr<unsigned char>(i);
		unsigned char *src_data = match.ptr<unsigned char>(i);

		for (auto j = 0; j < srcImg_.cols; ++j)
			for (int k = 0; k < 3; ++k)
				src_data[j * 3 + k] = data[j * 3 + k];
	}

	for (auto i = 0; i < searchImg_.rows; ++i)
	{
		const unsigned char *data = searchImg_.ptr<unsigned char>(i);
		unsigned char *src_data = match.ptr<unsigned char>(i);

		for (auto j = 0; j < searchImg_.cols; ++j)
			for (int k = 0; k < 3; ++k)
				src_data[(j + srcImg_.cols)*3+k] = data[j * 3 + k];
	}

	const int x_differ = srcImg_.cols;

	for (auto it = m_matchers.cbegin(); it != m_matchers.cend(); ++it)
	{
		const cv::Point pt_l = it->getPt_l();
		const cv::Point pt_r = cv::Point(it->getPt_r().x + x_differ, it->getPt_r().y);

		static std::default_random_engine e;
		static std::uniform_int_distribution<int> u(0, 255);

		cv::Scalar color(u(e), u(e), u(e));

		cv::circle(match, pt_l, 2, color);
		cv::circle(match, pt_r, 2, color);

		cv::line(match, pt_l, pt_r, color);
	}

	cv::imshow("Matches", match);
	cv::waitKey(0);
}

void CorrelationCoefficientMatcher::singleThreadSearch(const cv::Mat &srcImg, const cv::Mat &searchImg, const cv::Point &pt_l)
{
	cv::Mat srcImg_ = srcImg.clone();
	cv::Mat searchImg_ = searchImg.clone();

	if (srcImg.channels() != 1)
		cv::cvtColor(srcImg, srcImg_, CV_BGR2GRAY);
	if (searchImg.channels() != 1)
		cv::cvtColor(searchImg, searchImg_, CV_BGR2GRAY);

	if (isPointLValid(srcImg_, pt_l))
	{
		cv::Rect r(cv::Point(pt_l.x - m_winWidth / 2, pt_l.y - m_winHeight / 2), cv::Point(pt_l.x + m_winWidth / 2 + 1, pt_l.y + m_winHeight / 2 + 1));
		cv::Mat templateImg = cv::Mat(srcImg_, r).clone();

		struct PointWithIndex
		{
			double index;
			cv::Point pt;
		};
		std::vector<PointWithIndex> pts;
		for (auto i = m_winWidth / 2; i < searchImg_.cols - m_winWidth / 2; ++i)
		{
			for (auto j = m_winHeight / 2; j < searchImg_.rows - m_winHeight / 2; ++j)
			{
				double index = getCorrelationIndex(templateImg, searchImg_, cv::Rect(cv::Point(i - m_winWidth / 2, j - m_winHeight / 2), cv::Point(i + m_winWidth / 2, j + m_winHeight / 2)));
				if (index > 0.8)
					pts.push_back({ index,cv::Point(i,j) });
			}
		}

		Matcher m;
		for (auto it = pts.cbegin(); it != pts.cend(); ++it)
			m = it->index > m.getCorrelationIndex() ? Matcher(pt_l, it->pt, it->index) : m;

		if (m.getCorrelationIndex() != 0)
			m_matchers.push_back(m);
	}
}

double CorrelationCoefficientMatcher::getCorrelationIndex(const cv::Mat &template_l, const cv::Mat &win_r, const cv::Rect &region_r)const
{
	double g_lSum = 0, g_rSum = 0;
	for (auto i = 0; i < template_l.rows; ++i)
	{
		const unsigned char *data_l = template_l.ptr<unsigned char>(i);
		const unsigned char *data_r = win_r.ptr<unsigned char>(i+ region_r.y);
		for (auto j = 0; j < template_l.cols; ++j)
		{
			g_lSum += data_l[j];
			g_rSum += data_r[j + region_r.x];
		}
	}

	double g_l = g_lSum / (m_winWidth*m_winHeight);
	double g_r = g_rSum / (m_winWidth*m_winHeight);

	double numerator = 0;
	double denominator1 = 0;
	double denominator2 = 0;

	for (int i = 0; i < m_winHeight; ++i)
	{
		const unsigned char *data_l = template_l.ptr<unsigned char>(i);
		const unsigned char *data_r = win_r.ptr<unsigned char>(i + region_r.y);
		for (int j = 0; j < m_winWidth; ++j)
		{
			numerator += (data_l[j] - g_l)*(data_r[j + region_r.x] - g_r);
			denominator1 += (data_l[j] - g_l)*(data_l[j] - g_l);
			denominator2 += (data_r[j + region_r.x] - g_r)*(data_r[j + region_r.x] - g_r);
		}
	}

	double denominator = sqrt(denominator1* denominator2);

	return abs(numerator / denominator);
}

inline bool CorrelationCoefficientMatcher::isPointLValid(const cv::Mat &srcImg, const cv::Point &pt_l)const
{
	if (pt_l.x<m_winWidth / 2 || pt_l.y<m_winHeight / 2 || pt_l.x>=srcImg.cols - m_winWidth / 2 || pt_l.y>=srcImg.rows - m_winHeight / 2)
		return false;
	return true;
}

const std::vector<CorrelationCoefficientMatcher::Matcher> &CorrelationCoefficientMatcher::detectMatches(const cv::Mat &srcImg, const cv::Mat &searchImg, const std::vector<cv::Point> &pts_l)
{
	cv::Mat srcImg_ = srcImg.clone();
	cv::Mat searchImg_ = searchImg.clone();

	if (srcImg.channels() != 1)
		cv::cvtColor(srcImg, srcImg_, CV_BGR2GRAY);
	if (searchImg.channels() != 1)
		cv::cvtColor(searchImg, searchImg_, CV_BGR2GRAY);

	for (auto it = pts_l.cbegin(); it != pts_l.cend(); ++it)
		singleThreadSearch(srcImg_, searchImg_, *it);

	return m_matchers;
}