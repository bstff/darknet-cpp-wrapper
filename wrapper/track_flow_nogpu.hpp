#ifndef _DARKNET_WRAPPER_CV_TRACK_FLOW_NO_GPU_HPP_
#define _DARKNET_WRAPPER_CV_TRACK_FLOW_NO_GPU_HPP_

#if defined(TRACK_OPTFLOW) && defined(OPENCV)

//#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>

#include "box_image.h"

class Tracker_optflow {
public:
	const int flow_error;


	Tracker_optflow(int win_size = 9, int max_level = 3, int iterations = 8000, int _flow_error = -1) :
		flow_error((_flow_error > 0)? _flow_error:(win_size*4))
	{
		sync_PyrLKOpticalFlow = cv::SparsePyrLKOpticalFlow::create();
		sync_PyrLKOpticalFlow->setWinSize(cv::Size(win_size, win_size));	// 9, 15, 21, 31
		sync_PyrLKOpticalFlow->setMaxLevel(max_level);		// +- 3 pt

	}

	// just to avoid extra allocations
	cv::Mat dst_grey;
	cv::Mat prev_pts_flow, cur_pts_flow;
	cv::Mat status, err;

	cv::Mat src_grey;	// used in both functions
	cv::Ptr<cv::SparsePyrLKOpticalFlow> sync_PyrLKOpticalFlow;

	std::vector<bbox_t> cur_bbox_vec;
	std::vector<bool> good_bbox_vec_flags;

	void update_cur_bbox_vec(std::vector<bbox_t> _cur_bbox_vec)
	{
		cur_bbox_vec = _cur_bbox_vec;
		good_bbox_vec_flags = std::vector<bool>(cur_bbox_vec.size(), true);
		cv::Mat prev_pts, cur_pts_flow;

		for (auto &i : cur_bbox_vec) {
			float x_center = (i.x + i.w / 2.0F);
			float y_center = (i.y + i.h / 2.0F);
			prev_pts.push_back(cv::Point2f(x_center, y_center));
		}

		if (prev_pts.rows == 0)
			prev_pts_flow = cv::Mat();
		else
			cv::transpose(prev_pts, prev_pts_flow);
	}


	void update_tracking_flow(cv::Mat new_src_mat, std::vector<bbox_t> _cur_bbox_vec)
	{
		if (new_src_mat.channels() == 3) {

			update_cur_bbox_vec(_cur_bbox_vec);

			cv::cvtColor(new_src_mat, src_grey, CV_BGR2GRAY, 1);
		}
	}


	std::vector<bbox_t> tracking_flow(cv::Mat new_dst_mat, bool check_error = true)
	{
		if (sync_PyrLKOpticalFlow.empty()) {
			std::cout << "sync_PyrLKOpticalFlow isn't initialized \n";
			return cur_bbox_vec;
		}

		cv::cvtColor(new_dst_mat, dst_grey, CV_BGR2GRAY, 1);

		if (src_grey.rows != dst_grey.rows || src_grey.cols != dst_grey.cols) {
			src_grey = dst_grey.clone();
			return cur_bbox_vec;
		}

		if (prev_pts_flow.cols < 1) {
			return cur_bbox_vec;
		}

		////sync_PyrLKOpticalFlow_gpu.sparse(src_grey_gpu, dst_grey_gpu, prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, &err_gpu);	// OpenCV 2.4.x
		sync_PyrLKOpticalFlow->calc(src_grey, dst_grey, prev_pts_flow, cur_pts_flow, status, err);	// OpenCV 3.x

		dst_grey.copyTo(src_grey);

		std::vector<bbox_t> result_bbox_vec;

		if (err.rows == cur_bbox_vec.size() && status.rows == cur_bbox_vec.size())
		{
			for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
			{
				cv::Point2f cur_key_pt = cur_pts_flow.at<cv::Point2f>(0, i);
				cv::Point2f prev_key_pt = prev_pts_flow.at<cv::Point2f>(0, i);

				float moved_x = cur_key_pt.x - prev_key_pt.x;
				float moved_y = cur_key_pt.y - prev_key_pt.y;

				if (abs(moved_x) < 100 && abs(moved_y) < 100 && good_bbox_vec_flags[i])
					if (err.at<float>(0, i) < flow_error && status.at<unsigned char>(0, i) != 0 &&
						((float)cur_bbox_vec[i].x + moved_x) > 0 && ((float)cur_bbox_vec[i].y + moved_y) > 0)
					{
						cur_bbox_vec[i].x += moved_x + 0.5;
						cur_bbox_vec[i].y += moved_y + 0.5;
						result_bbox_vec.push_back(cur_bbox_vec[i]);
					}
					else good_bbox_vec_flags[i] = false;
				else good_bbox_vec_flags[i] = false;

				//if(!check_error && !good_bbox_vec_flags[i]) result_bbox_vec.push_back(cur_bbox_vec[i]);
			}
		}

		prev_pts_flow = cur_pts_flow.clone();

		return result_bbox_vec;
	}

};
#else

class Tracker_optflow {};

#endif	// defined(TRACK_OPTFLOW) && defined(OPENCV)

#endif