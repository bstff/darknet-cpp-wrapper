#ifndef _DARKNET_WRAPPER_CV_TRACK_FLOW_GPU_HPP_
#define _DARKNET_WRAPPER_CV_TRACK_FLOW_GPU_HPP_


#if defined(TRACK_OPTFLOW) && defined(OPENCV) && defined(GPU)

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

#include "box_image.h"

class Tracker_optflow {
public:
	const int gpu_count;
	const int gpu_id;
	const int flow_error;


	Tracker_optflow(int _gpu_id = 0, int win_size = 9, int max_level = 3, int iterations = 8000, int _flow_error = -1) :
		gpu_count(cv::cuda::getCudaEnabledDeviceCount()), gpu_id(std::min(_gpu_id, gpu_count-1)),
		flow_error((_flow_error > 0)? _flow_error:(win_size*4))
	{
		int const old_gpu_id = cv::cuda::getDevice();
		cv::cuda::setDevice(gpu_id);

		stream = cv::cuda::Stream();

		sync_PyrLKOpticalFlow_gpu = cv::cuda::SparsePyrLKOpticalFlow::create();
		sync_PyrLKOpticalFlow_gpu->setWinSize(cv::Size(win_size, win_size));	// 9, 15, 21, 31
		sync_PyrLKOpticalFlow_gpu->setMaxLevel(max_level);		// +- 3 pt
		sync_PyrLKOpticalFlow_gpu->setNumIters(iterations);	// 2000, def: 30

		cv::cuda::setDevice(old_gpu_id);
	}

	// just to avoid extra allocations
	cv::cuda::GpuMat src_mat_gpu;
	cv::cuda::GpuMat dst_mat_gpu, dst_grey_gpu;
	cv::cuda::GpuMat prev_pts_flow_gpu, cur_pts_flow_gpu;
	cv::cuda::GpuMat status_gpu, err_gpu;

	cv::cuda::GpuMat src_grey_gpu;	// used in both functions
	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> sync_PyrLKOpticalFlow_gpu;
	cv::cuda::Stream stream;

	std::vector<bbox_t> cur_bbox_vec;
	std::vector<bool> good_bbox_vec_flags;
	cv::Mat prev_pts_flow_cpu;

	void update_cur_bbox_vec(std::vector<bbox_t> _cur_bbox_vec)
	{
		cur_bbox_vec = _cur_bbox_vec;
		good_bbox_vec_flags = std::vector<bool>(cur_bbox_vec.size(), true);
		cv::Mat prev_pts, cur_pts_flow_cpu;

		for (auto &i : cur_bbox_vec) {
			float x_center = (i.x + i.w / 2.0F);
			float y_center = (i.y + i.h / 2.0F);
			prev_pts.push_back(cv::Point2f(x_center, y_center));
		}

		if (prev_pts.rows == 0)
			prev_pts_flow_cpu = cv::Mat();
		else
			cv::transpose(prev_pts, prev_pts_flow_cpu);

		if (prev_pts_flow_gpu.cols < prev_pts_flow_cpu.cols) {
			prev_pts_flow_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), prev_pts_flow_cpu.type());
			cur_pts_flow_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), prev_pts_flow_cpu.type());

			status_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), CV_8UC1);
			err_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), CV_32FC1);
		}

		prev_pts_flow_gpu.upload(cv::Mat(prev_pts_flow_cpu), stream);
	}


	void update_tracking_flow(cv::Mat src_mat, std::vector<bbox_t> _cur_bbox_vec)
	{
		int const old_gpu_id = cv::cuda::getDevice();
		if (old_gpu_id != gpu_id)
			cv::cuda::setDevice(gpu_id);

		if (src_mat.channels() == 3) {
			if (src_mat_gpu.cols == 0) {
				src_mat_gpu = cv::cuda::GpuMat(src_mat.size(), src_mat.type());
				src_grey_gpu = cv::cuda::GpuMat(src_mat.size(), CV_8UC1);
			}

			update_cur_bbox_vec(_cur_bbox_vec);

			//src_grey_gpu.upload(src_mat, stream);	// use BGR
			src_mat_gpu.upload(src_mat, stream);
			cv::cuda::cvtColor(src_mat_gpu, src_grey_gpu, CV_BGR2GRAY, 1, stream);
		}
		if (old_gpu_id != gpu_id)
			cv::cuda::setDevice(old_gpu_id);
	}


	std::vector<bbox_t> tracking_flow(cv::Mat dst_mat, bool check_error = true)
	{
		if (sync_PyrLKOpticalFlow_gpu.empty()) {
			std::cout << "sync_PyrLKOpticalFlow_gpu isn't initialized \n";
			return cur_bbox_vec;
		}

		int const old_gpu_id = cv::cuda::getDevice();
		if(old_gpu_id != gpu_id)
			cv::cuda::setDevice(gpu_id);

		if (dst_mat_gpu.cols == 0) {
			dst_mat_gpu = cv::cuda::GpuMat(dst_mat.size(), dst_mat.type());
			dst_grey_gpu = cv::cuda::GpuMat(dst_mat.size(), CV_8UC1);
		}

		//dst_grey_gpu.upload(dst_mat, stream);	// use BGR
		dst_mat_gpu.upload(dst_mat, stream);
		cv::cuda::cvtColor(dst_mat_gpu, dst_grey_gpu, CV_BGR2GRAY, 1, stream);

		if (src_grey_gpu.rows != dst_grey_gpu.rows || src_grey_gpu.cols != dst_grey_gpu.cols) {
			stream.waitForCompletion();
			src_grey_gpu = dst_grey_gpu.clone();
			cv::cuda::setDevice(old_gpu_id);
			return cur_bbox_vec;
		}

		////sync_PyrLKOpticalFlow_gpu.sparse(src_grey_gpu, dst_grey_gpu, prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, &err_gpu);	// OpenCV 2.4.x
		sync_PyrLKOpticalFlow_gpu->calc(src_grey_gpu, dst_grey_gpu, prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, err_gpu, stream);	// OpenCV 3.x

		cv::Mat cur_pts_flow_cpu;
		cur_pts_flow_gpu.download(cur_pts_flow_cpu, stream);

		dst_grey_gpu.copyTo(src_grey_gpu, stream);

		cv::Mat err_cpu, status_cpu;
		err_gpu.download(err_cpu, stream);
		status_gpu.download(status_cpu, stream);

		stream.waitForCompletion();

		std::vector<bbox_t> result_bbox_vec;

		if (err_cpu.cols == cur_bbox_vec.size() && status_cpu.cols == cur_bbox_vec.size()) 
		{
			for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
			{
				cv::Point2f cur_key_pt = cur_pts_flow_cpu.at<cv::Point2f>(0, i);
				cv::Point2f prev_key_pt = prev_pts_flow_cpu.at<cv::Point2f>(0, i);

				float moved_x = cur_key_pt.x - prev_key_pt.x;
				float moved_y = cur_key_pt.y - prev_key_pt.y;

				if (abs(moved_x) < 100 && abs(moved_y) < 100 && good_bbox_vec_flags[i])
					if (err_cpu.at<float>(0, i) < flow_error && status_cpu.at<unsigned char>(0, i) != 0 &&
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

		cur_pts_flow_gpu.swap(prev_pts_flow_gpu);
		cur_pts_flow_cpu.copyTo(prev_pts_flow_cpu);

		if (old_gpu_id != gpu_id)
			cv::cuda::setDevice(old_gpu_id);

		return result_bbox_vec;
	}

};

#endif //defined(TRACK_OPTFLOW) && defined(OPENCV) && defined(GPU)

#endif