#ifndef _DARKNET_WRAPPER_PREVIEW_HPP_
#define _DARKNET_WRAPPER_PREVIEW_HPP_

#ifdef __cplusplus
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#endif

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/highgui/highgui_c.h"	// C
#include "opencv2/imgproc/imgproc_c.h"	// C
#endif	// OPENCV

#include "box_image.h"

#ifdef OPENCV

static cv::Scalar obj_id_to_color(int obj_id) {
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}

class preview_boxes_t {
	enum { frames_history = 30 };	// how long to keep the history saved

	struct preview_box_track_t {
		unsigned int track_id, obj_id, last_showed_frames_ago;
		bool current_detection;
		bbox_t bbox;
		cv::Mat mat_obj, mat_resized_obj;
		preview_box_track_t() : track_id(0), obj_id(0), last_showed_frames_ago(frames_history), current_detection(false) {}
	};
	std::vector<preview_box_track_t> preview_box_track_id;
	size_t const preview_box_size, bottom_offset;
	bool const one_off_detections;
public:
	preview_boxes_t(size_t _preview_box_size = 100, size_t _bottom_offset = 100, bool _one_off_detections = false) :
		preview_box_size(_preview_box_size), bottom_offset(_bottom_offset), one_off_detections(_one_off_detections)
	{}

	void set(cv::Mat src_mat, std::vector<bbox_t> result_vec)
	{
		size_t const count_preview_boxes = src_mat.cols / preview_box_size;
		if (preview_box_track_id.size() != count_preview_boxes) preview_box_track_id.resize(count_preview_boxes);

		// increment frames history
		for (auto &i : preview_box_track_id)
			i.last_showed_frames_ago = std::min((unsigned)frames_history, i.last_showed_frames_ago + 1);

		// occupy empty boxes
		for (auto &k : result_vec) {
			bool found = false;
			// find the same (track_id)
			for (auto &i : preview_box_track_id) {
				if (i.track_id == k.track_id) {
					if (!one_off_detections) i.last_showed_frames_ago = 0; // for tracked objects
					found = true;
					break;
				}
			}
			if (!found) {
				// find empty box
				for (auto &i : preview_box_track_id) {
					if (i.last_showed_frames_ago == frames_history) {
						if (!one_off_detections && k.frames_counter == 0) break; // don't show if obj isn't tracked yet
						i.track_id = k.track_id;
						i.obj_id = k.obj_id;
						i.bbox = k;
						i.last_showed_frames_ago = 0;
						break;
					}
				}
			}
		}

		// draw preview box (from old or current frame)
		for (size_t i = 0; i < preview_box_track_id.size(); ++i)
		{
			// get object image
			cv::Mat dst = preview_box_track_id[i].mat_resized_obj;
			preview_box_track_id[i].current_detection = false;

			for (auto &k : result_vec) {
				if (preview_box_track_id[i].track_id == k.track_id) {
					if (one_off_detections && preview_box_track_id[i].last_showed_frames_ago > 0) {
						preview_box_track_id[i].last_showed_frames_ago = frames_history; break;
					}
					bbox_t b = k;
					cv::Rect r(b.x, b.y, b.w, b.h);
					cv::Rect img_rect(cv::Point2i(0, 0), src_mat.size());
					cv::Rect rect_roi = r & img_rect;
					if (rect_roi.width > 1 || rect_roi.height > 1) {
						cv::Mat roi = src_mat(rect_roi);
						cv::resize(roi, dst, cv::Size(preview_box_size, preview_box_size), cv::INTER_NEAREST);
						preview_box_track_id[i].mat_obj = roi.clone();
						preview_box_track_id[i].mat_resized_obj = dst.clone();
						preview_box_track_id[i].current_detection = true;
						preview_box_track_id[i].bbox = k;
					}
					break;
				}
			}
		}
	}


	void draw(cv::Mat draw_mat, bool show_small_boxes = false)
	{
		// draw preview box (from old or current frame)
		for (size_t i = 0; i < preview_box_track_id.size(); ++i)
		{
			auto &prev_box = preview_box_track_id[i];

			// draw object image
			cv::Mat dst = prev_box.mat_resized_obj;
			if (prev_box.last_showed_frames_ago < frames_history &&
				dst.size() == cv::Size(preview_box_size, preview_box_size))
			{
				cv::Rect dst_rect_roi(cv::Point2i(i * preview_box_size, draw_mat.rows - bottom_offset), dst.size());
				cv::Mat dst_roi = draw_mat(dst_rect_roi);
				dst.copyTo(dst_roi);

				cv::Scalar color = obj_id_to_color(prev_box.obj_id);
				int thickness = (prev_box.current_detection) ? 5 : 1;
				cv::rectangle(draw_mat, dst_rect_roi, color, thickness);

				unsigned int const track_id = prev_box.track_id;
				std::string track_id_str = (track_id > 0) ? std::to_string(track_id) : "";
				putText(draw_mat, track_id_str, dst_rect_roi.tl() - cv::Point2i(-4, 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, cv::Scalar(0, 0, 0), 2);

				std::string size_str = std::to_string(prev_box.bbox.w) + "x" + std::to_string(prev_box.bbox.h);
				putText(draw_mat, size_str, dst_rect_roi.tl() + cv::Point2i(0, 12), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);

				if (!one_off_detections && prev_box.current_detection) {
					cv::line(draw_mat, dst_rect_roi.tl() + cv::Point2i(preview_box_size, 0),
						cv::Point2i(prev_box.bbox.x, prev_box.bbox.y + prev_box.bbox.h),
						color);
				}

				if (one_off_detections && show_small_boxes) {
					cv::Rect src_rect_roi(cv::Point2i(prev_box.bbox.x, prev_box.bbox.y),
						cv::Size(prev_box.bbox.w, prev_box.bbox.h));
					unsigned int const color_history = (255 * prev_box.last_showed_frames_ago) / frames_history;
					color = cv::Scalar(255 - 3 * color_history, 255 - 2 * color_history, 255 - 1 * color_history);
					if (prev_box.mat_obj.size() == src_rect_roi.size()) {
						prev_box.mat_obj.copyTo(draw_mat(src_rect_roi));
					}
					cv::rectangle(draw_mat, src_rect_roi, color, thickness);
					putText(draw_mat, track_id_str, src_rect_roi.tl() - cv::Point2i(0, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
				}
			}
		}
	}
};
#endif	// OPENCV

#endif