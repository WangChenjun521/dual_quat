#include "nn_cuda.h"

void backproject_depth_float(py::array_t<float>& image_in, py::array_t<float>& point_image_out,
                             float fx, float fy, float cx, float cy) {
	assert(image_in.ndim() == 2);
	assert(point_image_out.ndim() == 3);

    int height = image_in.shape(0);
	int width = image_in.shape(1);

	assert(point_image_out.shape(0) == height);
	assert(point_image_out.shape(1) == width);
	assert(point_image_out.shape(2) ==  3 );

#pragma omp parallel for default(none) shared(image_in, point_image_out) firstprivate(height, width, fx, fy, cx, cy)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float depth = *image_in.data(y, x);

			if (depth > 0) {
				float pos_x = depth * (static_cast<float>(x) - cx) / fx;
				float pos_y = depth * (static_cast<float>(y) - cy) / fy;
				float pos_z = depth;

				*point_image_out.mutable_data(y, x, 0) = pos_x;
				*point_image_out.mutable_data(y, x, 1) = pos_y;
				*point_image_out.mutable_data(y, x, 2) = pos_z;
			}
		}
	}
}