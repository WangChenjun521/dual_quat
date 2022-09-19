#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n);
void backproject_depth_float(py::array_t<float>& image_in, py::array_t<float>& point_image_out,
                             float fx, float fy, float cx, float cy);