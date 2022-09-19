#include <torch/extension.h>
#include "nn_cuda.h"

void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
    m.def("launch_add2",
          &launch_add2,
          "add2 kernel warpper");
    m.def("backproject_depth_float",
          &backproject_depth_float,
          "backproject_depth_float");
}

TORCH_LIBRARY(nn_cuda, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}