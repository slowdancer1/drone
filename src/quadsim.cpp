#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <env.hpp>

PYBIND11_MODULE(quadsim, m) {
    py::class_<Env>(m, "Env")
        .def(py::init<int>())
        .def("render", &Env::render)
        .def("set_obstacles", &Env::set_obstacles);
}
