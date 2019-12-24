#include "board.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_board, m) {
  m.doc() = "class Board";
  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def("place", &Board::place)
      .def("__repr__",
           [](const Board &rhs) {
             std::ostringstream out;
             out << rhs;
             return out.str();
           })
      .def_property("features",
                    [](Board &b) {
                      return py::array({4, 9, 9}, b.get_features());
                    },
                    nullptr);
}