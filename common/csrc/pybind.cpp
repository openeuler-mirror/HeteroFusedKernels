#include <pybind11/pybind11.h>
#include "mem_alloc.h"
#include "managed_mem.h"

namespace py = pybind11;

PYBIND11_MODULE(memory, m) {
    m.def("alloc_numa_pinned_tensor", &alloc_numa_pinned_tensor);
    m.def("host_register", &register_tensor);
    m.def("get_device_ptr", &get_device_ptr);
}