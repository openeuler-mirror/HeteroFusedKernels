cmake_minimum_required(VERSION 3.16.0)
project(pcie_through_C)
set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(RPATH_VALUE $ORIGIN)

message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
set(_GLIBCXX_USE_CXX11_ABI ${GLIBCXX_USE_CXX11_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")

include(utils.cmake)
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11")
find_package(pybind11 REQUIRED)

# user-defined configuration
set(SOC_VERSION ${SOC_VERSION} CACHE STRING "system on chip type")
message(STATUS "SOC_VERSION is: ${SOC_VERSION}")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH_SUBDIR "x86_64-linux")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()


if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
FILE(GLOB HOST_SRCS
   csrc/host/*.cpp
   csrc/host/tiling/*.cpp)
message(STATUS "HOST SRCS: ${HOST_SRCS}")

FILE(GLOB KERNEL_SRCS
   csrc/kernel/*.cpp)
message(STATUS "KERNELS SRCS: ${KERNEL_SRCS}")


set(LIB_NAME pcie_through)
ascendc_library(${LIB_NAME}_kernels STATIC
    ${KERNEL_SRCS}
)

ascendc_compile_definitions(${LIB_NAME}_kernels PRIVATE
  -DHAVE_TILING
)

add_library(${LIB_NAME} SHARED ${HOST_SRCS})
if (NOT HETERO_FUSED_KERNELS_PATH)
message(FATAL_ERROR "HeteroFusedKernels does not exist, please check whether the HeteroFusedKernels package is installed.")
endif()
message(STATUS "HeteroFusedKernels Path: ${HETERO_FUSED_KERNELS_PATH}")

target_link_libraries(${LIB_NAME} PRIVATE
  managed_memory
  ${LIB_NAME}_kernels
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
)

target_link_directories(${LIB_NAME} PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
  ${HETERO_FUSED_KERNELS_PATH}
)

target_include_directories(${LIB_NAME} PRIVATE
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${HETERO_FUSED_KERNELS_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/platform
  ${TORCH_NPU_PATH}/include
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)
