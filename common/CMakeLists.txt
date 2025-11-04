# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)
# project information
project(memory)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.14")
  set(CMAKE_BUILD_RPATH "$ORIGIN")
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
  set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
  set(CMAKE_INSTALL_RPATH_USE_ORIGIN ON)
  set(CMAKE_SKIP_RPATH FALSE)
endif()

message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
set(_GLIBCXX_USE_CXX11_ABI ${GLIBCXX_USE_CXX11_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

# for the device kernels compilation
set(SOC_VERSION ${SOC_VERSION})
set(ARCH ${ARCH})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH_SUBDIR "x86_64-linux")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_subdirectory(csrc)

set(TORCH_LIBS_DIR "${TORCH_PATH}/lib")

target_link_options(memory PRIVATE
    "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
)

target_link_directories(
  memory
  PRIVATE
  ${TORCH_LIBS_DIR}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
)

target_link_libraries(
  memory
  PUBLIC
  managed_memory
  ${TORCH_LIBRARIES}
  libtorch_npu.so
  ascendcl
  platform
  ascend_hal
  numa
)


install(TARGETS memory managed_memory DESTINATION ${CMAKE_INSTALL_PREFIX})