
file(GLOB MEM_SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

add_library(managed_memory SHARED ${MEM_SRC_FILES})

target_include_directories(managed_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_directories(
  managed_memory
  PRIVATE
  ${TORCH_LIBS_DIR}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
)

target_include_directories(managed_memory PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${PYTHON_INCLUDE_PATH}
    ${pybind11_INCLUDE_DIRS}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/ascendc/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/ascend_hal
)

target_link_libraries(managed_memory PRIVATE
    ${TORCH_LIBRARIES}
    libtorch_npu.so
    ascendcl
    platform
    ascend_hal
)

file(GLOB MEM_HEADER_FILES
${CMAKE_CURRENT_SOURCE_DIR}/*.h)

install(FILES 
    ${MEM_HEADER_FILES}
    DESTINATION include
)