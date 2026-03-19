cmake_minimum_required(VERSION 3.14)
project(dpgo_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use system pybind11 headers directly (avoid conda pybind11 cmake which drags in conda compiler)
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_MODULE_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python3 -c "from sysconfig import get_paths; print(get_paths()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)

# DPGO paths
set(DPGO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../repos/dpgo")
set(DPGO_BUILD "${DPGO_ROOT}/build")
set(ROPTLIB_SRC "${DPGO_BUILD}/roptlib-src")

# Find DPGO shared library
find_library(DPGO_LIB NAMES DPGO PATHS "${DPGO_BUILD}/lib" NO_DEFAULT_PATH)
find_library(ROPTLIB_LIB NAMES roptlib PATHS "${DPGO_BUILD}/lib" NO_DEFAULT_PATH)

if(NOT DPGO_LIB)
    message(FATAL_ERROR "libDPGO.so not found in ${DPGO_BUILD}/lib. Build DPGO first.")
endif()
if(NOT ROPTLIB_LIB)
    message(FATAL_ERROR "libroptlib.so not found in ${DPGO_BUILD}/lib. Build DPGO first.")
endif()

# SuiteSparse (CHOLMOD, SPQR)
find_package(BLAS REQUIRED)

# Build as shared library (pybind11 module)
add_library(dpgo_pybind SHARED dpgo_pybind.cpp)
set_target_properties(dpgo_pybind PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_SUFFIX}"
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(dpgo_pybind PRIVATE
    "${PYTHON_INCLUDE}"
    "${PYBIND11_INCLUDE}"
    "${DPGO_ROOT}/include"
    "${ROPTLIB_SRC}"
    "${ROPTLIB_SRC}/Manifolds"
    "${ROPTLIB_SRC}/Others"
    "${ROPTLIB_SRC}/Problems"
    "${ROPTLIB_SRC}/Solvers"
    "${ROPTLIB_SRC}/cwrapper"
    "${ROPTLIB_SRC}/cwrapper/blas"
    "${ROPTLIB_SRC}/cwrapper/lapack"
    ${EIGEN3_INCLUDE_DIR}
    "/usr/include/suitesparse"
)

target_link_libraries(dpgo_pybind PRIVATE
    ${DPGO_LIB}
    ${ROPTLIB_LIB}
    ${BLAS_LIBRARIES}
    cholmod
    spqr
)

# Set RPATH so it finds libDPGO.so and libroptlib.so at runtime
set_target_properties(dpgo_pybind PROPERTIES
    INSTALL_RPATH "${DPGO_BUILD}/lib"
    BUILD_RPATH "${DPGO_BUILD}/lib"
)
