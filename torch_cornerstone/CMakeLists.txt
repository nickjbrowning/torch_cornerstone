cmake_minimum_required(VERSION 3.10)
project(torch_cornerstone LANGUAGES CXX CUDA)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

find_package(MPI REQUIRED)
find_package(Python COMPONENTS Interpreter REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Could not find CUDA Compiler. Make sure $CUDA_HOME is set.")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
)

if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
message(FATAL_ERROR "failed to find your pytorch installation, error: ${TORCH_CMAKE_PATH_ERROR}\n")
endif()

string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")

find_package(Torch 1.13 REQUIRED)

set(BUILD_TESTING OFF CACHE BOOL "Disable cornerstone tests" FORCE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -ccbin=mpicxx")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_SOURCE_DIR}/cornerstone)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cornerstone/include
)

add_library(torch_cornerstone SHARED
    "src/torch_cornerstone.cpp"
)

target_link_libraries(torch_cornerstone "${TORCH_LIBRARIES}" cstone_gpu MPI::MPI_CXX)

target_compile_features(torch_cornerstone PRIVATE cxx_std_20)

set_target_properties(torch_cornerstone PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(torch_cornerstone PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_STANDARD 17)

install(TARGETS torch_cornerstone
    LIBRARY DESTINATION "lib"
)