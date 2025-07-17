cmake_minimum_required(VERSION 3.10)
project(vesin_cornerstone LANGUAGES CXX CUDA)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

#OPTION(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)
SET(CUDA_SEPARABLE_COMPILATION ON)

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

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

add_library(vesin_cornerstone SHARED
    "src/torch_cornerstone.cpp"
)

target_link_libraries(vesin_cornerstone "${TORCH_LIBRARIES}")

target_compile_features(vesin_cornerstone PRIVATE cxx_std_17)

set_target_properties(vesin_cornerstone PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vesin_cornerstone PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(vesin_cornerstone PROPERTIES CUDA_ARCHITECTURES native)

set(CMAKE_CUDA_STANDARD 17)

install(TARGETS vesin_cornerstone
    LIBRARY DESTINATION "lib"
)