# cpp-2021-vulkan

cmake_minimum_required(VERSION 3.20.0)

# See https://github.com/apache/arrow/blob/master/cpp/cmake_modules/Usevcpkg.cmake

# cpp-2021-vulkan

project(cpp-2021-vulkan VERSION 0.0.1)

include(CTest)
enable_testing()

add_executable(cpp-2021-vulkan main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
