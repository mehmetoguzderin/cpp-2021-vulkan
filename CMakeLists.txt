# cpp-2021-vulkan

cmake_minimum_required(VERSION 3.20.0)

# See https://github.com/apache/arrow/blob/master/cpp/cmake_modules/Usevcpkg.cmake

# cpp-2021-vulkan

project(cpp-2021-vulkan VERSION 0.0.1)

include(CTest)
enable_testing()

add_executable(cpp-2021-vulkan main.cpp)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE benchmark::benchmark benchmark::benchmark_main)

find_path(CGLTF_INCLUDE_DIRS "cgltf.h")
target_include_directories(cpp-2021-vulkan PRIVATE ${CGLTF_INCLUDE_DIRS})

find_package(EnTT CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE EnTT::EnTT)

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE nlohmann_json nlohmann_json::nlohmann_json)

find_package(pugixml CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE pugixml pugixml::static pugixml::pugixml)

find_package(unofficial-sqlite3 CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE unofficial::sqlite3::sqlite3)

find_package(zstd CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE zstd::libzstd_static)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
