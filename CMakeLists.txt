# cpp-2021-vulkan

cmake_minimum_required(VERSION 3.21.0)

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

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE glfw)

find_package(glm CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE glm::glm)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE imgui::imgui)

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE nlohmann_json nlohmann_json::nlohmann_json)

find_package(pugixml CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE pugixml pugixml::static pugixml::pugixml)

find_package(sdl2-image CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE SDL2::SDL2_image)

find_path(SPIRV_HEADERS_INCLUDE_DIRS "spirv/1.0/GLSL.std.450.h")
target_include_directories(cpp-2021-vulkan PRIVATE ${SPIRV_HEADERS_INCLUDE_DIRS})

add_library(SPIRV-Tools INTERFACE)
find_package(SPIRV-Tools CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE SPIRV-Tools-static)

find_package(SPIRV-Tools-opt CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE SPIRV-Tools-opt)

find_package(unofficial-sqlite3 CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE unofficial::sqlite3::sqlite3)

find_package(Taskflow CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE Taskflow::Taskflow)

find_path(TCLAP_INCLUDE_DIRS "tclap/Arg.h")
target_include_directories(cpp-2021-vulkan PRIVATE ${TCLAP_INCLUDE_DIRS})

find_path(VULKAN_HEADERS_INCLUDE_DIRS "vk_video/vulkan_video_codec_h264std.h")
target_include_directories(cpp-2021-vulkan PRIVATE ${VULKAN_HEADERS_INCLUDE_DIRS})

find_package(unofficial-vulkan-memory-allocator CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE unofficial::vulkan-memory-allocator::vulkan-memory-allocator)

find_package(zstd CONFIG REQUIRED)
target_link_libraries(cpp-2021-vulkan PRIVATE zstd::libzstd_static)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
