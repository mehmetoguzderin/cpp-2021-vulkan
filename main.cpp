#include <iostream>

#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "glm/glm.hpp"
using namespace glm;

struct Main {
  Main() = delete;
  Main(int argc, char** argv) {
    std::cout << "cpp-2021-vulkan\n";  // main
  }
};

int main(int argc, char** argv) {
  auto main = Main(argc, argv);
  return 0;
}