/* Inspired by
 * https://github.com/KhronosGroup/Vulkan-Hpp/tree/master/samples
 * https://github.com/KhronosGroup/Vulkan-Tools/tree/master/cube
 * https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/extensions/raytracing_basic
 * https://github.com/glfw/glfw/blob/master/tests/triangle-vulkan.c
 * https://github.com/charles-lunarg/vk-bootstrap/tree/master/example
 * https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp
 * https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR
 */

#include <cassert>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#define VK_NO_PROTOTYPES
#define VULKAN_HPP_TYPESAFE_CONVERSION
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define VMA_VULKAN_VERSION 1002000
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#define IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "glm/glm.hpp"
using namespace glm;

struct Main {
  vk::DynamicLoader dynamicLoader;
  std::string applicationName = "cpp-2021-vulkan";

  Main() = delete;
  Main(int argc, char** argv) {
    try {
      if (!glfwInit()) {
        throw std::runtime_error("!glfwInit()");
      }
      if (!glfwVulkanSupported()) {
        throw std::runtime_error("!glfwVulkanSupported()");
      }
      auto vkGetInstanceProcAddr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
      VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
      vk::ApplicationInfo applicationInfo(applicationName.c_str(), 1, applicationName.c_str(), 1, VK_API_VERSION_1_2);
      std::vector<char const*> instanceLayers;
#if !defined(NDEBUG)
      instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
#endif
      for (auto& i : instanceLayers)
        std::cout << i << "\n";
      std::vector<char const*> instanceExtensions;
      {
        uint32_t glfwInstanceExtensionsCount = 0;
        auto glfwInstanceExtensions = glfwGetRequiredInstanceExtensions(&glfwInstanceExtensionsCount);
        if (glfwInstanceExtensions == NULL) {
          throw std::runtime_error("glfwInstanceExtensions == NULL");
        }
        for (auto i = 0; i < glfwInstanceExtensionsCount; ++i) {
          instanceExtensions.push_back(glfwInstanceExtensions[i]);
        }
      }
#if !defined(NDEBUG)
      instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
      for (auto& i : instanceExtensions)
        std::cout << i << "\n";
      vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo, instanceLayers, instanceExtensions);
      auto instance = vk::createInstance(instanceCreateInfo);
      VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
      auto physicalDevice = instance.enumeratePhysicalDevices().front();
      auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
      auto queueFamilyPropertiesIterator =
          std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](vk::QueueFamilyProperties const& i) {
            return (i.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics)) ==
                   (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics);
          });
      size_t queueFamilyIndex = std::distance(queueFamilyProperties.begin(), queueFamilyPropertiesIterator);
      if (queueFamilyIndex >= queueFamilyProperties.size()) {
        throw std::runtime_error("queueFamilyIndex >= queueFamilyProperties.size()");
      }
      auto queuePriority = 0.0f;
      vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, static_cast<uint32_t>(queueFamilyIndex), 1, &queuePriority);
      auto device = physicalDevice.createDevice(vk::DeviceCreateInfo({}, deviceQueueCreateInfo));
      VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
      device.destroy();
      instance.destroy();
      std::cout << "cpp-2021-vulkan\n";  // main
    } catch (vk::SystemError& err) {
      std::cout << "vk::SystemError: " << err.what() << "\n";
      exit(-1);
    } catch (std::exception& err) {
      std::cout << "std::exception: " << err.what() << "\n";
      exit(-1);
    } catch (...) {
      std::cout << "unknown error\n";
      exit(-1);
    }
  }
};

int main(int argc, char** argv) {
  auto main = Main(argc, argv);
  return 0;
}