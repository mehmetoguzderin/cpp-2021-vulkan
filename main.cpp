/* Inspired by
 * https://github.com/KhronosGroup/Vulkan-Hpp/tree/master/RAII_Samples
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
#include <optional>
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

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

struct Main {
  std::string applicationName{"cpp-2021-vulkan"};
  Main() = delete;
  Main(int argc, char** argv) {
    try {
      CLI::App cliApp{applicationName};
      if (auto cliExit = [&]() -> std::optional<int> {
            CLI11_PARSE(cliApp, argc, argv);
            return std::nullopt;
          }()) {
        if (cliExit == 0)
          exit(0);
        throw std::runtime_error("CLI11_PARSE(cliApp, argc, argv)");
      }
      vk::raii::Context context;
      if (!glfwInit()) {
        throw std::runtime_error("!glfwInit()");
      }
      if (!glfwVulkanSupported()) {
        throw std::runtime_error("!glfwVulkanSupported()");
      }
      vk::ApplicationInfo applicationInfo(applicationName.c_str(), 1, applicationName.c_str(), 1, VK_API_VERSION_1_2);
      std::vector<char const*> instanceLayers;
#if !defined(NDEBUG)
      instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
#endif
      for (auto& i : instanceLayers)
        std::cout << i << "\n";
      std::vector<char const*> instanceExtensions;
      {
        auto glfwInstanceExtensionsCount = 0u;
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
      vk::raii::Instance instance(context, instanceCreateInfo);
      auto physicalDevice = std::move(vk::raii::PhysicalDevices(instance).front());
      auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
      auto queueFamilyPropertiesIterator =
          std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](vk::QueueFamilyProperties const& i) {
            return (i.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics)) ==
                   (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics);
          });
      auto queueFamilyIndex = std::distance(queueFamilyProperties.begin(), queueFamilyPropertiesIterator);
      if (queueFamilyIndex >= queueFamilyProperties.size()) {
        throw std::runtime_error("queueFamilyIndex >= queueFamilyProperties.size()");
      }
      auto queuePriority = 0.0f;
      vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, static_cast<uint32_t>(queueFamilyIndex), 1, &queuePriority);
      vk::DeviceCreateInfo deviceCreateInfo({}, deviceQueueCreateInfo);
      vk::raii::Device device(physicalDevice, deviceCreateInfo);
      VmaVulkanFunctions allocatorVulkanFunctions{};
#define VMA_VULKAN_FUNCTIONS_RAII_INSTANCE(functionName) \
  allocatorVulkanFunctions.functionName = instance.getDispatcher()->functionName
#define VMA_VULKAN_FUNCTIONS_RAII_DEVICE(functionName) \
  allocatorVulkanFunctions.functionName = device.getDispatcher()->functionName;
#define VMA_VULKAN_KHR_FUNCTIONS_RAII_INSTANCE(functionName)                             \
  if (instance.getDispatcher()->functionName##KHR == nullptr)                            \
    allocatorVulkanFunctions.functionName##KHR = instance.getDispatcher()->functionName; \
  else                                                                                   \
    allocatorVulkanFunctions.functionName##KHR = instance.getDispatcher()->functionName##KHR;
#define VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE(functionName)                             \
  if (device.getDispatcher()->functionName##KHR == nullptr)                            \
    allocatorVulkanFunctions.functionName##KHR = device.getDispatcher()->functionName; \
  else                                                                                 \
    allocatorVulkanFunctions.functionName##KHR = device.getDispatcher()->functionName##KHR;
      VMA_VULKAN_FUNCTIONS_RAII_INSTANCE(vkGetPhysicalDeviceProperties);
      VMA_VULKAN_FUNCTIONS_RAII_INSTANCE(vkGetPhysicalDeviceMemoryProperties);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkAllocateMemory);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkFreeMemory);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkMapMemory);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkUnmapMemory);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkFlushMappedMemoryRanges);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkInvalidateMappedMemoryRanges);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkBindBufferMemory);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkBindImageMemory);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkGetBufferMemoryRequirements);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkGetImageMemoryRequirements);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkCreateBuffer);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkDestroyBuffer);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkCreateImage);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkDestroyImage);
      VMA_VULKAN_FUNCTIONS_RAII_DEVICE(vkCmdCopyBuffer);
      VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE(vkGetBufferMemoryRequirements2);
      VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE(vkGetImageMemoryRequirements2);
      VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE(vkBindBufferMemory2);
      VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE(vkBindImageMemory2);
      VMA_VULKAN_KHR_FUNCTIONS_RAII_INSTANCE(vkGetPhysicalDeviceMemoryProperties2);
#undef VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE
#undef VMA_VULKAN_KHR_FUNCTIONS_RAII_INSTANCE
#undef VMA_VULKAN_FUNCTIONS_RAII_DEVICE
#undef VMA_VULKAN_FUNCTIONS_RAII_INSTANCE
      VmaAllocatorCreateInfo allocatorCreateInfo{
          .flags = 0,
          .physicalDevice = static_cast<VkPhysicalDevice>(*physicalDevice),
          .device = static_cast<VkDevice>(*device),
          .pVulkanFunctions = &allocatorVulkanFunctions,
          .instance = static_cast<VkInstance>(*instance),
          .vulkanApiVersion = VK_API_VERSION_1_2,
      };
      VmaAllocator allocator;
      if (vmaCreateAllocator(&allocatorCreateInfo, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateAllocator(&allocatorCreateInfo, &allocator) != VK_SUCCESS");
      }
      {  // Allocator
      }
      vmaDestroyAllocator(allocator);
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
  auto main = Main{argc, argv};
  return 0;
}