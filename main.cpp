/* Inspired by
 * https://github.com/KhronosGroup/Vulkan-Hpp/tree/master/RAII_Samples
 * https://github.com/KhronosGroup/Vulkan-Tools/tree/master/cube
 * https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/extensions/raytracing_basic
 * https://github.com/glfw/glfw/blob/master/tests/triangle-vulkan.c
 * https://github.com/charles-lunarg/vk-bootstrap/tree/master/example
 * https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp
 * https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR
 */

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
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
#include <vector>

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
  std::unique_ptr<vk::raii::Context> context;
  std::unique_ptr<vk::raii::Instance> instance;
  std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
  std::vector<uint32_t> queueFamilyIndices;
  std::unique_ptr<vk::raii::Device> device;
  std::unique_ptr<vk::raii::Queue> queue;
  std::unique_ptr<vk::raii::CommandPool> commandPool;
  void commandPoolSubmit(const std::function<void(const vk::raii::CommandBuffer& commandBuffer)> encoder) {
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo(**commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto commandBuffer = std::move(vk::raii::CommandBuffers(*device, commandBufferAllocateInfo).front());
    commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    encoder(commandBuffer);
    commandBuffer.end();
    vk::SubmitInfo submitInfo(nullptr, nullptr, *commandBuffer);
    queue->submit(submitInfo, nullptr);
    queue->waitIdle();
  }
  VmaAllocator allocator;
  void allocatorCreate() {
    VmaVulkanFunctions allocatorVulkanFunctions{};
#define VMA_VULKAN_FUNCTIONS_RAII_INSTANCE(functionName) allocatorVulkanFunctions.functionName = instance->getDispatcher()->functionName
#define VMA_VULKAN_FUNCTIONS_RAII_DEVICE(functionName) allocatorVulkanFunctions.functionName = device->getDispatcher()->functionName;
#define VMA_VULKAN_KHR_FUNCTIONS_RAII_INSTANCE(functionName)                              \
  if (instance->getDispatcher()->functionName##KHR == nullptr)                            \
    allocatorVulkanFunctions.functionName##KHR = instance->getDispatcher()->functionName; \
  else                                                                                    \
    allocatorVulkanFunctions.functionName##KHR = instance->getDispatcher()->functionName##KHR;
#define VMA_VULKAN_KHR_FUNCTIONS_RAII_DEVICE(functionName)                              \
  if (device->getDispatcher()->functionName##KHR == nullptr)                            \
    allocatorVulkanFunctions.functionName##KHR = device->getDispatcher()->functionName; \
  else                                                                                  \
    allocatorVulkanFunctions.functionName##KHR = device->getDispatcher()->functionName##KHR;
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
        .physicalDevice = static_cast<VkPhysicalDevice>(**physicalDevice),
        .device = static_cast<VkDevice>(**device),
        .pVulkanFunctions = &allocatorVulkanFunctions,
        .instance = static_cast<VkInstance>(**instance),
        .vulkanApiVersion = VK_API_VERSION_1_2,
    };
    if (vmaCreateAllocator(&allocatorCreateInfo, &allocator) != VK_SUCCESS) {
      throw std::runtime_error("vmaCreateAllocator(&allocatorCreateInfo, &allocator) != VK_SUCCESS");
    };
  }
  void allocatorDestroy() { vmaDestroyAllocator(allocator); }
  struct Buffer {
    vk::Buffer buffer;
    vk::DescriptorBufferInfo descriptor;
    VmaAllocation allocation;
    VmaAllocationInfo info;
  };
  Buffer bufferCreate(const vk::BufferCreateInfo bufferCreateInfo, const VmaAllocationCreateInfo allocationCreateInfo) {
    VkBufferCreateInfo vkBufferCreateInfo = static_cast<VkBufferCreateInfo>(bufferCreateInfo);
    VkBuffer vkBuffer;
    VmaAllocation vmaAllocation;
    VmaAllocationInfo vmaInfo;
    if (vmaCreateBuffer(allocator, &vkBufferCreateInfo, &allocationCreateInfo, &vkBuffer, &vmaAllocation, &vmaInfo) != VK_SUCCESS) {
      throw std::runtime_error(
          "vmaCreateBuffer(allocator, &vkBufferCreateInfo, &allocationCreateInfo, &vkBuffer, &vmaAllocation, &vmaInfo) != VK_SUCCESS");
    }
    return Buffer{
        .buffer = static_cast<vk::Buffer>(vkBuffer),
        .descriptor = vk::DescriptorBufferInfo(vkBuffer, 0, bufferCreateInfo.size),
        .allocation = vmaAllocation,
        .info = vmaInfo,
    };
  }
  template <typename T>
  void bufferUse(const Buffer buffer, const std::function<void(T* data)> user) {
    void* data;
    if (vmaMapMemory(allocator, buffer.allocation, &data) != VK_SUCCESS)
      throw std::runtime_error("vmaMapMemory(allocator, buffer.allocation, &data) != VK_SUCCESS");
    user(reinterpret_cast<T*>(data));
    vmaUnmapMemory(allocator, buffer.allocation);
  }
  void bufferDestroy(const Buffer buffer) { vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation); }
  Main() = delete;
  Main(const Main&) = delete;
  Main& operator=(const Main&) = delete;
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
      context = std::make_unique<vk::raii::Context>();
      if (!glfwInit()) {
        throw std::runtime_error("!glfwInit()");
      }
      if (!glfwVulkanSupported()) {
        throw std::runtime_error("!glfwVulkanSupported()");
      }
      vk::ApplicationInfo applicationInfo(applicationName.c_str(), 1, applicationName.c_str(), 1, VK_API_VERSION_1_2);
      std::vector<char const*> instanceLayers;
#if !defined(NDEBUG)
      instanceLayers.push_back("VK_LAYER_KHRONOS_synchronization2");
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
      instance = std::make_unique<vk::raii::Instance>(*context, instanceCreateInfo);
      physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(std::move(vk::raii::PhysicalDevices(*instance).front()));
      auto queueFamilyProperties = physicalDevice->getQueueFamilyProperties();
      auto queueFamilyPropertiesIterator =
          std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](vk::QueueFamilyProperties const& i) {
            return (i.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eTransfer)) ==
                   (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eTransfer);
          });
      queueFamilyIndices = {static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), queueFamilyPropertiesIterator))};
      if (queueFamilyIndices[0] >= queueFamilyProperties.size()) {
        throw std::runtime_error("queueFamilyIndices[0] >= queueFamilyProperties.size()");
      }
      auto queuePriority = 0.0f;
      vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, queueFamilyIndices[0], 1, &queuePriority);
      vk::DeviceCreateInfo deviceCreateInfo({}, deviceQueueCreateInfo);
      device = std::make_unique<vk::raii::Device>(*physicalDevice, deviceCreateInfo);
      queue = std::make_unique<vk::raii::Queue>(*device, queueFamilyIndices[0], 0);
      vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices[0]);
      commandPool = std::make_unique<vk::raii::CommandPool>(*device, commandPoolCreateInfo);
      {  // Allocator
        allocatorCreate();
        vk::BufferCreateInfo bufferCreateInfo({}, sizeof(uint32_t),
                                              vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                              vk::SharingMode::eExclusive, queueFamilyIndices);
        VmaAllocationCreateInfo allocationCreateInfo{
            .flags = {},
            .usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
        };
        auto buffer = bufferCreate(bufferCreateInfo, allocationCreateInfo);
        bufferUse<uint32_t>(buffer, [&](auto data) { data[0] = 128; });
        commandPoolSubmit([&](auto& commandBuffer) { commandBuffer.fillBuffer(buffer.buffer, 0, buffer.descriptor.range, 256); });
        bufferUse<uint32_t>(buffer, [&](auto data) { std::cout << data[0] << "\n"; });
        bufferDestroy(buffer);
        allocatorDestroy();
      }
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