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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
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

#include "glslang/SPIRV/GlslangToSpv.h"

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

struct Main {
  std::string applicationName{"cpp-2021-vulkan"};
  std::unique_ptr<vk::raii::Context> context;
  std::unique_ptr<vk::raii::Instance> instance;
  std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
  uint32_t queueFamilyIndex;
  std::unique_ptr<vk::raii::Device> device;
  std::unique_ptr<vk::raii::Queue> queue;
  std::unique_ptr<vk::raii::CommandPool> commandPool;
  void commandPoolSubmit(const std::function<void(const vk::raii::CommandBuffer& commandBuffer)> encoder,
                         vk::Fence waitFence = {},
                         const vk::ArrayProxyNoTemporaries<const vk::PipelineStageFlags>& waitStageMask = {},
                         const vk::ArrayProxyNoTemporaries<const vk::Semaphore>& waitSemaphores = {}) {
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo(**commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto commandBuffer = std::move(vk::raii::CommandBuffers(*device, commandBufferAllocateInfo).front());
    commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    encoder(commandBuffer);
    commandBuffer.end();
    vk::SubmitInfo submitInfo(waitSemaphores, waitStageMask, *commandBuffer);
    queue->submit(submitInfo, waitFence);
    queue->waitIdle();
  }
  std::unique_ptr<vk::raii::DescriptorPool> descriptorPool;
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
    vmaInvalidateAllocation(allocator, buffer.allocation, 0, buffer.descriptor.range);
    user(reinterpret_cast<T*>(data));
    vmaFlushAllocation(allocator, buffer.allocation, 0, buffer.descriptor.range);
    vmaUnmapMemory(allocator, buffer.allocation);
  }
  void bufferDestroy(const Buffer buffer) { vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation); }
  vk::raii::ShaderModule shaderModuleCreateFromGlslFile(vk::ShaderStageFlagBits shaderStage, std::filesystem::path shaderGlsl) {
    std::ifstream shaderModuleMainCompInput(shaderGlsl, std::ios::binary);
    if (shaderModuleMainCompInput.fail()) {
      throw std::runtime_error("shaderModuleMainCompInput.fail()");
    }
    std::stringstream shaderModuleMainCompStream;
    shaderModuleMainCompStream << shaderModuleMainCompInput.rdbuf();
    std::string shaderSource = shaderModuleMainCompStream.str();
    std::vector<unsigned int> shaderSpirv;
    EShLanguage stage;
    switch (shaderStage) {
      case vk::ShaderStageFlagBits::eVertex:
        stage = EShLangVertex;
        break;
      case vk::ShaderStageFlagBits::eTessellationControl:
        stage = EShLangTessControl;
        break;
      case vk::ShaderStageFlagBits::eTessellationEvaluation:
        stage = EShLangTessEvaluation;
        break;
      case vk::ShaderStageFlagBits::eGeometry:
        stage = EShLangGeometry;
        break;
      case vk::ShaderStageFlagBits::eFragment:
        stage = EShLangFragment;
        break;
      case vk::ShaderStageFlagBits::eCompute:
        stage = EShLangCompute;
        break;
      case vk::ShaderStageFlagBits::eRaygenKHR:
        stage = EShLangRayGen;
        break;
      case vk::ShaderStageFlagBits::eAnyHitKHR:
        stage = EShLangAnyHit;
        break;
      case vk::ShaderStageFlagBits::eClosestHitKHR:
        stage = EShLangClosestHit;
        break;
      case vk::ShaderStageFlagBits::eMissKHR:
        stage = EShLangMiss;
        break;
      case vk::ShaderStageFlagBits::eIntersectionKHR:
        stage = EShLangIntersect;
        break;
      case vk::ShaderStageFlagBits::eCallableKHR:
        stage = EShLangCallable;
        break;
      default:
        throw std::runtime_error("shaderStage");
    }
    const char* shaderStrings[1]{shaderSource.data()};
    glslang::TShader shader(stage);
    shader.setStrings(shaderStrings, 1);
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
    TBuiltInResource buildInResources{.maxLights = 32,
                                      .maxClipPlanes = 6,
                                      .maxTextureUnits = 32,
                                      .maxTextureCoords = 32,
                                      .maxVertexAttribs = 64,
                                      .maxVertexUniformComponents = 4096,
                                      .maxVaryingFloats = 64,
                                      .maxVertexTextureImageUnits = 32,
                                      .maxCombinedTextureImageUnits = 80,
                                      .maxTextureImageUnits = 32,
                                      .maxFragmentUniformComponents = 4096,
                                      .maxDrawBuffers = 32,
                                      .maxVertexUniformVectors = 128,
                                      .maxVaryingVectors = 8,
                                      .maxFragmentUniformVectors = 16,
                                      .maxVertexOutputVectors = 16,
                                      .maxFragmentInputVectors = 15,
                                      .minProgramTexelOffset = -8,
                                      .maxProgramTexelOffset = 7,
                                      .maxClipDistances = 8,
                                      .maxComputeWorkGroupCountX = 65535,
                                      .maxComputeWorkGroupCountY = 65535,
                                      .maxComputeWorkGroupCountZ = 65535,
                                      .maxComputeWorkGroupSizeX = 1024,
                                      .maxComputeWorkGroupSizeY = 1024,
                                      .maxComputeWorkGroupSizeZ = 64,
                                      .maxComputeUniformComponents = 1024,
                                      .maxComputeTextureImageUnits = 16,
                                      .maxComputeImageUniforms = 8,
                                      .maxComputeAtomicCounters = 8,
                                      .maxComputeAtomicCounterBuffers = 1,
                                      .maxVaryingComponents = 60,
                                      .maxVertexOutputComponents = 64,
                                      .maxGeometryInputComponents = 64,
                                      .maxGeometryOutputComponents = 128,
                                      .maxFragmentInputComponents = 128,
                                      .maxImageUnits = 8,
                                      .maxCombinedImageUnitsAndFragmentOutputs = 8,
                                      .maxCombinedShaderOutputResources = 8,
                                      .maxImageSamples = 0,
                                      .maxVertexImageUniforms = 0,
                                      .maxTessControlImageUniforms = 0,
                                      .maxTessEvaluationImageUniforms = 0,
                                      .maxGeometryImageUniforms = 0,
                                      .maxFragmentImageUniforms = 8,
                                      .maxCombinedImageUniforms = 8,
                                      .maxGeometryTextureImageUnits = 16,
                                      .maxGeometryOutputVertices = 256,
                                      .maxGeometryTotalOutputComponents = 1024,
                                      .maxGeometryUniformComponents = 1024,
                                      .maxGeometryVaryingComponents = 64,
                                      .maxTessControlInputComponents = 128,
                                      .maxTessControlOutputComponents = 128,
                                      .maxTessControlTextureImageUnits = 16,
                                      .maxTessControlUniformComponents = 1024,
                                      .maxTessControlTotalOutputComponents = 4096,
                                      .maxTessEvaluationInputComponents = 128,
                                      .maxTessEvaluationOutputComponents = 128,
                                      .maxTessEvaluationTextureImageUnits = 16,
                                      .maxTessEvaluationUniformComponents = 1024,
                                      .maxTessPatchComponents = 120,
                                      .maxPatchVertices = 32,
                                      .maxTessGenLevel = 64,
                                      .maxViewports = 16,
                                      .maxVertexAtomicCounters = 0,
                                      .maxTessControlAtomicCounters = 0,
                                      .maxTessEvaluationAtomicCounters = 0,
                                      .maxGeometryAtomicCounters = 0,
                                      .maxFragmentAtomicCounters = 8,
                                      .maxCombinedAtomicCounters = 8,
                                      .maxAtomicCounterBindings = 1,
                                      .maxVertexAtomicCounterBuffers = 0,
                                      .maxTessControlAtomicCounterBuffers = 0,
                                      .maxTessEvaluationAtomicCounterBuffers = 0,
                                      .maxGeometryAtomicCounterBuffers = 0,
                                      .maxFragmentAtomicCounterBuffers = 1,
                                      .maxCombinedAtomicCounterBuffers = 1,
                                      .maxAtomicCounterBufferSize = 16384,
                                      .maxTransformFeedbackBuffers = 4,
                                      .maxTransformFeedbackInterleavedComponents = 64,
                                      .maxCullDistances = 8,
                                      .maxCombinedClipAndCullDistances = 8,
                                      .maxSamples = 4,
                                      .maxMeshOutputVerticesNV = 256,
                                      .maxMeshOutputPrimitivesNV = 512,
                                      .maxMeshWorkGroupSizeX_NV = 32,
                                      .maxMeshWorkGroupSizeY_NV = 1,
                                      .maxMeshWorkGroupSizeZ_NV = 1,
                                      .maxTaskWorkGroupSizeX_NV = 32,
                                      .maxTaskWorkGroupSizeY_NV = 1,
                                      .maxTaskWorkGroupSizeZ_NV = 1,
                                      .maxMeshViewCountNV = 4,
                                      .maxDualSourceDrawBuffersEXT = 1,
                                      .limits = {
                                          .nonInductiveForLoops = 1,
                                          .whileLoops = 1,
                                          .doWhileLoops = 1,
                                          .generalUniformIndexing = 1,
                                          .generalAttributeMatrixVectorIndexing = 1,
                                          .generalVaryingIndexing = 1,
                                          .generalSamplerIndexing = 1,
                                          .generalVariableIndexing = 1,
                                          .generalConstantMatrixVectorIndexing = 1,
                                      }};
    if (!shader.parse(&buildInResources, 100, false, messages)) {
      throw std::runtime_error(std::string("!shader.parse(&buildInResources, 100, false, messages): getInfoLog:\n") +
                               std::string(shader.getInfoLog()) + std::string("\ngetInfoDebugLog:\n") + std::string(shader.getInfoDebugLog()));
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
      throw std::runtime_error(std::string("!program.link(messages): getInfoLog:\n") + std::string(shader.getInfoLog()) +
                               std::string("\ngetInfoDebugLog:\n") + std::string(shader.getInfoDebugLog()));
    }
    glslang::GlslangToSpv(*program.getIntermediate(stage), shaderSpirv);
    return vk::raii::ShaderModule(*device, vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shaderSpirv));
  }
  vk::raii::ShaderModule shaderModuleCreateFromSpirvFile(std::filesystem::path shaderSpirvFile) {
    std::ifstream shaderModuleMainCompInput(shaderSpirvFile, std::ios::ate | std::ios::binary);
    if (shaderModuleMainCompInput.fail()) {
      throw std::runtime_error("shaderModuleMainCompInput.fail()");
    }
    size_t shaderModuleMainCompInputSize = (size_t)shaderModuleMainCompInput.tellg();
    std::vector<char> shaderModuleMainCompSpirv(shaderModuleMainCompInputSize);
    shaderModuleMainCompInput.seekg(0);
    shaderModuleMainCompInput.read(shaderModuleMainCompSpirv.data(), static_cast<std::streamsize>(shaderModuleMainCompInputSize));
    return vk::raii::ShaderModule(*device, vk::ShaderModuleCreateInfo({}, shaderModuleMainCompInputSize,
                                                                      reinterpret_cast<const uint32_t*>(shaderModuleMainCompSpirv.data())));
  }
  Main() = delete;
  Main(const Main&) = delete;
  Main& operator=(const Main&) = delete;
  Main(int argc, char** argv);
};