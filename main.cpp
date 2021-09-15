#include "main.hpp"

Main::Main(int argc, char** argv) {
  try {
    std::filesystem::path sourceDirectory{"./"};
    sourceDirectory = std::filesystem::absolute(sourceDirectory);
    while (!std::filesystem::exists(sourceDirectory / "main.txt")) {
      if (sourceDirectory.parent_path() == sourceDirectory)
        throw std::runtime_error("!std::filesystem::exists(sourceDirectory / \"main.txt\")");
      sourceDirectory = sourceDirectory.parent_path();
    }
    CLI::App cliApp{applicationName};
    std::string mainDoubleCompGlsl = "main.shader.double.comp";
    cliApp.add_option("-g,--main-comp-glsl", mainDoubleCompGlsl, "GLSL")->envname("main-comp-glsl");
    std::string mainDoubleCompSpv = "main.shader.double.comp.spv";
    cliApp.add_option("-s,--main-comp-spirv", mainDoubleCompSpv, "SPIRV")->envname("main-comp-spirv");
    if (auto cliExit = [&]() -> std::optional<int> {
          CLI11_PARSE(cliApp, argc, argv);
          return std::nullopt;
        }()) {
      if (cliExit == 0)
        exit(0);
      throw std::runtime_error("CLI11_PARSE(cliApp, argc, argv)");
    }
    context = std::make_unique<vk::raii::Context>();
    glslang::InitializeProcess();
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
    queueFamilyIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), queueFamilyPropertiesIterator));
    if (queueFamilyIndex >= queueFamilyProperties.size()) {
      throw std::runtime_error("queueFamilyIndex >= queueFamilyProperties.size()");
    }
    std::vector<char const*> deviceExtensions;
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    for (auto& i : deviceExtensions)
      std::cout << i << "\n";
    auto queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, queueFamilyIndex, 1, &queuePriority);
    vk::DeviceCreateInfo deviceCreateInfo({}, deviceQueueCreateInfo, {}, deviceExtensions);
    device = std::make_unique<vk::raii::Device>(*physicalDevice, deviceCreateInfo);
    queue = std::make_unique<vk::raii::Queue>(*device, queueFamilyIndex, 0);
    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex);
    commandPool = std::make_unique<vk::raii::CommandPool>(*device, commandPoolCreateInfo);
    std::vector<vk::DescriptorPoolSize> poolSizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1 << 7),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1 << 7),
    };
    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1u << 15u, poolSizes);
    descriptorPool = std::make_unique<vk::raii::DescriptorPool>(*device, descriptorPoolCreateInfo);
    {  // Allocator
      allocatorCreate();
      vk::BufferCreateInfo bufferCreateInfo({}, sizeof(uint32_t),
                                            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
                                                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer,
                                            vk::SharingMode::eExclusive, queueFamilyIndex);
      VmaAllocationCreateInfo allocationCreateInfo{
          .flags = {},
          .usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
      };
      auto buffer = bufferCreate(bufferCreateInfo, allocationCreateInfo);
      bufferUse<uint32_t>(buffer, [&](auto data) { data[0] = 128; });
      bufferUse<uint32_t>(buffer, [&](auto data) { std::cout << data[0] << "\n"; });
      std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings{
          {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
          {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
      };
      vk::raii::DescriptorSetLayout descriptorSetLayout(*device, {{}, descriptorSetLayoutBindings});
      auto descriptorSet = std::move(vk::raii::DescriptorSets(*device, {**descriptorPool, *descriptorSetLayout}).front());
      device->updateDescriptorSets(
          {{*descriptorSet, descriptorSetLayoutBindings[0].binding, 0, descriptorSetLayoutBindings[0].descriptorType, {}, buffer.descriptor},
           {*descriptorSet, descriptorSetLayoutBindings[1].binding, 0, descriptorSetLayoutBindings[1].descriptorType, {}, buffer.descriptor}},
          nullptr);
      std::unique_ptr<vk::raii::ShaderModule> shaderModuleMainDoubleComp;
      if (mainDoubleCompSpv.ends_with("spv")) {
        shaderModuleMainDoubleComp =
            std::make_unique<vk::raii::ShaderModule>(shaderModuleCreateFromSpirvFile(sourceDirectory / mainDoubleCompSpv));
      } else {
        shaderModuleMainDoubleComp = std::make_unique<vk::raii::ShaderModule>(
            shaderModuleCreateFromGlslFile(vk::ShaderStageFlagBits::eCompute, sourceDirectory / mainDoubleCompGlsl));
      }
      commandPoolSubmit(
          [&](const vk::raii::CommandBuffer& commandBuffer) { commandBuffer.fillBuffer(buffer.buffer, 0, buffer.descriptor.range, 256); });
      bufferUse<uint32_t>(buffer, [&](auto data) { std::cout << data[0] << "\n"; });
      vk::raii::PipelineLayout pipelineLayout(*device, {{}, *descriptorSetLayout});
      vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, **shaderModuleMainDoubleComp,
                                                                      "main");
      vk::ComputePipelineCreateInfo pipelineCreateInfo({}, pipelineShaderStageCreateInfo, *pipelineLayout);
      vk::raii::Pipeline pipeline(*device, nullptr, pipelineCreateInfo);
      commandPoolSubmit([&](const vk::raii::CommandBuffer& commandBuffer) {
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descriptorSet, {});
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commandBuffer.dispatch(1, 1, 1);
      });
      bufferUse<uint32_t>(buffer, [&](auto data) { std::cout << data[0] << "\n"; });
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
      auto window = glfwCreateWindow(1280, 720, "cpp-2021-vulkan", NULL, NULL);
      VkSurfaceKHR vkSurface;
      if (glfwCreateWindowSurface(**instance, window, nullptr, &vkSurface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface(**instance, window, nullptr, &vkSurface) != VK_SUCCESS");
      vk::raii::SurfaceKHR surface(*instance, vkSurface);
      if (!physicalDevice->getSurfaceSupportKHR(queueFamilyIndex, *surface))
        throw std::runtime_error("!physicalDevice->getSurfaceSupportKHR(queueFamilyIndex, *surface)");
      uint32_t width, height;
      glfwGetFramebufferSize(window, reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height));
      vk::SurfaceFormatKHR surfaceFormat = vk::SurfaceFormatKHR(vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear);
      vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR(*surface);
      surfaceCapabilities.minImageCount = std::max(surfaceCapabilities.minImageCount, 2u);
      VkExtent2D swapchainExtent;
      if (surfaceCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
        swapchainExtent.width = std::clamp(width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
        swapchainExtent.height = std::clamp(height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
      } else {
        swapchainExtent = surfaceCapabilities.currentExtent;
      }
      vk::SwapchainCreateInfoKHR swapchainCreateInfo(
          {}, *surface, surfaceCapabilities.minImageCount, surfaceFormat.format, surfaceFormat.colorSpace, swapchainExtent, 1,
          vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage,
          vk::SharingMode::eExclusive, queueFamilyIndex, vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque,
          vk::PresentModeKHR::eFifo, true, nullptr);
      vk::raii::SwapchainKHR swapchain(*device, swapchainCreateInfo);
      std::vector<VkImage> swapchainImages = swapchain.getImages();
      std::vector<vk::raii::ImageView> swapchainImageViews;
      for (auto swapchainImage : swapchainImages) {
        vk::ImageViewCreateInfo swapchainImageViewCreateInfo(
            {}, static_cast<vk::Image>(swapchainImage), vk::ImageViewType::e2D, surfaceFormat.format,
            vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        swapchainImageViews.push_back({*device, swapchainImageViewCreateInfo});
      }
      vk::AttachmentDescription attachmentDescription{{},
                                                      surfaceFormat.format,
                                                      vk::SampleCountFlagBits::e1,
                                                      vk::AttachmentLoadOp::eLoad,
                                                      vk::AttachmentStoreOp::eStore,
                                                      vk::AttachmentLoadOp::eDontCare,
                                                      vk::AttachmentStoreOp::eDontCare,
                                                      vk::ImageLayout::eUndefined,
                                                      vk::ImageLayout::ePresentSrcKHR};
      vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
      vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, colorReference);
      vk::RenderPassCreateInfo renderPassCreateInfo({}, attachmentDescription, subpass);
      vk::raii::RenderPass renderPass(*device, renderPassCreateInfo);
      std::vector<vk::raii::Framebuffer> framebuffers;
      for (auto& swapchainImageView : swapchainImageViews) {
        std::vector<vk::ImageView> swapchainImageViewsProxy{*swapchainImageView};
        vk::FramebufferCreateInfo framebufferCreateInfo({}, *renderPass, swapchainImageViewsProxy, width, height, 1);
        framebuffers.push_back(vk::raii::Framebuffer(*device, framebufferCreateInfo));
      }
      ImGui::CreateContext();
      ImGuiIO& io = ImGui::GetIO();
      io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
      io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
      ImGui::StyleColorsDark();
      ImGui_ImplGlfw_InitForVulkan(window, false);
      ImGui_ImplVulkan_LoadFunctions(
          [](const char* function_name, void* instance) { return static_cast<vk::Instance*>(instance)->getProcAddr(function_name); },
          const_cast<void*>(static_cast<const void*>(&**instance)));
      ImGui_ImplVulkan_InitInfo imguiVulkanInfo = {};
      imguiVulkanInfo.Instance = **instance;
      imguiVulkanInfo.PhysicalDevice = **physicalDevice;
      imguiVulkanInfo.Device = **device;
      imguiVulkanInfo.QueueFamily = queueFamilyIndex;
      imguiVulkanInfo.Queue = **queue;
      imguiVulkanInfo.DescriptorPool = **descriptorPool;
      imguiVulkanInfo.MinImageCount = surfaceCapabilities.minImageCount;
      imguiVulkanInfo.ImageCount = surfaceCapabilities.minImageCount;
      if (!ImGui_ImplVulkan_Init(&imguiVulkanInfo, *renderPass))
        throw std::runtime_error("!ImGui_ImplVulkan_Init(&imguiVulkanInfo, *renderPass)");
      commandPoolSubmit([&](const vk::raii::CommandBuffer& commandBuffer) { ImGui_ImplVulkan_CreateFontsTexture(*commandBuffer); });
      ImGui_ImplVulkan_DestroyFontUploadObjects();
      auto image = imageCreate(
          {{},
           vk::ImageType::e2D,
           vk::Format::eR32G32B32A32Sfloat,
           vk::Extent3D(width, height, 1),
           1,
           1,
           vk::SampleCountFlagBits::e1,
           vk::ImageTiling::eOptimal,
           vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
               vk::ImageUsageFlagBits::eTransferDst,
           vk::SharingMode::eExclusive,
           queueFamilyIndex},
          {{},
           {},
           vk::ImageViewType::e2D,
           vk::Format::eR32G32B32A32Sfloat,
           vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
           vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)},
          {.flags = {}, .usage = VMA_MEMORY_USAGE_GPU_ONLY});
      std::vector<vk::DescriptorSetLayoutBinding> imageDescriptorSetLayoutBindings{
          {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
          {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
      };
      vk::raii::DescriptorSetLayout imageDescriptorSetLayout(*device, {{}, imageDescriptorSetLayoutBindings});
      auto imageDescriptorSet = std::move(vk::raii::DescriptorSets(*device, {**descriptorPool, *imageDescriptorSetLayout}).front());
      std::vector<vk::DescriptorImageInfo> imageDescriptorImageInfo{{{}, **image.view, vk::ImageLayout::eGeneral}};
      device->updateDescriptorSets({{*imageDescriptorSet, imageDescriptorSetLayoutBindings[1].binding, 0,
                                     imageDescriptorSetLayoutBindings[1].descriptorType, imageDescriptorImageInfo}},
                                   nullptr);
      auto shaderModuleMainShaderImageComp =
          std::make_unique<vk::raii::ShaderModule>(shaderModuleCreateFromSpirvFile(sourceDirectory / "main.shader.image.comp.spv"));
      vk::PushConstantRange imagePushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(Constants)};
      vk::raii::PipelineLayout imagePipelineLayout(*device, {{}, *imageDescriptorSetLayout, imagePushConstantRange});
      vk::PipelineShaderStageCreateInfo imagePipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute,
                                                                           **shaderModuleMainShaderImageComp, "main");
      vk::ComputePipelineCreateInfo imagePipelineCreateInfo({}, imagePipelineShaderStageCreateInfo, *imagePipelineLayout);
      vk::raii::Pipeline imagePipeline(*device, nullptr, imagePipelineCreateInfo);
      Constants constants{.offset = {0, 0}, .wh = {static_cast<int>(width), static_cast<int>(height)}, .clearColor = {0.5, 0.5, 0.5, 1.0}};
      uint64_t frameCount = 0;
      double frameDuration = 0.0;
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
          ImGui::Begin("cpp-2021-vulkan", {}, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
          ImGui::SetWindowPos(ImVec2(0, 0));
          ImGui::SetWindowSize(ImVec2(256, height));
          if (frameCount > 0) {
            ImGui::Text("Average frame duration:");
            ImGui::Text("\t%f ms", frameDuration / static_cast<double>(frameCount));
          }
          ImGui::ColorPicker4("Clear Color", constants.clearColor);
          ImGui::End();
        }
        ImGui::Render();
        ImDrawData* imguiDrawData = ImGui::GetDrawData();
        vk::raii::Semaphore imageAcquiredSemaphore(*device, vk::SemaphoreCreateInfo());
        vk::Result result;
        uint32_t imageIndex;
        std::tie(result, imageIndex) = swapchain.acquireNextImage(100000000, *imageAcquiredSemaphore);
        if (result != vk::Result::eSuccess)
          throw std::runtime_error("result != vk::Result::eSuccess");
        if (imageIndex >= swapchainImages.size())
          throw std::runtime_error("imageIndex >= swapchainImages.size()");
        vk::PipelineStageFlags waitStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        vk::raii::Fence drawFence(*device, vk::FenceCreateInfo());
        std::vector<vk::DescriptorImageInfo> imageDescriptorImageInfo{{{}, *swapchainImageViews[imageIndex], vk::ImageLayout::eGeneral}};
        device->updateDescriptorSets({{*imageDescriptorSet, imageDescriptorSetLayoutBindings[0].binding, 0,
                                       imageDescriptorSetLayoutBindings[0].descriptorType, imageDescriptorImageInfo}},
                                     nullptr);
        commandPoolSubmit([&](const vk::raii::CommandBuffer& commandBuffer) {
          commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {},
                                        {{vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eNoneKHR, vk::ImageLayout::eUndefined,
                                          vk::ImageLayout::eGeneral, queueFamilyIndex, queueFamilyIndex, swapchainImages[imageIndex],
                                          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)}});
        });
        auto t0 = std::chrono::steady_clock::now();
        for (auto x = 0; x < width; x += TILE_SIZE) {
          for (auto y = 0; y < height; y += TILE_SIZE) {
            commandPoolSubmit([&](const vk::raii::CommandBuffer& commandBuffer) {
              commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
                                            {{vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eUndefined,
                                              vk::ImageLayout::eGeneral, queueFamilyIndex, queueFamilyIndex, swapchainImages[imageIndex],
                                              vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)},
                                             {vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                                              vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, queueFamilyIndex, queueFamilyIndex,
                                              image.image, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)}});
              commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *imagePipelineLayout, 0, *imageDescriptorSet, {});
              commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *imagePipeline);
              constants.offset[0] = x;
              constants.offset[1] = y;
              commandBuffer.pushConstants<Constants>(*imagePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, {constants});
              commandBuffer.dispatch(TILE_SIZE / LOCAL_SIZE + 1, TILE_SIZE / LOCAL_SIZE + 1, 1);
            });
          }
        }
        auto t1 = std::chrono::steady_clock::now();
        frameCount += 1;
        frameDuration += std::chrono::duration<double, std::milli>(t1 - t0).count();
        commandPoolSubmit(
            [&](const vk::raii::CommandBuffer& commandBuffer) {
              vk::RenderPassBeginInfo renderPassBeginInfo(*renderPass, *framebuffers[imageIndex], vk::Rect2D{{}, {width, height}});
              commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
              ImGui_ImplVulkan_RenderDrawData(imguiDrawData, *commandBuffer);
              commandBuffer.endRenderPass();
              commandBuffer.pipelineBarrier(
                  vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {},
                  {{vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eColorAttachmentRead, vk::ImageLayout::eUndefined,
                    vk::ImageLayout::ePresentSrcKHR, queueFamilyIndex, queueFamilyIndex, swapchainImages[imageIndex],
                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)}});
            },
            *drawFence, waitStageMask, *imageAcquiredSemaphore);
        while (vk::Result::eTimeout == device->waitForFences(*drawFence, VK_TRUE, 100000000))
          ;
        vk::PresentInfoKHR presentInfoKHR(nullptr, *swapchain, imageIndex);
        result = queue->presentKHR(presentInfoKHR);
      }
      device->waitIdle();
      ImGui_ImplVulkan_Shutdown();
      ImGui_ImplGlfw_Shutdown();
      ImGui::DestroyContext();
      glfwDestroyWindow(window);
      imageDestroy(image);
      bufferDestroy(buffer);
      allocatorDestroy();
    }
    glfwTerminate();
    glslang::FinalizeProcess();
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

int main(int argc, char** argv) {
  auto main = Main{argc, argv};
  return 0;
}