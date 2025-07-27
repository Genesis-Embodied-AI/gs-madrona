#pragma once

#include "vk/memory.hpp"
#include <memory>
#include <madrona/importer.hpp>
#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "ecs_interop.hpp"
#include "render_common.hpp"

#include <madrona/render/render_mgr.hpp>

namespace madrona::render {

struct RenderContext;

enum class LatestOperation {
    None,
    RenderPrepare,
    RenderViews,
    Transition,
};

enum ComponentNames {
    RGB = 0,
    Depth = 1,
    Normal = 2,
    Segmentation = 3,
};

////////////////////////////////////////////////////////////////////////////////
// DRAW COMMAND BUFFER CREATION                                               //
////////////////////////////////////////////////////////////////////////////////
struct DrawCommandPackage {
    // Draw cmds and drawdata
    vk::LocalBuffer drawBuffer;

    // This descriptor set contains draw information
    VkDescriptorSet drawBufferSetPrepare;
    VkDescriptorSet drawBufferSetDraw;

    uint32_t drawCmdOffset;
    uint32_t drawCmdBufferSize;

    uint32_t numDrawCounts;
};

struct BatchImportedBuffers {
    render::vk::LocalBuffer views;
    render::vk::LocalBuffer viewOffsets;

    render::vk::LocalBuffer instances;
    render::vk::LocalBuffer instanceOffsets;

    render::vk::LocalBuffer lights;
    render::vk::LocalBuffer lightOffsets;

    render::vk::LocalBuffer shadowViewData;
};

struct LayeredTarget {
    // Contains a uint for triangle ID and another for instance ID
    std::vector<std::unique_ptr<vk::LocalImage>> components;
    HeapArray<VkImageView> componentsView;

    // Shadow map
    render::vk::LocalImage shadowMap;
    VkImageView shadowMapView;
    render::vk::LocalImage shadowDepth;
    VkImageView shadowDepthView;

    uint32_t numViews;

    uint32_t pixelWidth;
    uint32_t pixelHeight;

    uint32_t viewWidth;
    uint32_t viewHeight;

    uint32_t shadowTextureWidth;
    uint32_t shadowTextureHeight;

    uint32_t shadowMapSize;
    
    const VkImageView &getImageView(uint32_t component) const;
};

struct BatchFrame {
    BatchImportedBuffers buffers;

    vk::LocalBuffer skyInput;
    vk::HostBuffer skyInputStaging;

    vk::LocalBuffer renderOptionsBuffer;
    vk::HostBuffer renderOptionsStagingBuffer;

    // View, instance info, instance data
    VkDescriptorSet viewInstanceSetPrepare;
    VkDescriptorSet viewAABBSetPrepare;
    VkDescriptorSet drawViewSet;
    VkDescriptorSet viewInstanceSetLighting;
    VkDescriptorSet shadowGenSet;
    VkDescriptorSet shadowDrawSet;
    VkDescriptorSet shadowAssetSet;

    HeapArray<LayeredTarget> targets;
    uint64_t numPixels;
    std::vector<bool> allocated;
    std::vector<std::unique_ptr<vk::DedicatedBuffer>> componentOutputs;
#ifdef MADRONA_VK_CUDA_SUPPORT
    std::vector<std::unique_ptr<vk::CudaImportedBuffer>> componentOutputsCUDA;
#endif

    // Swapchain of draw packages which get used to feed to the rasterizer
    HeapArray<DrawCommandPackage> drawPackageSwapchain;

    // Descriptor set which contains all the vizBuffer outputs and
    // the lighting outputs
    VkDescriptorSet targetsSetLighting;
    VkDescriptorSet pbrSet;

    VkCommandPool prepareCmdPool;
    VkCommandBuffer prepareCmdbuf;

    VkCommandPool renderCmdPool;
    VkCommandBuffer renderCmdbuf;
    VkSemaphore prepareFinished;    // Waited for by the viewer or the batch renderer
    VkSemaphore renderFinished;     // Waited for by the viewer to render stuff to the window
    VkSemaphore layoutTransitionFinished;   // Waited for if that latest thing was a transition
    VkFence prepareFence;   // Waited for at the beginning of each renderViews call
    VkFence renderFence;

    // Keep track of which semaphore to wait on
    LatestOperation latestOp;

    VkFence &getLatestFence();
    const vk::LocalBuffer &getComponentOutputBuffer(uint32_t component) const;
#ifdef MADRONA_VK_CUDA_SUPPORT
    const void *getComponentOutputBufferCUDA(uint32_t component);
#endif
    void initComponent(
        uint32_t component, 
        const vk::Device &dev,
        vk::MemoryAllocator &alloc,
        bool allocate
    );
};

struct BatchRenderInfo {
    uint32_t numViews;
    uint32_t numInstances;
    uint32_t numWorlds;
    uint32_t numLights;
    RenderOptions renderOptions;
};


struct BatchRenderer {
    struct Impl;
    std::unique_ptr<Impl> impl;

    bool didRender;
    RenderOptions renderOptions;

    struct Config {
        bool enableBatchRenderer;

        RenderManager::Config::RenderMode renderMode;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t maxLightsPerWorld;
        uint32_t numFrames;
    };

    BatchRenderer(const Config& cfg, RenderContext &rctx);
    ~BatchRenderer();
    void importCudaData(VkCommandBuffer);

    void setRenderOptions(const render::RenderOptions &render_options);
    void prepareForRendering(BatchRenderInfo info, EngineInterop *interop);
    void renderViews(BatchRenderInfo info,
                     const DynArray<AssetData> &loaded_assets,
                     EngineInterop *interop,
                     RenderContext &rctx);

    BatchImportedBuffers &getImportedBuffers(uint32_t frame_id);

    const vk::LocalBuffer &getComponentBuffer(uint32_t frame_id, uint32_t component) const;
    // Get the semaphore that the viewer renderer has to wait on.
    // This is either going to be the semaphore from prepareForRendering,
    // or it's the one from renderViews.
    VkSemaphore getLatestWaitSemaphore();
    const void *getComponentCUDAPtr(uint32_t frame_id, uint32_t component) const;
};


}
