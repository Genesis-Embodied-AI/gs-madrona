#include <madrona/render/render_mgr.hpp>

#include "render_ctx.hpp"
#include <renderdoc/renderdoc_app.h>
#include <dlfcn.h>

namespace madrona::render {


// RenderDoc API
RENDERDOC_API_1_4_0 *rdoc_api = nullptr;

static void setupRenderDoc() {
    // Try to open the library (should already be loaded if RenderDoc injected)
    void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (mod) {
        printf("RenderDoc library is loaded.\n");

        // Get pointer to the RENDERDOC_GetAPI function
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");

        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_0, (void **)&rdoc_api);
        if (ret == 1 && rdoc_api != nullptr) {
            printf("RenderDoc API initialized.\n");
        } else {
            printf("RenderDoc API version mismatch or failure.\n");
        }
    } else {
        printf("RenderDoc is not injected (dlopen returned NULL).\n");
    }
}
const render::RenderECSBridge * RenderManager::bridge() const
{
    return rctx_->engine_interop_.gpuBridge ?
        rctx_->engine_interop_.gpuBridge : &rctx_->engine_interop_.bridge;
}

CountT RenderManager::loadObjects(Span<const imp::SourceObject> objs,
                                  Span<const imp::SourceMaterial> mats,
                                  Span<const imp::SourceTexture> textures,
                                  bool override_materials)
{
    return rctx_->loadObjects(objs, mats, textures, override_materials);
}

void RenderManager::configureLighting(Span<const LightDesc> lights)
{
    rctx_->configureLighting(lights);
}

RenderManager::RenderManager(
        APIBackend *render_backend,
        GPUDevice *render_dev,
        const Config &cfg)
    : rctx_(new RenderContext(render_backend, render_dev, cfg))
{
    setupRenderDoc();
}

RenderManager::RenderManager(RenderManager &&) = default;
RenderManager::~RenderManager() = default;

void RenderManager::batchRender(const RenderOptions &render_options)
{
    uint32_t cur_num_views = *rctx_->engine_interop_.bridge.totalNumViews;
    uint32_t cur_num_instances = *rctx_->engine_interop_.bridge.totalNumInstances;
    uint32_t cur_num_lights = *rctx_->engine_interop_.bridge.totalNumLights;

    BatchRenderInfo info = {
        .numViews = cur_num_views,
        .numInstances = cur_num_instances,
        .numWorlds = rctx_->num_worlds_,
        .numLights = cur_num_lights,
    };

    if (rdoc_api) {
        printf("Starting frame capture\n");
        rdoc_api->StartFrameCapture(rctx_->dev.hdl, nullptr);
    }

    rctx_->batchRenderer->setRenderOptions(render_options);
    rctx_->batchRenderer->prepareForRendering(info, &rctx_->engine_interop_);
    rctx_->batchRenderer->renderViews(
        info, rctx_->loaded_assets_, &rctx_->engine_interop_, *rctx_);

    if (rdoc_api) {
        printf("Ending frame capture\n");
        rdoc_api->EndFrameCapture(rctx_->dev.hdl, nullptr);
        char path[1024];
        rdoc_api->GetCapture(0, path, nullptr, nullptr);
        printf("Saved capture: %s\n", path);       
    }
}


const uint8_t * RenderManager::batchRendererRGBOut() const
{
    return rctx_->batchRenderer->getRGBCUDAPtr();
}

const float * RenderManager::batchRendererDepthOut() const
{
    return rctx_->batchRenderer->getDepthCUDAPtr();
}

}
