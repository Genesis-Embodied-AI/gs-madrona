#include <madrona/render/render_mgr.hpp>

#include "render_ctx.hpp"

namespace madrona::render {

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
}

RenderManager::RenderManager(RenderManager &&) = default;
RenderManager::~RenderManager() = default;

void RenderManager::batchRender(const Optional<OutputOptions> &output_options_opt)
{
    uint32_t cur_num_views = *rctx_->engine_interop_.bridge.totalNumViews;
    uint32_t cur_num_instances = *rctx_->engine_interop_.bridge.totalNumInstances;
    uint32_t cur_num_lights = *rctx_->engine_interop_.bridge.totalNumLights;
    OutputOptions output_options = output_options_opt.has_value() ? *output_options_opt : OutputOptions {};

    BatchRenderInfo info = {
        .numViews = cur_num_views,
        .numInstances = cur_num_instances,
        .numWorlds = rctx_->num_worlds_,
        .numLights = cur_num_lights,
        .outputOptions = output_options,
    };

    rctx_->batchRenderer->prepareForRendering(info, &rctx_->engine_interop_);
    rctx_->batchRenderer->renderViews(
        info, rctx_->loaded_assets_, &rctx_->engine_interop_, *rctx_);
}


const uint8_t * RenderManager::batchRendererRGBOut() const
{
    return rctx_->batchRenderer->getRGBCUDAPtr();
}

const uint8_t * RenderManager::batchRendererNormalOut() const
{
    return rctx_->batchRenderer->getNormalCUDAPtr();
}

const float * RenderManager::batchRendererDepthOut() const
{
    return rctx_->batchRenderer->getDepthCUDAPtr();
}

}
