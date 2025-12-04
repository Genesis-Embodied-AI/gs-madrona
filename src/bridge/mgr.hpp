#pragma once
#ifdef madgs_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

namespace madGS {

struct VisualizerGPUHandles {
    madrona::render::APIBackend *renderAPI;
    madrona::render::GPUDevice *renderDev;
};

struct GSModelGeometry {
    madrona::math::Vector3 *vertices;
    uint32_t *indices;
    uint32_t *vertexOffsets;
    uint32_t *triOffsets;
    madrona::math::Vector2 *texCoords;
    int32_t *texCoordOffsets;
    uint32_t numVertices;
    uint32_t numTris;
    uint32_t numMeshes;
};

struct GSModel {
    GSModelGeometry meshGeo;
    int32_t *geomTypes;
    int32_t *geomGroups;
    int32_t *geomDataIDs;
    int32_t *geomMatIDs;
    int32_t *enabledGeomGroups;
    madrona::math::Vector3 *geomSizes;
    madrona::math::Vector4 *matRGBA;
    int32_t *matTexIDs;
    int32_t *matTexOffsets;
    uint8_t *texData;
    int64_t *texOffsets;
    int32_t *texWidths;
    int32_t *texHeights;
    int32_t *texNChans;
    uint32_t numGeoms;
    uint32_t numMats;
    uint32_t numMatTextures;
    uint32_t numTextures;
    uint32_t numCams;
    uint32_t numLights;
    uint32_t numEnabledGeomGroups;
    float *camFovy;
    float *camZNear;
    float *camZFar;
};

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t batchRenderViewWidth;
        uint32_t batchRenderViewHeight;
        bool addCamDebugGeometry = false;
        bool useRT = false;
    };

    MGR_EXPORT Manager(
        const Config &cfg,
        const GSModel &gs_model,
        madrona::Optional<VisualizerGPUHandles> viz_gpu_hdls =
            madrona::Optional<VisualizerGPUHandles>::none());
    MGR_EXPORT ~Manager();

    MGR_EXPORT void init(const madrona::math::Vector3 *geom_pos,
                         const madrona::math::Quat *geom_rot,
                         const madrona::math::Vector3 *cam_pos,
                         const madrona::math::Quat *cam_rot,
                         const int32_t *mat_ids,
                         const uint32_t *geom_rgb,
                         const madrona::math::Diag3x3 *geom_sizes,
                         const madrona::math::Vector3 *light_pos,
                         const madrona::math::Vector3 *light_dir,
                         const uint32_t *light_rgb,
                         const bool *light_isdir,
                         const bool *light_castshadow,
                         const float *light_cutoff,
                         const float *light_attenuation,
                         const float *light_intensity);
    

    MGR_EXPORT void render(const madrona::math::Vector3 *geom_pos,
                           const madrona::math::Quat *geom_rot,
                           const madrona::math::Vector3 *cam_pos,
                           const madrona::math::Quat *cam_rot,
                           const uint32_t *render_options);
     
    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    //
    MGR_EXPORT madrona::py::Tensor instancePositionsTensor() const;
    MGR_EXPORT madrona::py::Tensor instanceRotationsTensor() const;
    MGR_EXPORT madrona::py::Tensor cameraPositionsTensor() const;
    MGR_EXPORT madrona::py::Tensor cameraRotationsTensor() const;

    MGR_EXPORT madrona::py::Tensor rgbTensor() const;
    MGR_EXPORT madrona::py::Tensor depthTensor() const;
    MGR_EXPORT madrona::py::Tensor normalTensor() const;
    MGR_EXPORT madrona::py::Tensor segmentationTensor() const;

    MGR_EXPORT uint32_t numWorlds() const;
    MGR_EXPORT uint32_t numCams() const;

    MGR_EXPORT uint32_t batchViewWidth() const;
    MGR_EXPORT uint32_t batchViewHeight() const;

    MGR_EXPORT madrona::render::RenderManager & getRenderManager();

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
