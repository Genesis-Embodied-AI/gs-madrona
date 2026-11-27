#include "mgr.hpp"
#include "sim.hpp"
#include "geometry.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/tracing.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::py;
using namespace madrona::imp;

namespace madGS {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    if (viz_gpu_hdls.has_value()) {
        return Optional<RenderGPUState>::none();
    }

#ifdef MGR_DISABLE_VULKAN
    return Optional<RenderGPUState>::none();
#endif

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const GSModel &gs_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls,
    const Optional<RenderGPUState> &render_gpu_state)
{
    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        assert(viz_gpu_hdls.has_value());

        render_api = viz_gpu_hdls->renderAPI;
        render_dev = viz_gpu_hdls->renderDev;
    }

    uint32_t max_instances_per_world = gs_model.numGeoms;
    if (mgr_cfg.addCamDebugGeometry) {
        max_instances_per_world += gs_model.numCams;
    }

    // Texture descriptor capacity: enough for every texture in the scene, at
    // least one (Vulkan forbids a zero-sized descriptor array), and raised to
    // the requested floor when a larger override is supplied.
    uint32_t max_textures = gs_model.numTextures;
    if (mgr_cfg.maxTextures > max_textures) {
        max_textures = mgr_cfg.maxTextures;
    }
    if (max_textures == 0) {
        max_textures = 1;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = true,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .maxTextures = max_textures,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = gs_model.numCams,
        .maxLightsPerWorld = gs_model.numLights,
        .maxInstancesPerWorld = max_instances_per_world,
        .execMode = ExecMode::CUDA,
    });
}

struct Manager::Impl {
    Config cfg;
    uint32_t numGeoms;
    uint32_t numCams;
    uint32_t numLights;

    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph renderGraph;

    Optional<MWCudaLaunchGraph> raytraceGraph;

    // The render graph also writing the materials and lights of the instances,
    // run at init and by the first render after a change of the materials
    Optional<MWCudaLaunchGraph> renderInitGraph =
        Optional<MWCudaLaunchGraph>::none();
    bool matsDirty = false;

    // The textures of the scene, which 'updateTextures' writes into those of
    // the pipeline drawing them
    uint32_t numMats = 0;
    std::vector<uint32_t> texWidths;
    std::vector<uint32_t> texHeights;
    Optional<render::MaterialData> rtMatData =
        Optional<render::MaterialData>::none();

    static inline Impl * make(
        const Config &cfg,
        const GSModel &gs_model,
        const Optional<VisualizerGPUHandles> &viz_gpu_hdls);

    inline Impl(const Manager::Config &mgr_cfg,
                uint32_t num_geoms,
                uint32_t num_cams,
                uint32_t num_lights,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                MWCudaExecutor &&gpu_exec,
                Optional<MWCudaLaunchGraph> &&raytrace_graph)

        : cfg(mgr_cfg),
          numGeoms(num_geoms),
          numCams(num_cams),
          numLights(num_lights),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          renderGraph(gpuExec.buildLaunchGraph(TaskGraphID::Render)),
          raytraceGraph(std::move(raytrace_graph))
    {}

    inline ~Impl()
    {
        if (rtMatData.has_value()) {
            free(rtMatData->textureMipmapBufferIndices);
        }
    }

    inline void renderImpl(const render::RenderOptions &render_options)
    {
        if (cfg.useRT) {
            gpuExec.run(*raytraceGraph);
        } else {
            if (renderMgr.has_value()) {
                renderMgr->batchRender(render_options);
            }
        }
    }

    inline void copyInTransforms(Vector3 *geom_positions,
                                 Quat *geom_rotations,
                                 Vector3 *cam_positions,
                                 Quat *cam_rotations,
                                 cudaStream_t strm)
    {
        uint32_t total_geoms = numGeoms * cfg.numWorlds;
        if (geom_positions != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::InstancePositions),
                geom_positions, sizeof(Vector3) * total_geoms, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (geom_rotations != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::InstanceRotations),
                geom_rotations, sizeof(Quat) * total_geoms, cudaMemcpyDeviceToDevice, strm
            );
        }

        uint32_t total_cams = numCams * cfg.numWorlds;
        if (cam_positions != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::CameraPositions),
                cam_positions, sizeof(Vector3) * total_cams, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (cam_rotations != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::CameraRotations),
                cam_rotations, sizeof(Quat) * total_cams, cudaMemcpyDeviceToDevice, strm
            );
        }
    }

    inline void copyInProperties(
        Diag3x3 *geom_sizes,
        int32_t *mat_overrides,
        uint32_t *col_overrides,
        Vector3 *light_pos,
        Vector3 *light_dir,
        uint32_t *light_color,
        bool *light_isdir,
        bool *light_castshadow,
        float *light_cutoff,
        float *light_attenuation,
        float *light_intensity,
        cudaStream_t strm)
    {
        uint32_t total_geoms = numGeoms * cfg.numWorlds;
        if (mat_overrides != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::InstanceMatOverrides),
                mat_overrides, sizeof(MaterialOverride) * total_geoms, cudaMemcpyDeviceToDevice, strm
            );
        }
        // TODO: Remove ColorOverride from ECS
        if (col_overrides != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::InstanceColorOverrides),
                col_overrides, sizeof(ColorOverride) * total_geoms, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (geom_sizes != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::InstanceScales),
                geom_sizes, sizeof(Diag3x3) * total_geoms, cudaMemcpyDeviceToDevice, strm
            );
        }

        // Copy light properties to GPU
        uint32_t total_lights = numLights * cfg.numWorlds;
        if (light_pos != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightPositions),
                light_pos, sizeof(Vector3) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_dir != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightDirections),
                light_dir, sizeof(Vector3) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_color != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightColors),
                light_color, sizeof(ColorOverride) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_isdir != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightTypes),
                light_isdir, sizeof(bool) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_castshadow != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightShadows),
                light_castshadow, sizeof(bool) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_cutoff != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightCutoffAngles),
                light_cutoff, sizeof(float) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_attenuation != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightAttenuations),
                light_attenuation, sizeof(float) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
        if (light_intensity != nullptr) {
            cudaMemcpyAsync(
                gpuExec.getExported((CountT)ExportID::LightIntensities),
                light_intensity, sizeof(float) * total_lights, cudaMemcpyDeviceToDevice, strm
            );
        }
    }

    inline void init(const Vector3 *geom_positions,
                     const Quat *geom_rotations,
                     const Vector3 *cam_positions,
                     const Quat *cam_rotations,
                     const int32_t *mat_ids,
                     const uint32_t *geom_rgb,
                     const Diag3x3 *geom_sizes,
                     const Vector3 *light_pos,
                     const Vector3 *light_dir,
                     const uint32_t *light_rgb,
                     const bool *light_isdir,
                     const bool *light_castshadow,
                     const float *light_cutoff,
                     const float *light_attenuation,
                     const float *light_intensity)
    {
        MWCudaLaunchGraph init_graph = gpuExec.buildLaunchGraph(TaskGraphID::Init);
        renderInitGraph = gpuExec.buildLaunchGraph(TaskGraphID::RenderInit);

        gpuExec.run(init_graph);

        copyInTransforms(
            const_cast<Vector3 *>(geom_positions),
            const_cast<Quat *>(geom_rotations),
            const_cast<Vector3 *>(cam_positions),
            const_cast<Quat *>(cam_rotations), 0);
        copyInProperties(
            const_cast<Diag3x3 *>(geom_sizes),
            const_cast<int32_t *>(mat_ids),
            const_cast<uint32_t *>(geom_rgb),
            const_cast<Vector3 *>(light_pos),
            const_cast<Vector3 *>(light_dir),
            const_cast<uint32_t *>(light_rgb),
            const_cast<bool *>(light_isdir),
            const_cast<bool *>(light_castshadow),
            const_cast<float *>(light_cutoff),
            const_cast<float *>(light_attenuation),
            const_cast<float *>(light_intensity), 0);

        gpuExec.run(*renderInitGraph);
        matsDirty = false;
    }

    inline void setGeomMatIDs(const int32_t *mat_ids)
    {
        if (!renderInitGraph.has_value()) {
            throw std::logic_error(
                "The materials of the geoms are set after init");
        }
        copyInProperties(nullptr, const_cast<int32_t *>(mat_ids), nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, 0);
        matsDirty = true;
    }

    inline void updateTextures(uint32_t num_textures,
                               const int32_t *texture_ids,
                               const uint8_t *pixels,
                               uint32_t width,
                               uint32_t height)
    {
        size_t num_bytes = (size_t)width * height * 4;
        for (uint32_t i = 0; i < num_textures; i++) {
            uint32_t texture_idx = (uint32_t)texture_ids[i];
            const uint8_t *rgba = pixels + i * num_bytes;
            if (cfg.useRT) {
                render::AssetProcessor::updateMaterialTexture(
                    *rtMatData, texture_idx, rgba, width, height);
            } else if (renderMgr.has_value()) {
                renderMgr->updateTexture(texture_idx, rgba, width, height);
            }
        }
    }

    inline render::RenderOptions makeRenderOptions(const uint32_t *render_options)
    {
        return render::RenderOptions{
            .outputs = {
                static_cast<bool>(render_options[0]),
                static_cast<bool>(render_options[1]),
                static_cast<bool>(render_options[2]),
                static_cast<bool>(render_options[3]),
            },
            .enableAntialiasing = static_cast<bool>(render_options[4]),
        };
    }

    inline void render(const Vector3 *geom_pos,
                             const Quat *geom_rot,
                             const Vector3 *cam_pos,
                             const Quat *cam_rot,
                             const uint32_t *render_options)
    {
        copyInTransforms(
            (Vector3 *)geom_pos,
            (Quat *)geom_rot,
            (Vector3 *)cam_pos,
            (Quat *)cam_rot, 0);

        // The first render after a change of the materials writes them into
        // the instances along with their transforms
        if (matsDirty) {
            gpuExec.runAsync(*renderInitGraph, 0);
            matsDirty = false;
        } else {
            gpuExec.runAsync(renderGraph, 0);
        }
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        // TODO: Can we remove this? where is the total number read after this?
        REQ_CUDA(cudaStreamSynchronize(0));

        renderImpl(makeRenderOptions(render_options));
    }

    inline const uint8_t * getRGBOut() const
    {
        if (cfg.useRT) {
            return (uint8_t *)gpuExec.getExported((uint32_t)ExportID::RaycastRGB);
        } else {
            return renderMgr->batchRendererRGBOut();
        }
    }

    inline const float * getDepthOut() const
    {
        if (cfg.useRT) {
            return (float *)gpuExec.getExported((uint32_t)ExportID::RaycastDepth);
        } else {
            return renderMgr->batchRendererDepthOut();
        }
    }

    inline const uint8_t * getNormalOut() const
    {
        if (cfg.useRT) {
            return (uint8_t *)gpuExec.getExported((uint32_t)ExportID::RaycastNormal);
        } else {
            return renderMgr->batchRendererNormalOut();
        }
    }

    inline const int32_t * getSegmentationOut() const
    {
        if (cfg.useRT) {
            return (int32_t *)gpuExec.getExported((uint32_t)ExportID::RaycastSegmentation);
        } else {
            return renderMgr->batchRendererSegmentationOut();
        }
    }

    inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};

struct RTAssets {
    render::MeshBVHData bvhData;
    render::MaterialData matData;
};

static RTAssets loadRenderObjects(
    const GSModel &model,
    Optional<render::RenderManager> &render_mgr,
    bool use_rt)
{
    std::array<std::string, 1> render_asset_paths;
    const char *py_root_env = getenv("MADRONA_ROOT_PATH");
    std::filesystem::path data_dir = py_root_env ? (std::string(py_root_env) + "/data") : DATA_DIR;
    render_asset_paths[(size_t)RenderPrimObjectIDs::DebugCam] = (data_dir / "debugcam.obj").string();

    std::array<const char *, render_asset_paths.size()> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    AssetImporter asset_importer;

    std::array<char, 1024> import_err;
    auto disk_render_assets = asset_importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!disk_render_assets.has_value()) {
        FATAL("Failed to load render assets from disk: %s", import_err);
    }

    // Used to store the loaded geomentry data for pointers to remain valid
    ImportedAssets generated_assets {
        .geoData = ImportedAssets::GeometryData {
            .positionArrays { 0 },
            .normalArrays { 0 },
            .tangentAndSignArrays { 0 },
            .uvArrays { 0 },
            .indexArrays { 0 },
            .faceCountArrays { 0 },
            .meshArrays { 0 },
        },
        .objects { 0 },
        .materials { 0 },
        .materialTextures { 0 },
        .instances { 0 },
        .textures { 0 },
    };

    HeapArray<SourceMesh> meshes(model.meshGeo.numMeshes + (size_t)RenderPrimObjectIDs::NumPrims);
    const CountT num_meshes = (CountT)model.meshGeo.numMeshes;

    meshes[(size_t)RenderPrimObjectIDs::DebugCam] = disk_render_assets->objects[(size_t)RenderPrimObjectIDs::DebugCam].meshes[0];
    meshes[(size_t)RenderPrimObjectIDs::Plane] = CreatePlane(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Sphere] = CreateSphere(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Box] = CreateBox(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Cylinder] = CreateCylinder(generated_assets);
    meshes[(size_t)RenderPrimObjectIDs::Capsule] = CreateCapsule(generated_assets);

    // Allocate memory for normals and tangentAndSigns
    HeapArray<Vector3> normals(model.meshGeo.numVertices);
    HeapArray<Vector4> tangentAndSigns(model.meshGeo.numVertices);
    
    for (CountT mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        uint32_t mesh_vert_offset = model.meshGeo.vertexOffsets[mesh_idx];
        uint32_t next_vert_offset = mesh_idx < num_meshes - 1 ?
            model.meshGeo.vertexOffsets[mesh_idx + 1] : model.meshGeo.numVertices;

        uint32_t mesh_tri_offset = model.meshGeo.triOffsets[mesh_idx];
        uint32_t next_tri_offset = mesh_idx < num_meshes - 1 ?
            model.meshGeo.triOffsets[mesh_idx + 1] : model.meshGeo.numTris;

        uint32_t mesh_num_verts = next_vert_offset - mesh_vert_offset;
        uint32_t mesh_num_tris = next_tri_offset - mesh_tri_offset;
        uint32_t mesh_idx_offset = mesh_tri_offset * 3;

        math::Vector2 *uvs;
        if (model.meshGeo.texCoordOffsets[mesh_idx] != -1) {
            uvs = model.meshGeo.texCoords + model.meshGeo.texCoordOffsets[mesh_idx];
        } else {
            uvs = nullptr;
        }

        meshes[mesh_idx + (size_t)RenderPrimObjectIDs::NumPrims] = {
            .positions = model.meshGeo.vertices + mesh_vert_offset,
            .normals = normals.data() + mesh_vert_offset,
            .tangentAndSigns = tangentAndSigns.data() + mesh_vert_offset,
            .uvs = uvs,
            .indices = model.meshGeo.indices + mesh_idx_offset,
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = mesh_num_verts,
            .numFaces = mesh_num_tris,
            .materialIDX = 0,
        };
    }

    // Use HeapArray with default allocator - automatically cleaned up when it goes out of scope
    HeapArray<SourceTexture> textures(model.numTextures);

    for (CountT i = 0; i < model.numTextures; i++) {
        // TODO: NChans is not used.
        uint64_t tex_offset = model.texOffsets[i];
        textures[i] = SourceTexture {
            .data = model.texData + tex_offset,
            .format = SourceTextureFormat::R8G8B8A8,
            .width = (uint32_t)model.texWidths[i],
            .height = (uint32_t)model.texHeights[i],
            .numBytes = (size_t)(model.texWidths[i] * model.texHeights[i] * 4),
        };
    }

    // Use HeapArray with default allocator - automatically cleaned up when it goes out of scope
    // This avoids dynamic reallocation and provides better performance
    HeapArray<imp::SourceMaterial> materials(model.numMats);
    for (CountT i = 0; i < model.numMats; i++) {
        const math::Vector4 &rgba = model.matRGBA[i];
        uint32_t mat_tex_offset = model.matTexOffsets[i];
        uint32_t next_tex_offset = i < model.numMats - 1 ?
            model.matTexOffsets[i + 1] : model.numMatTextures;
        uint32_t mat_tex_num = next_tex_offset - mat_tex_offset;

        materials[i] = SourceMaterial {
            .color = math::Vector4{rgba.x, rgba.y, rgba.z, rgba.w},
            .textureIdx = model.matTexIDs + mat_tex_offset,
            .numTextures = mat_tex_num,
            .roughness = 0.0f,
            .metalness = 0.0f
        };
    }

    // Create materials for geoms that do not have one assigned
    // for (CountT i = 0; i < model.numGeoms; i++) {
    //     if (model.geomMatIDs[i] == -1) {
    //         const math::Vector4 &rgba_i = model.geomRGBA[i];
    //         SourceMaterial mat = {
    //             .color = math::Vector4{rgba_i.x, rgba_i.y, rgba_i.z, rgba_i.w},
    //             .textureIdx = nullptr,
    //             .numTextures = 0,
    //             .roughness = 0.8f,
    //             .metalness = 0.2f,
    //         };
    //         materials.push_back(mat);
    //         model.geomMatIDs[i] = materials.size() - 1;

    //         for (CountT j = i + 1; j < model.numGeoms; j++) {
    //             // FIX: Should probably implement == op for Vector4
    //             const math::Vector4 &rgba_j = model.geomRGBA[j];
    //             if (model.geomMatIDs[j] == -1 && 
    //                 rgba_i.x == rgba_j.x &&
    //                 rgba_i.y == rgba_j.y &&
    //                 rgba_i.z == rgba_j.z &&
    //                 rgba_i.w == rgba_j.w
    //             ) {
    //                 model.geomMatIDs[j] = materials.size() - 1;
    //             }
    //         }
    //     }
    // }

    HeapArray<SourceObject> objs(model.numGeoms + 1);

    // Create a new mesh for each geom to avoid geoms that share the same mesh
    // from pointing to the same source mesh
    HeapArray<SourceMesh> dest_meshes(model.numGeoms + 1);
    
    for (CountT i = 0; i < model.numGeoms; i++) {
        int source_mesh_idx = -1;
        switch ((GSGeomType)model.geomTypes[i]) {
        case GSGeomType::Plane: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Plane;
        } break;
        case GSGeomType::Sphere: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Sphere;
        } break;
        case GSGeomType::Capsule: {
            dest_meshes[i] = CreateCapsule(
                generated_assets,
                model.geomSizes[i].x,
                model.geomSizes[i].y * 2);
            dest_meshes[i].materialIDX = static_cast<uint32_t>(model.geomMatIDs[i]);
        } break;
        case GSGeomType::Box: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Box;
        } break;
        case GSGeomType::Cylinder: {
            source_mesh_idx = (int)RenderPrimObjectIDs::Cylinder;
        } break;
        case GSGeomType::Mesh: {
            source_mesh_idx = (int)RenderPrimObjectIDs::NumPrims + model.geomDataIDs[i];
        } break;
        case GSGeomType::Heightfield:
        case GSGeomType::Ellipsoid:
        default:
            FATAL("Unsupported geom type");
            break;
        }

        if (source_mesh_idx != -1) {
            const SourceMesh& source_mesh = meshes[source_mesh_idx];
            dest_meshes[i] = {
                .positions = source_mesh.positions,
                .normals = source_mesh.normals,
                .tangentAndSigns = source_mesh.tangentAndSigns,
                .uvs = source_mesh.uvs,
                .indices = source_mesh.indices,
                .faceCounts = source_mesh.faceCounts,
                .faceMaterials = source_mesh.faceMaterials,
                .numVertices = source_mesh.numVertices,
                .numFaces = source_mesh.numFaces,
                .materialIDX = static_cast<uint32_t>(model.geomMatIDs[i]),
            };
            assert(model.geomMatIDs[i] >= 0);
        }

        objs[i] = {
            .meshes = Span<SourceMesh>(&dest_meshes[i], 1),
        };

        model.geomDataIDs[i] = -1;
        for (CountT geom_i = 0; geom_i < model.numEnabledGeomGroups; geom_i++)
        {
            if (model.geomGroups[i] == model.enabledGeomGroups[geom_i])
            {
                model.geomDataIDs[i] = i;
                break;
            }
        }
    }
    
    objs[model.numGeoms] = disk_render_assets->objects[(int)RenderPrimObjectIDs::DebugCam];
    for (auto &mesh : objs[model.numGeoms].meshes) {
        mesh.materialIDX = 0;
    }

    if (render_mgr.has_value()) {
        // HeapArray automatically converts to Span via template constructor
        render_mgr->loadObjects(objs, materials, textures);
    }

    if (use_rt) {
        auto ret = RTAssets {
            render::AssetProcessor::makeBVHData(objs),
            render::AssetProcessor::initMaterialData(
                materials.data(),
                materials.size(),
                textures.data(),
                textures.size()
            )
        };

        return ret;
    } else {
        return {};
    }
}

Manager::Impl * Manager::Impl::make(
    const Manager::Config &mgr_cfg,
    const GSModel &gs_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    bool use_rt = mgr_cfg.useRT;

    if (use_rt) {
        MADRONA_DEBUG_LOG("Using raytracer\n");
    } else {
        MADRONA_DEBUG_LOG("Using raytracer\n");
    }

    Sim::Config sim_cfg;
    sim_cfg.numGeoms = gs_model.numGeoms;
    sim_cfg.numCams = gs_model.numCams;
    sim_cfg.numLights = gs_model.numLights;
    sim_cfg.useDebugCamEntity = mgr_cfg.addCamDebugGeometry;
    sim_cfg.useRT = use_rt;

    CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
    Optional<RenderGPUState> render_gpu_state = initRenderGPUState(mgr_cfg, viz_gpu_hdls);
    Optional<render::RenderManager> render_mgr =
        initRenderManager(mgr_cfg, gs_model, viz_gpu_hdls, render_gpu_state);

    RTAssets rt_assets = loadRenderObjects(gs_model, render_mgr, use_rt);
    if (render_mgr.has_value()) {
        sim_cfg.renderBridge = render_mgr->bridge();
    } else {
        sim_cfg.renderBridge = nullptr;
    }

    int32_t *geom_types_gpu = (int32_t *)cu::allocGPU(sizeof(int32_t) * gs_model.numGeoms);
    int32_t *geom_data_ids_gpu = (int32_t *)cu::allocGPU(sizeof(int32_t) * gs_model.numGeoms);
    Vector3 *geom_sizes_gpu = (Vector3 *)cu::allocGPU(sizeof(Vector3) * gs_model.numGeoms);
    float *cam_fovy = (float *)cu::allocGPU(sizeof(float) * gs_model.numCams);
    float *cam_zfar = (float *)cu::allocGPU(sizeof(float) * gs_model.numCams);
    float *cam_znear = (float *)cu::allocGPU(sizeof(float) * gs_model.numCams);
    uint32_t *cam_proj_type = (uint32_t *)cu::allocGPU(sizeof(uint32_t) * gs_model.numCams);

    REQ_CUDA(cudaMemcpy(geom_types_gpu, gs_model.geomTypes, sizeof(int32_t) * gs_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(geom_data_ids_gpu, gs_model.geomDataIDs, sizeof(int32_t) * gs_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(geom_sizes_gpu, gs_model.geomSizes, sizeof(Vector3) * gs_model.numGeoms, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(cam_fovy, gs_model.camFovy, sizeof(float) * gs_model.numCams, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(cam_znear, gs_model.camZNear, sizeof(float) * gs_model.numCams, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(cam_zfar, gs_model.camZFar, sizeof(float) * gs_model.numCams, cudaMemcpyHostToDevice));
    REQ_CUDA(cudaMemcpy(cam_proj_type, gs_model.camProjType, sizeof(uint32_t) * gs_model.numCams, cudaMemcpyHostToDevice));

    // Optional per-(world, geom) visibility mask for heterogeneous entities.
    int32_t *geom_env_mask_gpu = nullptr;
    if (gs_model.geomEnvMask != nullptr) {
        size_t mask_count = (size_t)mgr_cfg.numWorlds * gs_model.numGeoms;
        geom_env_mask_gpu = (int32_t *)cu::allocGPU(sizeof(int32_t) * mask_count);
        REQ_CUDA(cudaMemcpy(geom_env_mask_gpu, gs_model.geomEnvMask,
            sizeof(int32_t) * mask_count, cudaMemcpyHostToDevice));
    }

    sim_cfg.geomTypes = geom_types_gpu;
    sim_cfg.geomDataIDs = geom_data_ids_gpu;
    sim_cfg.geomSizes = geom_sizes_gpu;
    sim_cfg.geomEnvMask = geom_env_mask_gpu;
    sim_cfg.camFovy = cam_fovy;
    sim_cfg.camZNear = cam_znear;
    sim_cfg.camZFar = cam_zfar;
    sim_cfg.camProjType = cam_proj_type;

    HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);
    Optional<CudaBatchRenderConfig> render_cfg = Optional<CudaBatchRenderConfig>::none();
    if (use_rt) {
        render_cfg = {
            .renderMode = CudaBatchRenderConfig::RenderMode::RGBD,
            .geoBVHData = rt_assets.bvhData,
            .materialData = rt_assets.matData,
            .renderWidth = mgr_cfg.batchRenderViewWidth,
            .renderHeight = mgr_cfg.batchRenderViewHeight,
        };
    }

    std::vector<std::string> hideseek_srcs = { GPU_HIDESEEK_SRC_LIST };
    const char *py_root_env = getenv("MADRONA_ROOT_PATH");
    std::filesystem::path root_dir = py_root_env ? py_root_env : std::filesystem::current_path();
    std::for_each(
        hideseek_srcs.begin(), hideseek_srcs.end(),
        [&root_dir](std::string &src) {
            src = std::filesystem::weakly_canonical(root_dir / src).string();
        }
    );
    std::vector<const char*> hideseek_srcs_cstr{};
    for (std::string const & src : hideseek_srcs) {
        hideseek_srcs_cstr.push_back(src.c_str());
    }

    MWCudaExecutor gpu_exec({
        .worldInitPtr = world_inits.data(),
        .numWorldInitBytes = sizeof(Sim::WorldInit),
        .userConfigPtr = (void *)&sim_cfg,
        .numUserConfigBytes = sizeof(Sim::Config),
        .numWorldDataBytes = sizeof(Sim),
        .worldDataAlignment = alignof(Sim),
        .numWorlds = mgr_cfg.numWorlds,
        .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
        .numExportedBuffers = (uint32_t)ExportID::NumExports, 
    }, {
        hideseek_srcs_cstr,
        { GPU_HIDESEEK_COMPILE_FLAGS },
        CompileConfig::OptMode::LTO,
    }, cu_ctx, render_cfg);

    Optional<MWCudaLaunchGraph> raytrace_graph = Optional<MWCudaLaunchGraph>::none();

    if (use_rt) {
        raytrace_graph = gpu_exec.buildRenderGraph();
    }

    cu::deallocGPU(geom_types_gpu);
    cu::deallocGPU(geom_data_ids_gpu);
    cu::deallocGPU(geom_sizes_gpu);
    if (geom_env_mask_gpu != nullptr) {
        cu::deallocGPU(geom_env_mask_gpu);
    }
    cu::deallocGPU(cam_fovy);
    cu::deallocGPU(cam_znear);
    cu::deallocGPU(cam_zfar);
    cu::deallocGPU(cam_proj_type);

    Impl *impl = new Impl {
        mgr_cfg,
        gs_model.numGeoms,
        gs_model.numCams,
        gs_model.numLights,
        std::move(render_gpu_state),
        std::move(render_mgr),
        std::move(gpu_exec),
        std::move(raytrace_graph)
    };
    impl->numMats = gs_model.numMats;
    impl->texWidths.assign(gs_model.texWidths,
                           gs_model.texWidths + gs_model.numTextures);
    impl->texHeights.assign(gs_model.texHeights,
                            gs_model.texHeights + gs_model.numTextures);
    if (use_rt) {
        impl->rtMatData = rt_assets.matData;
    }
    return impl;
}

Manager::Manager(const Config &cfg,
                 const GSModel &gs_model,
                 Optional<VisualizerGPUHandles> viz_gpu_hdls)
    : impl_(nullptr),
      lastError_()
{
    try {
        impl_.reset(Impl::make(cfg, gs_model, viz_gpu_hdls));
    } catch (const std::exception &err) {
        recordFailure(err);
    }
}

Manager::~Manager() {}

void Manager::recordFailure(const std::exception &err)
{
    lastError_ = err.what();
}

const char * Manager::lastError() const
{
    return lastError_.empty() ? nullptr : lastError_.c_str();
}

void Manager::init(const math::Vector3 *geom_pos, const math::Quat *geom_rot,
                   const math::Vector3 *cam_pos, const math::Quat *cam_rot,
                   const int32_t *mat_ids, const uint32_t *geom_rgb,
                   const math::Diag3x3 *geom_sizes, const math::Vector3 *light_pos,
                   const math::Vector3 *light_dir, const uint32_t *light_rgb,
                   const bool *light_isdir, const bool *light_castshadow,
                   const float *light_cutoff, const float *light_attenuation,
                   const float *light_intensity)
{
    if (!lastError_.empty()) {
        return;
    }
    try {
        impl_->init(
            geom_pos, geom_rot, cam_pos, cam_rot, mat_ids, geom_rgb, geom_sizes,
            light_pos, light_dir, light_rgb, light_isdir, light_castshadow,
            light_cutoff, light_attenuation, light_intensity);
    } catch (const std::exception &err) {
        recordFailure(err);
    }
}

void Manager::render(const math::Vector3 *geom_pos, const math::Quat *geom_rot,
                     const math::Vector3 *cam_pos, const math::Quat *cam_rot,
                     const uint32_t *render_options)
{
    if (!lastError_.empty()) {
        return;
    }
    try {
        impl_->render(geom_pos, geom_rot, cam_pos, cam_rot, render_options);
    } catch (const std::exception &err) {
        recordFailure(err);
    }
}

void Manager::setGeomMatIDs(const int32_t *mat_ids)
{
    if (!lastError_.empty()) {
        return;
    }
    try {
        impl_->setGeomMatIDs(mat_ids);
    } catch (const std::logic_error &) {
        throw;
    } catch (const std::exception &err) {
        recordFailure(err);
    }
}

void Manager::updateTextures(uint32_t num_textures,
                             const int32_t *texture_ids,
                             const uint8_t *pixels,
                             uint32_t width,
                             uint32_t height)
{
    if (!lastError_.empty()) {
        return;
    }
    // A texture out of range or of another size is refused before any write
    for (uint32_t i = 0; i < num_textures; i++) {
        int32_t texture_idx = texture_ids[i];
        if (texture_idx < 0 ||
                (size_t)texture_idx >= impl_->texWidths.size()) {
            throw std::out_of_range(
                "Texture " + std::to_string(texture_idx) +
                " is out of range, the scene holds " +
                std::to_string(impl_->texWidths.size()) + " textures");
        }
        if (impl_->texWidths[texture_idx] != width ||
                impl_->texHeights[texture_idx] != height) {
            throw std::invalid_argument(
                "Texture " + std::to_string(texture_idx) + " is " +
                std::to_string(impl_->texWidths[texture_idx]) + "x" +
                std::to_string(impl_->texHeights[texture_idx]) +
                ", written with " + std::to_string(width) + "x" +
                std::to_string(height) + " pixels");
        }
    }
    try {
        impl_->updateTextures(num_textures, texture_ids, pixels, width, height);
    } catch (const std::exception &err) {
        recordFailure(err);
    }
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->getRGBOut();
    if(rgb_ptr == nullptr) {
        return Tensor::none();
    }
    
    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->getDepthOut();
    if(depth_ptr == nullptr) {
        return Tensor::none();
    }

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::normalTensor() const
{
    const uint8_t *normal_ptr = impl_->getNormalOut();
    if(normal_ptr == nullptr) {
        return Tensor::none();
    }

    return Tensor((void *)normal_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::segmentationTensor() const
{
    const int32_t *segmentation_ptr = impl_->getSegmentationOut();
    if(segmentation_ptr == nullptr) {
        return Tensor::none();
    }

    return Tensor((void *)segmentation_ptr, TensorElementType::Int32, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

uint32_t Manager::numWorlds() const
{
    return impl_->cfg.numWorlds;
}

uint32_t Manager::numCams() const
{
    return impl_->numCams;
}

uint32_t Manager::numGeoms() const
{
    return impl_->numGeoms;
}

uint32_t Manager::numMaterials() const
{
    return impl_->numMats;
}

uint32_t Manager::numTextures() const
{
    return (uint32_t)impl_->texWidths.size();
}

uint32_t Manager::batchViewWidth() const
{
    return impl_->cfg.batchRenderViewWidth;
}

uint32_t Manager::batchViewHeight() const
{
    return impl_->cfg.batchRenderViewHeight;
}

render::RenderManager & Manager::getRenderManager()
{
    return *(impl_->renderMgr);
}

}
