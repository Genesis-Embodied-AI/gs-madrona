#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>
#include <nanobind/ndarray.h>

using namespace madrona;

namespace nb = nanobind;

namespace madGS {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(_gs_madrona_batch_renderer, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<VisualizerGPUHandles>(m, "VisualizerGPUHandles");

    nb::class_<Manager>(m, "MadronaBatchRenderer")
        .def("__init__", [](
            Manager *self,
            int64_t gpu_id,
            nb::ndarray<const float, nb::shape<-1, 3>, nb::device::cpu> mesh_vertices,
            nb::ndarray<const int32_t, nb::shape<-1, 3>, nb::device::cpu> mesh_faces,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> mesh_vertex_offsets,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> mesh_face_offsets,
            nb::ndarray<const float, nb::shape<-1, 2>, nb::device::cpu> mesh_texcoords,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> mesh_texcoord_offsets,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> geom_types,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> geom_groups,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> geom_data_ids,
            nb::ndarray<const float, nb::shape<-1, 3>, nb::device::cpu> geom_sizes,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> geom_mat_ids,
            nb::ndarray<const float, nb::shape<-1, 4>, nb::device::cpu> mat_rgba,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> mat_tex_ids,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> mat_tex_offsets,
            nb::ndarray<const uint8_t, nb::shape<-1>, nb::device::cpu> tex_data,
            nb::ndarray<const int64_t, nb::shape<-1>, nb::device::cpu> tex_offsets,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> tex_widths,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> tex_heights,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> tex_nchans,
            int64_t num_lights,
            int64_t num_worlds,
            int64_t batch_render_view_width,
            int64_t batch_render_view_height,
            nb::ndarray<const float, nb::shape<-1>, nb::device::cpu> cam_fovy,
            nb::ndarray<const float, nb::shape<-1>, nb::device::cpu> cam_znear,
            nb::ndarray<const float, nb::shape<-1>, nb::device::cpu> cam_zfar,
            nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> enabled_geom_groups,
            bool add_cam_debug_geo,
            bool use_rt,
            VisualizerGPUHandles *viz_gpu_hdls)
        {
            GSModelGeometry mesh_geo {
                .vertices = (math::Vector3 *)mesh_vertices.data(),
                .indices = (uint32_t *)mesh_faces.data(),
                .vertexOffsets = (uint32_t *)mesh_vertex_offsets.data(),
                .triOffsets = (uint32_t *)mesh_face_offsets.data(),
                .texCoords = (math::Vector2 *)mesh_texcoords.data(),
                .texCoordOffsets = (int32_t *)mesh_texcoord_offsets.data(),
                .numVertices = (uint32_t)mesh_vertices.shape(0),
                .numTris = (uint32_t)mesh_faces.shape(0),
                .numMeshes = (uint32_t)mesh_vertex_offsets.shape(0),
            };

            // We need to make some copies because mgr.cpp will override
            int32_t *ptr_geom_mat_ids = (int32_t *)malloc(sizeof(int32_t) * geom_mat_ids.shape(0));
            int32_t *ptr_geom_data_ids = (int32_t *)malloc(sizeof(int32_t) * geom_data_ids.shape(0));

            memcpy(ptr_geom_mat_ids, geom_mat_ids.data(), sizeof(int32_t) * geom_mat_ids.shape(0));
            memcpy(ptr_geom_data_ids, geom_data_ids.data(), sizeof(int32_t) * geom_data_ids.shape(0));

            GSModel gs_model {
                .meshGeo = mesh_geo,
                .geomTypes = (int32_t *)geom_types.data(),
                .geomGroups = (int32_t *)geom_groups.data(),
                .geomDataIDs = ptr_geom_data_ids,     // (int32_t *)geom_data_ids.data(),
                .geomMatIDs = ptr_geom_mat_ids,       // (int32_t *)geom_mat_ids.data(),
                .enabledGeomGroups = (int32_t *)enabled_geom_groups.data(),
                .geomSizes = (math::Vector3 *)geom_sizes.data(),
                .matRGBA = (math::Vector4 *)mat_rgba.data(),
                .matTexIDs = (int32_t *)mat_tex_ids.data(),
                .matTexOffsets = (int32_t *)mat_tex_offsets.data(),
                .texData = (uint8_t *) tex_data.data(),
                .texOffsets = (int64_t *)tex_offsets.data(),
                .texWidths = (int32_t *)tex_widths.data(),
                .texHeights = (int32_t *)tex_heights.data(),
                .texNChans = (int32_t *)tex_nchans.data(),
                .numGeoms = (uint32_t)geom_types.shape(0),
                .numMats = (uint32_t)mat_rgba.shape(0),
                .numMatTextures = (uint32_t)mat_tex_ids.shape(0),
                .numTextures = (uint32_t)tex_offsets.shape(0),
                .numCams = (uint32_t)cam_fovy.shape(0),
                .numLights = (uint32_t)num_lights,
                .numEnabledGeomGroups = (uint32_t)enabled_geom_groups.shape(0),
                .camFovy = (float *)cam_fovy.data(),
                .camZNear = (float *)cam_znear.data(),
                .camZFar = (float *)cam_zfar.data(),
            };

            new (self) Manager(
                Manager::Config {
                    .gpuID = (int)gpu_id,
                    .numWorlds = (uint32_t)num_worlds,
                    .batchRenderViewWidth = (uint32_t)batch_render_view_width,
                    .batchRenderViewHeight = (uint32_t)batch_render_view_height,
                    .addCamDebugGeometry = add_cam_debug_geo,
                    .useRT = use_rt,
                },
                gs_model,
                viz_gpu_hdls != nullptr ? *viz_gpu_hdls : Optional<VisualizerGPUHandles>::none()
            );

            free(ptr_geom_mat_ids);
            free(ptr_geom_data_ids);
        }, nb::arg("gpu_id"),
           nb::arg("mesh_vertices"),
           nb::arg("mesh_faces"),
           nb::arg("mesh_vertex_offsets"),
           nb::arg("mesh_face_offsets"),
           nb::arg("mesh_texcoords"),
           nb::arg("mesh_texcoord_offsets"),
           nb::arg("geom_types"),
           nb::arg("geom_groups"),
           nb::arg("geom_data_ids"),
           nb::arg("geom_sizes"),
           nb::arg("geom_mat_ids"),
           nb::arg("mat_rgba"),
           nb::arg("mat_tex_ids"),
           nb::arg("mat_tex_offsets"),
           nb::arg("tex_data"),
           nb::arg("tex_offsets"),
           nb::arg("tex_widths"),
           nb::arg("tex_heights"),
           nb::arg("tex_nchans"),
           nb::arg("num_lights"),
           nb::arg("num_worlds"),
           nb::arg("batch_render_view_width"),
           nb::arg("batch_render_view_height"),
           nb::arg("cam_fovy"),
           nb::arg("cam_znear"),
           nb::arg("cam_zfar"),
           nb::arg("enabled_geom_groups"),
           nb::arg("add_cam_debug_geo") = false,
           nb::arg("use_rt") = false,
           nb::arg("visualizer_gpu_handles") = nb::none(),
           nb::keep_alive<1, 31>())
        .def("init", [](Manager &mgr,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> geom_pos,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 4>> geom_rot,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> cam_pos,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 4>> cam_rot,
            nb::ndarray<nb::pytorch, const int32_t, nb::shape<-1, -1>> mat_ids,
            nb::ndarray<nb::pytorch, const uint32_t, nb::shape<-1, -1>> geom_rgb,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> geom_sizes,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> light_pos,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> light_dir,
            nb::ndarray<nb::pytorch, const uint32_t, nb::shape<-1, -1>> light_rgb,
            nb::ndarray<nb::pytorch, const bool, nb::shape<-1, -1>> light_isdir,
            nb::ndarray<nb::pytorch, const bool, nb::shape<-1, -1>> light_castshadow,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1>> light_cutoff,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1>> light_attenuation,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1>> light_intensity)
        {
            mgr.init(
                geom_pos.shape(0) > 0 ? reinterpret_cast<const math::Vector3 *>(geom_pos.data()) : nullptr,
                geom_rot.shape(0) > 0 ? reinterpret_cast<const math::Quat *>(geom_rot.data()) : nullptr,
                cam_pos.shape(0) > 0 ? reinterpret_cast<const math::Vector3 *>(cam_pos.data()) : nullptr,
                cam_rot.shape(0) > 0 ? reinterpret_cast<const math::Quat *>(cam_rot.data()) : nullptr,
                mat_ids.shape(0) > 0 ? reinterpret_cast<const int32_t *>(mat_ids.data()) : nullptr,
                geom_rgb.shape(0) > 0 ? reinterpret_cast<const uint32_t *>(geom_rgb.data()) : nullptr,
                geom_sizes.shape(0) > 0 ? reinterpret_cast<const math::Diag3x3 *>(geom_sizes.data()) : nullptr,
                light_pos.shape(0) > 0 ? reinterpret_cast<const math::Vector3 *>(light_pos.data()) : nullptr,
                light_dir.shape(0) > 0 ? reinterpret_cast<const math::Vector3 *>(light_dir.data()) : nullptr,
                light_rgb.shape(0) > 0 ? reinterpret_cast<const uint32_t *>(light_rgb.data()) : nullptr,
                light_isdir.shape(0) > 0 ? reinterpret_cast<const bool *>(light_isdir.data()) : nullptr,
                light_castshadow.shape(0) > 0 ? reinterpret_cast<const bool *>(light_castshadow.data()) : nullptr,
                light_cutoff.shape(0) > 0 ? reinterpret_cast<const float *>(light_cutoff.data()) : nullptr,
                light_attenuation.shape(0) > 0 ? reinterpret_cast<const float *>(light_attenuation.data()) : nullptr,
                light_intensity.shape(0) > 0 ? reinterpret_cast<const float *>(light_intensity.data()) : nullptr
            );
        })
        .def("render", [](Manager &mgr,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> geom_pos,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 4>> geom_rot,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 3>> cam_pos,
            nb::ndarray<nb::pytorch, const float, nb::shape<-1, -1, 4>> cam_rot,
            nb::ndarray<const uint32_t, nb::shape<-1>> render_options)
        {
            mgr.render(
                reinterpret_cast<const math::Vector3 *>(geom_pos.data()),
                reinterpret_cast<const math::Quat *>(geom_rot.data()),
                reinterpret_cast<const math::Vector3 *>(cam_pos.data()),
                reinterpret_cast<const math::Quat *>(cam_rot.data()),
                reinterpret_cast<const uint32_t *>(render_options.data())
            );
        })
        .def("instance_positions_tensor", &Manager::instancePositionsTensor)
        .def("instance_rotations_tensor", &Manager::instanceRotationsTensor)
        .def("camera_positions_tensor", &Manager::cameraPositionsTensor)
        .def("camera_rotations_tensor", &Manager::cameraRotationsTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("normal_tensor", &Manager::normalTensor)
        .def("segmentation_tensor", &Manager::segmentationTensor)
    ;
}

}
