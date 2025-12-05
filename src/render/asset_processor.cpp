#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <madrona/mesh_bvh.hpp>
#include <madrona/render/asset_processor.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/mesh_bvh_builder.hpp>

#include <filesystem>

#include <stb_image.h>

#include <span>
#include <array>
#include <algorithm>
#include <cstring>

using bytes = std::span<const std::byte>;

template <>
struct std::hash<bytes>
{
    std::size_t operator()(const bytes& x) const noexcept
    {
        return std::hash<std::string_view>{}({reinterpret_cast<const char*>(x.data()), x.size()});
    }
};

using namespace madrona::imp;

namespace madrona::render {

namespace AssetProcessor {

#ifdef MADRONA_CUDA_SUPPORT
static bool loadCache(const char *location,
                      HeapArray<MeshBVH> &bvhs_out)
{
    FILE *ptr;
    ptr = fopen(location, "rb");

    if(ptr == nullptr){
        return false;
    }

    const CountT num_bvhs = bvhs_out.size();
    for (CountT i = 0; i < num_bvhs; i++) {
        uint32_t num_verts;
        uint32_t num_nodes;
        uint32_t num_leaves;
        math::AABB aabb_out;
        int32_t material_idx;
        fread(&num_verts, sizeof(num_verts), 1, ptr);
        fread(&num_nodes, sizeof(num_nodes), 1, ptr);
        fread(&num_leaves, sizeof(num_leaves), 1, ptr);
        fread(&aabb_out, sizeof(aabb_out), 1, ptr);
        fread(&material_idx, sizeof(material_idx), 1, ptr);

        DynArray<QBVHNode> nodes{num_nodes};
        fread(nodes.data(), sizeof(QBVHNode), num_nodes, ptr);

#if 0
        DynArray<MeshBVH::LeafGeometry> leaf_geos{num_leaves};
        fread(leaf_geos.data(), sizeof(MeshBVH::LeafGeometry), num_leaves, ptr);
#endif

        DynArray<MeshBVH::BVHVertex> vertices{num_verts};
        fread(vertices.data(), sizeof(MeshBVH::BVHVertex), num_verts, ptr);



#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        DynArray<MeshBVH::LeafMaterial> leaf_materials{num_verts/3};
        fread(leaf_materials.data(), sizeof(MeshBVH::LeafMaterial), num_verts/3, ptr);
#endif

        MeshBVH bvh;
        bvh.numNodes = num_nodes;
        bvh.numLeaves = num_leaves;
        bvh.numVerts = num_verts;

        bvh.nodes = nodes.retrieve_ptr();
        // bvh.leafGeos = leaf_geos.release(true);
        bvh.leafMats = leaf_materials.retrieve_ptr();
        bvh.vertices = vertices.retrieve_ptr();
        bvh.rootAABB = aabb_out;
        bvh.materialIDX = material_idx;
        bvhs_out.emplace(i, bvh);
    }

    fclose(ptr);
    return true;
}

static void writeCache(const char *location, HeapArray<MeshBVH> &bvhs)
{
    FILE *ptr;
    ptr = fopen(location, "wb");

    for (CountT i = 0; i < bvhs.size(); i++) {
        fwrite(&bvhs[i].numVerts, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].numNodes, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].numLeaves, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].rootAABB, sizeof(math::AABB), 1, ptr);
        fwrite(&bvhs[i].materialIDX, sizeof(int32_t), 1, ptr);

        fwrite(bvhs[i].nodes, sizeof(QBVHNode), bvhs[i].numNodes, ptr);
        // fwrite(bvhs[i].leafGeos, sizeof(MeshBVH::LeafGeometry), bvhs[i].numLeaves, ptr);
        fwrite(bvhs[i].vertices, sizeof(MeshBVH::BVHVertex), bvhs[i].numVerts, ptr);

#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        fwrite(bvhs[i].leafMats, sizeof(MeshBVH::LeafMaterial), bvhs[i].numVerts/3, ptr);
#endif
    }

    fclose(ptr);
}

static HeapArray<MeshBVH> createMeshBVHs(
    Span<const SourceObject> objs)
{
    char *bvh_cache_path = getenv("MADRONA_BVH_CACHE");

    bool regen_cache = false;
    {
       char *regen_env = getenv("MADRONA_REGEN_BVH_CACHE");
       if (regen_env) {
           regen_cache = atoi(regen_env);
       }
    }

    HeapArray<MeshBVH> mesh_bvhs(objs.size());

    if (bvh_cache_path && !regen_cache) {
        bool valid_cache = loadCache(bvh_cache_path, mesh_bvhs);

        if (valid_cache) {
            return mesh_bvhs;
        }
    }

    for (CountT obj_idx = 0; obj_idx < objs.size(); obj_idx++) {
        const SourceObject &obj = objs[obj_idx];

        MeshBVH bvh = MeshBVHBuilder::build(obj.meshes);
        mesh_bvhs[obj_idx] = bvh;
    }

     if (bvh_cache_path) {
         writeCache(bvh_cache_path, mesh_bvhs);
     }

    return mesh_bvhs;
}

MeshBVHData makeBVHData(Span<const imp::SourceObject> src_objs)
{
    HeapArray<MeshBVH> mesh_bvhs = createMeshBVHs(src_objs);

    uint64_t num_bvhs = (uint32_t)mesh_bvhs.size();
    uint64_t num_nodes = 0;
    uint64_t num_leaf_mats = 0;
    uint64_t num_vertices = 0;

    for (CountT bvh_idx = 0; bvh_idx < mesh_bvhs.size(); ++bvh_idx) {
        const MeshBVH &bvh = mesh_bvhs[bvh_idx];

        num_nodes += bvh.numNodes;
#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        num_leaf_mats += bvh.numVerts/3;
#endif
        num_vertices += bvh.numVerts;
    }

    uint64_t num_bvh_bytes = num_bvhs *
        sizeof(MeshBVH);
    uint64_t num_nodes_bytes = num_nodes *
        sizeof(QBVHNode);
    uint64_t num_leaf_mat_bytes = num_leaf_mats *
        sizeof(MeshBVH::LeafMaterial);
    uint64_t num_vertices_bytes = num_vertices *
        sizeof(MeshBVH::BVHVertex);

    // All pointers to GPU memory
    auto *bvhs = (MeshBVH *)cu::allocGPU(num_bvh_bytes);
    auto *nodes = (QBVHNode *)cu::allocGPU(num_nodes_bytes);
#if defined(MADRONA_COMPRESSED_DEINDEXED) || defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
#else
    MeshBVH::LeafGeometry *leaf_geos = (MeshBVH::LeafGeometry *)
        cu::allocGPU(num_leaf_geos_bytes);
#endif
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
    MeshBVH::LeafMaterial *leaf_mats =(MeshBVH::LeafMaterial*)cu::allocGPU(num_leaf_mat_bytes);
#else
    MeshBVH::LeafMaterial *leaf_mats = nullptr;
#endif

    MeshBVH::BVHVertex *vertices = (MeshBVH::BVHVertex *)cu::allocGPU(num_vertices_bytes);

    uint64_t node_offset = 0;
    uint64_t leaf_offset = 0;
    uint64_t vert_offset = 0;

    for (CountT bvh_idx = 0; bvh_idx < mesh_bvhs.size(); ++bvh_idx) {
        const MeshBVH &bvh = mesh_bvhs[bvh_idx];

        // Need to make sure the pointers in the BVH point to GPU memory
        MeshBVH tmp = bvh;
        tmp.nodes = nodes + node_offset;
        tmp.vertices = vertices + vert_offset;
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
        tmp.leafMats = leaf_mats + leaf_offset;
#else
        tmp.leafMats = nullptr;
#endif

        REQ_CUDA(cudaMemcpy(bvhs + bvh_idx,
            &tmp, sizeof(tmp), cudaMemcpyHostToDevice));
        REQ_CUDA(cudaMemcpy(nodes + node_offset,
            bvh.nodes, bvh.numNodes * sizeof(QBVHNode),
            cudaMemcpyHostToDevice));
#if !defined(MADRONA_COMPRESSED_DEINDEXED) && !defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
        REQ_CUDA(cudaMemcpy(leaf_geos + leaf_offset,
            bvh.leafGeos, bvh.numLeaves * sizeof(MeshBVH::LeafGeometry),
            cudaMemcpyHostToDevice));
#endif
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
        REQ_CUDA(cudaMemcpy(leaf_mats + leaf_offset,
            bvh.leafMats, (bvh.numVerts / 3) *
            sizeof(MeshBVH::LeafMaterial),
            cudaMemcpyHostToDevice));
#endif

        REQ_CUDA(cudaMemcpy(vertices + vert_offset,
                    bvh.vertices, bvh.numVerts * sizeof(MeshBVH::BVHVertex),
                    cudaMemcpyHostToDevice));

        node_offset += bvh.numNodes;
        leaf_offset += bvh.numVerts/3;
        vert_offset += bvh.numVerts;
    }

    MeshBVHData gpu_data = {
        .nodes = nodes,
        .numNodes = node_offset,
        .leafMaterial = leaf_mats,
        .numLeaves = leaf_offset,
        .vertices = vertices,
        .numVerts = vert_offset,
        .meshBVHs = bvhs,
        .numBVHs = (uint64_t)mesh_bvhs.size(),
    };

    return gpu_data;
}

// Generate mipmaps for an uncompressed texture
void generateMipmaps(const void* src_data, uint32_t width, uint32_t height, 
                    void** mip_data, uint32_t* mip_widths, uint32_t* mip_heights, 
                    uint32_t& num_mips) {
    // Calculate number of mip levels
    num_mips = 1;
    uint32_t w = width;
    uint32_t h = height;
    while (w > 1 || h > 1) {
        w = std::max(1u, w / 2);
        h = std::max(1u, h / 2);
        num_mips++;
    }

    // Use source data directly for mip level 0
    mip_data[0] = const_cast<void*>(src_data);
    mip_widths[0] = width;
    mip_heights[0] = height;

    // Generate mipmaps
    w = width;
    h = height;
    for (uint32_t i = 1; i < num_mips; i++) {
        uint32_t prev_w = mip_widths[i-1];
        uint32_t prev_h = mip_heights[i-1];
        w = std::max(1u, prev_w / 2);
        h = std::max(1u, prev_h / 2);
        
        mip_widths[i] = w;
        mip_heights[i] = h;
        mip_data[i] = malloc(w * h * 4);

        // Simple box filter for mipmap generation
        uint8_t* prev_data = static_cast<uint8_t*>(mip_data[i-1]);
        uint8_t* curr_data = static_cast<uint8_t*>(mip_data[i]);
        
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                for (uint32_t c = 0; c < 4; c++) {
                    uint32_t sum = 0;
                    uint32_t count = 0;
                    
                    // Sample 2x2 block from previous mip level
                    for (uint32_t dy = 0; dy < 2 && 2*y+dy < prev_h; dy++) {
                        for (uint32_t dx = 0; dx < 2 && 2*x+dx < prev_w; dx++) {
                            sum += prev_data[((2*y+dy)*prev_w + (2*x+dx))*4 + c];
                            count++;
                        }
                    }
                    
                    curr_data[(y*w + x)*4 + c] = sum / count;
                }
            }
        }
    }
}

MaterialData initMaterialData(
    const imp::SourceMaterial *materials,
    uint32_t num_materials,
    const imp::SourceTexture *textures,
    uint32_t num_textures)
{
    // TODO: Only generate mipmaps for RGBA textures
    // Count textures by type:
    // - BC7 (compressed): uses non-mipmapped cudaArray_t → textureBuffers
    // - RGBA (uncompressed): uses mipmapped cudaMipmappedArray_t → mipmapTextureBuffers
    uint32_t num_non_mipmap_textures = 0;
    uint32_t num_mipmap_textures = 0;
    for (uint32_t i = 0; i < num_textures; ++i) {
        if (textures[i].format == imp::SourceTextureFormat::BC7) {
            num_non_mipmap_textures++;
        } else {
            num_mipmap_textures++;
        }
    }
    
    uint32_t num_material_textures = 0;
    for (uint32_t i = 0; i < num_materials; ++i) {
        num_material_textures += materials[i].numTextures;
    }

    MaterialData cpu_mat_data = {
        .textures = (cudaTextureObject_t *) malloc(sizeof(cudaTextureObject_t) * num_textures),
        .textureBuffers = (cudaArray_t *) malloc(sizeof(cudaArray_t) * num_non_mipmap_textures),
        .numTextureBuffers = num_non_mipmap_textures,
        .mipmapTextureBuffers = (cudaMipmappedArray_t *) malloc(sizeof(cudaMipmappedArray_t) * num_mipmap_textures),
        .numMipmapTextureBuffers = num_mipmap_textures,
        .materialTextures = (int32_t *) malloc(sizeof(int32_t) * num_material_textures),
        .materials = (Material *) malloc(sizeof(Material) * num_materials)
    };

    // Separate indices for texture buffer arrays (they have different sizes)
    uint32_t non_mipmap_idx = 0;
    uint32_t mipmap_idx = 0;

    // Helper function to create texture descriptor (reduces duplication)
    auto createTextureDesc = [](bool use_mipmap, uint32_t max_mip_level = 0) {
        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.sRGB = 1;
        if (use_mipmap) {
            tex_desc.mipmapFilterMode = cudaFilterModeLinear;
            tex_desc.maxMipmapLevelClamp = max_mip_level;
        }
        return tex_desc;
    };

    for (uint32_t i = 0; i < num_textures; ++i) {
        const auto &tex = textures[i];

        if (tex.format == imp::SourceTextureFormat::BC7) {
            // BC7 compressed textures: use non-mipmapped array
            cudaChannelFormatDesc channel_desc = 
                cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>();
            cudaArray_t cuda_array;
            REQ_CUDA(cudaMallocArray(&cuda_array, &channel_desc, 
                                     tex.width, tex.height, cudaArrayDefault));

            REQ_CUDA(cudaMemcpy2DToArray(cuda_array, 0, 0, tex.data,
                                         16 * tex.width / 4,
                                         16 * tex.width / 4,
                                         tex.height / 4,
                                         cudaMemcpyHostToDevice));

            cudaResourceDesc res_desc = {};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = cuda_array;

            cudaTextureDesc tex_desc = createTextureDesc(false);
            cudaTextureObject_t tex_obj = 0;
            REQ_CUDA(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

            cpu_mat_data.textures[i] = tex_obj;
            cpu_mat_data.textureBuffers[non_mipmap_idx++] = cuda_array;
        } else {
            // TODO: Only generate mipmaps for RGBA textures
            // RGBA uncompressed textures: generate mipmaps
            const uint32_t MAX_MIPS = 16;
            void* mip_data[MAX_MIPS] = {};
            uint32_t mip_widths[MAX_MIPS];
            uint32_t mip_heights[MAX_MIPS];
            uint32_t num_mips;
            
            generateMipmaps(tex.data, tex.width, tex.height, 
                            mip_data, mip_widths, mip_heights, num_mips);

            // Create CUDA mipmapped array
            cudaMipmappedArray_t mipArray;
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
            cudaExtent extent = make_cudaExtent(tex.width, tex.height, 0);
            REQ_CUDA(cudaMallocMipmappedArray(&mipArray, &channelDesc, extent, num_mips));

            // Copy each mip level
            for (uint32_t mip = 0; mip < num_mips; mip++) {
                cudaArray_t cuda_array;
                REQ_CUDA(cudaGetMipmappedArrayLevel(&cuda_array, mipArray, mip));
                REQ_CUDA(cudaMemcpy2DToArray(cuda_array, 0, 0, mip_data[mip],
                                           mip_widths[mip] * 4,
                                           mip_widths[mip] * 4,
                                           mip_heights[mip],
                                           cudaMemcpyHostToDevice));
            }

            // Free mipmap data (skip mip level 0 since it points to source data)
            for (uint32_t mip = 1; mip < num_mips; mip++) {
                if (mip_data[mip]) {
                    free(mip_data[mip]);
                }
            }

            cudaResourceDesc res_desc = {};
            res_desc.resType = cudaResourceTypeMipmappedArray;
            res_desc.res.mipmap.mipmap = mipArray;

            cudaTextureDesc tex_desc = createTextureDesc(true, num_mips - 1);
            cudaTextureObject_t tex_obj = 0;
            REQ_CUDA(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
            cpu_mat_data.textures[i] = tex_obj;
            cpu_mat_data.mipmapTextureBuffers[mipmap_idx++] = mipArray;
        }
    }

    // Populate materials and material_textures array
    int32_t material_texture_offset = 0;
    for (uint32_t i = 0; i < num_materials; ++i) {
        const SourceMaterial &src_mat = materials[i];
        cpu_mat_data.materials[i] = Material {
            .color = src_mat.color,
            .textureOffset = material_texture_offset,
            .numTextures = src_mat.numTextures,
            .roughness = src_mat.roughness,
            .metalness = src_mat.metalness,
        };
        
        if (src_mat.textureIdx != nullptr && src_mat.numTextures > 0) {
            memcpy(&cpu_mat_data.materialTextures[material_texture_offset],
                   src_mat.textureIdx,
                   sizeof(int32_t) * src_mat.numTextures);
        }
        material_texture_offset += src_mat.numTextures;
    }

    // Copy texture objects and materials to GPU
    cudaTextureObject_t *gpu_tex_buffer;
    REQ_CUDA(cudaMalloc(&gpu_tex_buffer, sizeof(cudaTextureObject_t) * num_textures));
    REQ_CUDA(cudaMemcpy(gpu_tex_buffer, cpu_mat_data.textures, 
                sizeof(cudaTextureObject_t) * num_textures,
                cudaMemcpyHostToDevice));

    Material *gpu_mat_buffer;
    REQ_CUDA(cudaMalloc(&gpu_mat_buffer, sizeof(Material) * num_materials));
    REQ_CUDA(cudaMemcpy(gpu_mat_buffer, cpu_mat_data.materials, 
                sizeof(Material) * num_materials,
                cudaMemcpyHostToDevice));

    int32_t *gpu_mat_texs;
    REQ_CUDA(cudaMalloc(&gpu_mat_texs, sizeof(int32_t) * num_material_textures));
    REQ_CUDA(cudaMemcpy(gpu_mat_texs, cpu_mat_data.materialTextures, 
                sizeof(int32_t) * num_material_textures,
                cudaMemcpyHostToDevice));

    // Free CPU-side temporary buffers
    free(cpu_mat_data.textures);
    free(cpu_mat_data.materials);
    free(cpu_mat_data.materialTextures);
    
    // Note: textureBuffers and mipmapTextureBuffers are CUDA arrays that remain on GPU
    // They are not freed here as they're part of the returned MaterialData
    // Return GPU-side MaterialData
    MaterialData gpu_mat_data = cpu_mat_data;
    gpu_mat_data.textures = gpu_tex_buffer;
    gpu_mat_data.materials = gpu_mat_buffer;
    gpu_mat_data.materialTextures = gpu_mat_texs;

    return gpu_mat_data;
}
#endif

math::AABB *makeAABBs(Span<const imp::SourceObject> src_objs)
{
    int num_objects = (int)src_objs.size();
    math::AABB *aabbs = (math::AABB *)malloc(sizeof(math::AABB) * num_objects);

    for (int obj_idx = 0; obj_idx < num_objects; ++obj_idx) {
        auto &obj = src_objs[obj_idx];

        float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

        for (int mesh_idx = 0; mesh_idx < (int)obj.meshes.size(); ++mesh_idx) {
            auto &mesh = obj.meshes[mesh_idx];

            for (int v_i = 0; v_i < (int)mesh.numVertices; ++v_i) {
                auto &v = mesh.positions[v_i];

                min_x = std::min(min_x, v.x);
                min_y = std::min(min_y, v.y);
                min_z = std::min(min_z, v.z);

                max_x = std::max(max_x, v.x);
                max_y = std::max(max_y, v.y);
                max_z = std::max(max_z, v.z);
            }
        }

        aabbs[obj_idx] = math::AABB {
            { min_x, min_y, min_z },
            { max_x, max_y, max_z }
        };
    }

    return aabbs;
}

}

}
