#pragma once

#include <madrona/mesh_bvh.hpp>
#include <madrona/cuda_utils.hpp>

namespace madrona::render {

struct MeshBVHData {
    QBVHNode *nodes;
    uint64_t numNodes;

    MeshBVH::LeafMaterial *leafMaterial;
    uint64_t numLeaves;

    MeshBVH::BVHVertex *vertices;
    uint64_t numVerts;

    MeshBVH *meshBVHs;
    uint64_t numBVHs;
};

struct MaterialData {
    // GPU buffer containing array of texture objects
    cudaTextureObject_t *textures;
    cudaArray_t *textureBuffers;
    uint32_t numTextureBuffers;
    cudaMipmappedArray_t *mipmapTextureBuffers;
    uint32_t numMipmapTextureBuffers;
    int32_t *materialTextures;
    Material *materials;
    // Host array: for each texture, the index of its array among
    // mipmapTextureBuffers, -1 for a compressed texture held without mipmaps
    int32_t *textureMipmapBufferIndices;
    uint32_t numTextures;
    // Host copies of the materials and of their texture indices, and whether
    // each texture holds an alpha below one, which a texture write updates
    // along with the flag of the materials holding it
    Material *hostMaterials;
    int32_t *hostMaterialTextures;
    uint8_t *textureHasAlpha;
    uint32_t numMaterials;
};


}
