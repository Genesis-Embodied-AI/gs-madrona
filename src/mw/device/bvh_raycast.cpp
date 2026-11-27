#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4


#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

#ifndef NDEBUG
#define LOG_RECURSE(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG_RECURSE(...)
#endif

#ifndef NDEBUG
#define LOG_INST(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG_INST(...)
#endif

#ifndef NDEBUG
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::render;

namespace smem {
// Only shared memory to be used
static constexpr size_t kBufSize = 9 * 1024;
extern __shared__ uint8_t buffer[kBufSize];
}

extern "C" __constant__ BVHParams bvhParams;

inline Vector3 calculateOutRay(PerspectiveCameraData *view_data,
                               uint32_t pixel_x, uint32_t pixel_y)
{
    Quat rot = view_data->rotation;
    Vector3 look_at = rot.inv().rotateVec({0, 1, 0});
    Vector3 forward = look_at.normalize();
    Vector3 u = rot.inv().rotateVec({1, 0, 0}).normalize();
    Vector3 v = cross(forward, u).normalize();
    float ndc_x = ((float)pixel_x + 0.5f) / (float)bvhParams.renderOutputWidth * 2.0f - 1.0f;
    float ndc_y = ((float)pixel_y + 0.5f) / (float)bvhParams.renderOutputHeight * 2.0f - 1.0f;
    Vector3 ray_dir = Vector3(0, 0, 0);
 
    if (view_data->projectionType == static_cast<uint32_t>(RenderCamera::Perspective)) {
        // const float h = tanf(theta / 2);
        const float w = 1.0f / view_data->xScale;
        const float h = 1.0f / -view_data->yScale;
        ray_dir = (ndc_x * w * u + ndc_y * h * v + forward).normalize();
    } else {
        const float aspect = -view_data->yScale / view_data->xScale;
        float sx = ndc_x, sy = ndc_y;
        if (aspect > 1.0f) sx *= aspect; else sy /= aspect;
        const float rho = sqrtf(sx * sx + sy * sy);
        const float theta = rho * view_data->halfFov;
        const float plane_scale = tanf(theta) / rho;
        ray_dir = (sx * plane_scale * u + sy * plane_scale * v + forward).normalize();
    }

    return ray_dir;
}

struct TraceResult {
    bool hit;
    Vector3 color;
    // The opacity of the surface at the hit, its material color alpha times
    // the alpha of its texture
    float opacity;
    Vector3 normal;
    int segmentation;
    float metalness;
    float roughness;
    float depth;
};

// The opacity at or below which a ray goes through a surface as if nothing
// were there, the texels of zero alpha of a cutout.
static constexpr float kTransparentOpacity = 0.5f / 255.f;

// The opacity from which a surface covers a pixel in the depth, normal and
// segmentation outputs, as an alpha cutoff (the glTF MASK default). The
// rasterizer draws a surface below it in no output and one above it opaque, so
// that both pipelines report the same surface at each pixel.
static constexpr float kCoverageOpacity = 0.5f;

// The number of translucent surfaces a camera ray crosses before the last one
// counts as opaque, and a shadow ray crosses before it stops, letting through
// what is left.
static constexpr int kMaxTranslucentLayers = 8;

struct TraceInfo {
    Vector3 rayOrigin;
    Vector3 rayDirection;
    float tMin;
    float tMax;
    bool dOnly; // Depth only
    // The distance from the camera to the origin of the ray, for a ray
    // continuing past a surface or toward a light, which selects the level of
    // detail of the textures it samples along with the distance it travels
    float lodDistance;
};

struct TraceWorldInfo {
    QBVHNode *nodes;
    InstanceData *instances;

    LightDesc *lights;
    uint32_t numLights;
};

enum class GroupType : uint8_t {
    TopLevel,
    BottomLevelRoot,
    BottomLevel,
    Triangles,
    None
};

using NodeGroup = uint64_t;

// Node group just has 8 present bits
static NodeGroup encodeNodeGroup(
        uint32_t node_idx,
        uint32_t present_bits,
        GroupType type)
{
    return (uint64_t)node_idx |
           ((uint64_t)present_bits << 32) |
           ((uint64_t)type << 61);
}

static NodeGroup invalidNodeGroup()
{
    return encodeNodeGroup(0xFFFF'FFFF, 0, GroupType::None);
}

static NodeGroup getRootGroup(TraceWorldInfo world_info)
{
    uint32_t children_count = world_info.nodes[0].numChildren;
    uint8_t present_bits = (uint8_t)((1ull << children_count) - 1);
    return encodeNodeGroup(0, present_bits, GroupType::TopLevel);
}

static GroupType getGroupType(NodeGroup grp)
{
    return (GroupType)((grp >> 61ull) & 0b111);
}

static NodeGroup unsetPresentBit(NodeGroup grp, uint32_t idx)
{
    grp &= ~(1ull << (idx + 32ull));
    return grp;
}

static uint32_t getPresentBits(NodeGroup grp)
{
    return (uint32_t)((grp >> 32) & 0xFF);
}

static uint32_t getTrianglePresentBits(NodeGroup grp)
{
    // 24 bits used for triangle presence
    return (uint32_t)((grp >> 32) & ((1ull << 24) - 1));
}

static Vector3 getDirQuant(QBVHNode node, Diag3x3 inv_ray_d)
{
    return Vector3 {
        __uint_as_float((node.expX + 127) << 23) * inv_ray_d.d0,
        __uint_as_float((node.expY + 127) << 23) * inv_ray_d.d1,
        __uint_as_float((node.expZ + 127) << 23) * inv_ray_d.d2,
    };
}

static Vector3 getOriginQuant(QBVHNode node, Vector3 ray_o, Diag3x3 inv_ray_d)
{
    return Vector3 {
        (node.minPoint.x - ray_o.x) * inv_ray_d.d0,
        (node.minPoint.y - ray_o.y) * inv_ray_d.d1,
        (node.minPoint.z - ray_o.z) * inv_ray_d.d2,            
    };
}

static std::pair<float, float> getNearFar(
        QBVHNode node, 
        uint32_t child_idx,
        Vector3 dir_quant,
        Vector3 origin_quant,
        float t_max)
{
    Vector3 t_near3 = {
        node.qMinX[child_idx] * dir_quant.x + origin_quant.x,
        node.qMinY[child_idx] * dir_quant.y + origin_quant.y,
        node.qMinZ[child_idx] * dir_quant.z + origin_quant.z,
    };

    Vector3 t_far3 = {
        node.qMaxX[child_idx] * dir_quant.x + origin_quant.x,
        node.qMaxY[child_idx] * dir_quant.y + origin_quant.y,
        node.qMaxZ[child_idx] * dir_quant.z + origin_quant.z,
    };

    float t_near = fmaxf(fminf(t_near3.x, t_far3.x), 
                         fmaxf(fminf(t_near3.y, t_far3.y),
                               fmaxf(fminf(t_near3.z, t_far3.z), 
                                     0.f)));

    float t_far = fminf(fmaxf(t_far3.x, t_near3.x), 
                        fminf(fmaxf(t_far3.y, t_near3.y),
                              fminf(fmaxf(t_far3.z, t_near3.z), 
                                    t_max)));

    return {
        t_near, t_far
    };
}

struct TriangleIsectInfo {
    // TODO: can reduce register usage by storing kx/y/z in a single int32_t
    int32_t kx, ky, kz;
    float Sx, Sy, Sz;
};

static TriangleIsectInfo computeRayIsectInfo(
        Vector3 o, Vector3 d,
        Diag3x3 inv_d)
{
    // Woop et al 2013
    float abs_x = fabsf(d.x);
    float abs_y = fabsf(d.y);
    float abs_z = fabsf(d.z);

    int32_t kz;
    if (abs_x > abs_y && abs_x > abs_z) {
        kz = 0;
    } else if (abs_y > abs_z) {
        kz = 1;
    } else {
        kz = 2;
    }

    int32_t kx = kz + 1;
    if (kx == 3) {
        kx = 0;
    }

    int32_t ky = kx + 1;
    if (ky == 3) {
        ky = 0;
    }

    // swap kx and ky dimensions to preserve winding direction of triangles
    if (d[kz] < 0.f) {
        std::swap(kx, ky);
    }

    // Calculate shear constants
    float Sx = d[kx] * inv_d[kz];
    float Sy = d[ky] * inv_d[kz];
    float Sz = inv_d[kz];

    return {
        kx, ky, kz,
        Sx, Sy, Sz
    };
}

static bool instanceHasVolume(InstanceData *instance_data)
{
    return !(instance_data->scale.d0 == 0.f &&
             instance_data->scale.d1 == 0.f &&
             instance_data->scale.d2 == 0.f);
}

struct TriHitInfo {
    bool hit;

    float tHit;

    int32_t instanceIdx;

    Vector3 normal;
    Vector2 uv;

    uint32_t leafMaterialIndex;

    MeshBVH *bvh;

    InstanceData *instance;
};

struct TriangleFetch {
    Vector3 a, b, c;
    Vector2 uva, uvb, uvc;
    Vector3 normalA, normalB, normalC;
    Vector4 tangentAndSignA, tangentAndSignB, tangentAndSignC;
};

static TriangleFetch fetchLeafTriangle(int32_t leaf_idx,
                                       int32_t offset,
                                       MeshBVH *mesh_bvh)
{
    TriangleFetch fetched = {
        .a = mesh_bvh->vertices[(leaf_idx + offset)*3 + 0].pos,
        .b = mesh_bvh->vertices[(leaf_idx + offset)*3 + 1].pos,
        .c = mesh_bvh->vertices[(leaf_idx + offset)*3 + 2].pos,
        .uva = mesh_bvh->vertices[(leaf_idx + offset)*3 + 0].uv,
        .uvb = mesh_bvh->vertices[(leaf_idx + offset)*3 + 1].uv,
        .uvc = mesh_bvh->vertices[(leaf_idx + offset)*3 + 2].uv,
        .normalA = mesh_bvh->vertices[(leaf_idx + offset)*3 + 0].normal,
        .normalB = mesh_bvh->vertices[(leaf_idx + offset)*3 + 1].normal,
        .normalC = mesh_bvh->vertices[(leaf_idx + offset)*3 + 2].normal,
        .tangentAndSignA = mesh_bvh->vertices[(leaf_idx + offset)*3 + 0].tangentAndSign,
        .tangentAndSignB = mesh_bvh->vertices[(leaf_idx + offset)*3 + 1].tangentAndSign,
        .tangentAndSignC = mesh_bvh->vertices[(leaf_idx + offset)*3 + 2].tangentAndSign,
    };

    return fetched;
}

static bool rayTriangleIntersection(
    Vector3 tri_a, Vector3 tri_b, Vector3 tri_c,
    Vector3 normalA, Vector3 normalB, Vector3 normalC,
    int32_t kx, int32_t ky, int32_t kz,
    float Sx, float Sy, float Sz,
    Vector3 org,
    float t_max,
    float *out_hit_t,
    Vector3 *bary_out,
    Vector3 *out_hit_normal)
{
    // Calculate vertices relative to ray origin
    const Vector3 A = tri_a - org;
    const Vector3 B = tri_b - org;
    const Vector3 C = tri_c - org;

    const float a_kz = A[kz];
    const float a_kx = A[kx];
    const float a_ky = A[ky];
    const float b_kz = B[kz];
    const float b_kx = B[kx];
    const float b_ky = B[ky];
    const float c_kz = C[kz];
    const float c_kx = C[kx];
    const float c_ky = C[ky];

    // Perform shear and scale of vertices
    const float Ax = fmaf(-Sx, a_kz, a_kx);
    const float Ay = fmaf(-Sy, a_kz, a_ky);
    const float Bx = fmaf(-Sx, b_kz, b_kx);
    const float By = fmaf(-Sy, b_kz, b_ky);
    const float Cx = fmaf(-Sx, c_kz, c_kx);
    const float Cy = fmaf(-Sy, c_kz, c_ky);

    // calculate scaled barycentric coordinates
    float U = fmaf(Cx, By, - Cy * Bx);
    float V = fmaf(Ax, Cy, - Ay * Cx);
    float W = fmaf(Bx, Ay, - By * Ax);

    static constexpr float eps_tol = 1e-7;

    if (U > -eps_tol && U < eps_tol)
        U = 0.f;
    if (V > -eps_tol && V < eps_tol)
        V = 0.f;
    if (W > -eps_tol && W < eps_tol)
        W = 0.f;

    // Perform edge tests
#ifdef MADRONA_MESHBVH_BACKFACE_CULLING
    if (U < 0.0f || V < 0.0f || W < 0.0f) {
        return false;
    }
#else
    if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
            (U > 0.0f || V > 0.0f || W > 0.0f)) {
        return false;
    }
#endif

    // fallback  to testing against edges using double precision
    if (U == 0.0f || V == 0.0f || W == 0.0f) [[unlikely]] {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);

        // Perform edge tests
#ifdef MADRONA_MESHBVH_BACKFACE_CULLING
        if (U < 0.0f || V < 0.0f || W < 0.0f) {
            return false;
        }
#else
        if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
                (U > 0.0f || V > 0.0f || W > 0.0f)) {
            return false;
        }
#endif
    }

    // Calculate determinant
    float det = U + V + W;
    if (det == 0.f) return false;

    // Calculate scaled z-coordinates of vertices and use them to calculate
    // the hit distance
    const float Az = Sz * a_kz;
    const float Bz = Sz * b_kz;
    const float Cz = Sz * c_kz;
    const float T = fmaf(U, Az, fmaf(V, Bz, W * Cz));

#ifdef MADRONA_MESHBVH_BACKFACE_CULLING
    if (T < 0.0f || T > t_max * det) {
        return false;
    }

#else
#ifdef MADRONA_GPU_MODE
    uint32_t det_sign_mask =
        __float_as_uint(det) & 0x8000'0000_u32;

    float xor_T = __uint_as_float(
        __float_as_uint(T) ^ det_sign_mask);
#else
    uint32_t det_sign_mask =
        std::bit_cast<uint32_t>(det) & 0x8000'0000_u32;

    float xor_T = std::bit_cast<float>(
        std::bit_cast<uint32_t>(T) ^ det_sign_mask);
#endif
    
    float xor_det = copysignf(det, 1.f);
    if (xor_T < 0.0f || xor_T > t_max * xor_det) {
        return false;
    }
#endif

    // normalize U, V, W, and T
    const float rcpDet = 1.0f / det;

    *out_hit_t = T * rcpDet;
    *bary_out = Vector3{U,V,W} * rcpDet;

    // FIXME better way to get geo normal?
    *out_hit_normal = normalize(normalA * U + normalB * V + normalC * W);

    return true;
}

static TriHitInfo triangleIntersect(int32_t leaf_idx,
                                    int32_t tri_idx,
                                    TriangleIsectInfo isect_info,
                                    Vector3 ray_o,
                                    float t_max,
                                    MeshBVH *mesh_bvh)
{
    // Woop et al 2013 Watertight Ray/Triangle Intersection
    Vector3 hit_normal = { 0, 0, 0 };
    Vector3 baryout = { 0, 0, 0 };

    float hit_t;
    TriangleFetch fetched = fetchLeafTriangle(
            leaf_idx, tri_idx, mesh_bvh);

    bool intersects = rayTriangleIntersection(
            fetched.a, fetched.b, fetched.c,
            fetched.normalA, fetched.normalB, fetched.normalC,
            isect_info.kx,  isect_info.ky, isect_info.kz, 
            isect_info.Sx,  isect_info.Sy, isect_info.Sz, 
            ray_o,
            t_max,
            &hit_t,
            &baryout,
            &hit_normal);

    if (intersects) {
        Vector2 realuv = fetched.uva * baryout.x +
                         fetched.uvb * baryout.y +
                         fetched.uvc * baryout.z;

        return {
            .hit = true,
            .tHit = hit_t,
            .normal = hit_normal,
            .uv = realuv,
            .leafMaterialIndex = (uint32_t)(leaf_idx + tri_idx),
            .bvh = mesh_bvh,
        };
    } else {
        return {
            .hit = false,
        };
    }
}



static Vector3 hexToRgb(uint32_t hex)
{
    // Extract each color component and normalize to [0, 1] range
    float r = ((hex >> 16) & 0xFF) / 255.0f;
    float g = ((hex >> 8) & 0xFF) / 255.0f;
    float b = (hex & 0xFF) / 255.0f;

    return {r, g, b};
}

struct SurfaceSample {
    Vector3 color;
    float opacity;
    int32_t materialIdx;
};

// The level of detail a texture is sampled at for a hit at some distance from
// the origin of the ray, which stands in for the footprint of the pixel.
static __device__ inline float textureLod(float dist)
{
    return fminf(fmaxf(log2f(dist + 1e-3f), 0.0f), 8.0f);
}

// The color and the opacity of the surface of an instance at a hit, its
// material color times its texture, the texture of the world for a material
// holding one per world.
static __device__ SurfaceSample sampleSurface(const InstanceData *instance,
                                              const TriHitInfo &tri_hit,
                                              float dist)
{
    SurfaceSample sample = {
        .color = { 1.f, 1.f, 1.f },
        .opacity = 1.f,
        .materialIdx = -1,
    };

    int32_t override_mat_id = instance->matID;
    if (override_mat_id == MaterialOverride::UseOverrideColor) {
        sample.color = hexToRgb(instance->color);
        return sample;
    }
    if (bvhParams.materials == nullptr) {
        return sample;
    }

    int32_t material_idx = override_mat_id;
    if (override_mat_id == MaterialOverride::UseDefaultMaterial) {
        material_idx = tri_hit.bvh->getMaterialIDX(tri_hit.leafMaterialIndex);
    }
    if (material_idx == -1) {
        return sample;
    }

    const Material *mat = &bvhParams.materials[material_idx];
    sample.materialIdx = material_idx;
    sample.color = { mat->color.x, mat->color.y, mat->color.z };
    sample.opacity = mat->color.w;

    if (mat->numTextures > 0) {
        int32_t texture_idx = bvhParams.materialTextures[
            mat->textureOffset + instance->worldID % mat->numTextures];
        cudaTextureObject_t *tex = bvhParams.textures + texture_idx;
        float4 texel = tex2DLod<float4>(
            *tex, tri_hit.uv.x, tri_hit.uv.y, textureLod(dist));
        sample.color.x *= texel.x;
        sample.color.y *= texel.y;
        sample.color.z *= texel.z;
        sample.opacity *= texel.w;
    }

    return sample;
}

// The opacity of the surface of an instance at a hit (see 'sampleSurface'),
// one without sampling anything for a material that holds no alpha below one.
static __device__ float surfaceOpacity(const InstanceData *instance,
                                       const TriHitInfo &tri_hit,
                                       float dist)
{
    int32_t override_mat_id = instance->matID;
    if (override_mat_id == MaterialOverride::UseOverrideColor ||
            bvhParams.materials == nullptr) {
        return 1.f;
    }
    int32_t material_idx = override_mat_id;
    if (override_mat_id == MaterialOverride::UseDefaultMaterial) {
        material_idx = tri_hit.bvh->getMaterialIDX(tri_hit.leafMaterialIndex);
    }
    if (material_idx == -1 ||
            !bvhParams.materials[material_idx].mayBeTranslucent) {
        return 1.f;
    }
    return sampleSurface(instance, tri_hit, dist).opacity;
}

static __device__ TraceResult traceRay(
    TraceInfo trace_info,
    TraceWorldInfo world_info)
{


    // We create these so that we can keep track of the original ray origin,
    // direction in world space. We will need to transform them when we enter
    // a bottom level structure.
    Vector3 ray_o = trace_info.rayOrigin;
    Vector3 ray_d = trace_info.rayDirection;
    Diag3x3 inv_ray_d = Diag3x3::fromVec(ray_d).inv();
    float t_max = trace_info.tMax;

    TraceResult result = {
        .hit = false,
        .segmentation = -1,
    };

    NodeGroup stack[64];
    uint32_t stack_size = 0;

    NodeGroup current_grp = getRootGroup(world_info);
    NodeGroup triangle_grp = invalidNodeGroup();

    // Here is some stuff we need to store in case we are now in the bottom
    // level structure. Need to maintain some extra state unfortunately.
    TriangleIsectInfo isect_info = {};
    MeshBVH *current_bvh = nullptr;
    int32_t instance_idx = -1;
    float t_scale = 1.f;

    QBVHNode *node_buffer = world_info.nodes;

    TriHitInfo tri_hit = {
        .hit = false,
        .instanceIdx = -1
    };

    for (;;) {
        if (GroupType parent_grp_type = getGroupType(current_grp); 
                parent_grp_type != GroupType::Triangles &&
                getPresentBits(current_grp) != 0) {
            // TODO: Make sure to have the node traversal order
            // sorted according to the ray direction.
            // NOTE: This should never underflow
            uint32_t child_idx = __ffs(getPresentBits(current_grp)) - 1;
            current_grp = unsetPresentBit(current_grp, child_idx);

            // Bottom level root needs to be pushed no matter what
            // so that when popping, we can restore the state of the
            // ray data.
            if (getPresentBits(current_grp) != 0 ||
                    (parent_grp_type == GroupType::BottomLevelRoot)) {
                stack[stack_size++] = current_grp;
            }

            // Intersect with the children of the child to get a new node group
            // and calculate the present bits according to which were
            // intersected
            uint32_t child_node_idx = node_buffer[current_grp & 0xFFFF'FFFF].childrenIdx[child_idx];
            bool child_is_leaf = (child_node_idx & 0x8000'0000);
            child_node_idx = child_node_idx & (~0x8000'0000);

            GroupType new_grp_type = GroupType::TopLevel;

            if (parent_grp_type == GroupType::TopLevel && child_is_leaf) {
                // Need to compute new ray o/d/etc...
                instance_idx = (int32_t)(child_node_idx & ~0x8000'0000);
                InstanceData *instance_data = world_info.instances + instance_idx;
                current_bvh = bvhParams.bvhs + world_info.instances[instance_idx].objectID;

                // Should be able to just do a continue in this case - we'll
                // just resume processing the parent node
                if (!instanceHasVolume(instance_data))
                    continue;

                ray_o = instance_data->scale.inv() *
                    instance_data->rotation.inv().rotateVec(ray_o - instance_data->position);
                ray_d = instance_data->scale.inv() *
                    instance_data->rotation.inv().rotateVec(ray_d);
                t_scale = ray_d.length();
                t_max *= t_scale;

                ray_d /= t_scale;
                inv_ray_d = Diag3x3::fromVec(ray_d).inv();
                isect_info = computeRayIsectInfo(ray_o, ray_d, inv_ray_d);

                node_buffer = current_bvh->nodes;

                new_grp_type = GroupType::BottomLevelRoot;

                // Set this to 0 so that when we read from node_buffer,
                // we're reading the root node of the mesh BVH.
                child_node_idx = 0;
            } else if (parent_grp_type == GroupType::BottomLevelRoot ||
                       parent_grp_type == GroupType::BottomLevel) {
                new_grp_type = GroupType::BottomLevel;
            }

            QBVHNode new_current = node_buffer[child_node_idx];

            Vector3 dir_quant = getDirQuant(new_current, inv_ray_d);
            Vector3 origin_quant = getOriginQuant(new_current, ray_o, inv_ray_d);

            uint32_t grp_present_bits = 0;
            uint32_t tri_present_bits = 0;

#pragma unroll
            for (int i = 0; i < MADRONA_BVH_WIDTH; ++i) {
                if (new_current.childrenIdx[i] != 0xFFFF'FFFF) {
                    auto [t_near, t_far] = getNearFar(
                            new_current, i,
                            dir_quant,
                            origin_quant,
                            t_max);

                    if (t_near <= t_far) {
                        // If we are in BLAS, check for possible triangles!
                        if ((new_grp_type == GroupType::BottomLevel ||
                                    new_grp_type == GroupType::BottomLevelRoot) &&
                                new_current.childrenIdx[i] & 0x8000'0000) {
                            // Is this child triangles?? It is if this is the leaf
                            // of a bottom level tree
                            tri_present_bits |= (((1 << new_current.triSize[i]) - 1) << 
                                    (MADRONA_BLAS_LEAF_WIDTH * i));
                        } else {
                            grp_present_bits |= (1 << i);
                        }
                    }
                }
            }

            current_grp = encodeNodeGroup(child_node_idx, grp_present_bits, new_grp_type);

            if (tri_present_bits) {
                triangle_grp = encodeNodeGroup(child_node_idx, tri_present_bits, GroupType::Triangles);
            } else {
                triangle_grp = invalidNodeGroup();
            }
        } else {
            triangle_grp = current_grp;
        }

        if (getTrianglePresentBits(triangle_grp) != 0) {
            QBVHNode parent = node_buffer[triangle_grp & 0xFFFF'FFFF];

            while (getTrianglePresentBits(triangle_grp) != 0) {
                // TODO: check active mask against heuristic to exit if not enough
                // threads are working on this
                uint32_t local_node_tri_idx = __ffs(getTrianglePresentBits(triangle_grp)) - 1;
                uint32_t local_leaf_idx = local_node_tri_idx / MeshBVH::numTrisPerLeaf;
                uint32_t tri_idx = local_node_tri_idx % MeshBVH::numTrisPerLeaf;
                uint32_t glob_leaf_idx = parent.childrenIdx[local_leaf_idx] & (~0x8000'0000);

                TriHitInfo hit_info = triangleIntersect(
                    glob_leaf_idx, tri_idx, isect_info, ray_o, t_max, current_bvh);

                // A ray goes through the texels of a cutout, whose alpha is
                // zero, and stops at any other part of a surface
                if (hit_info.hit) {
                    InstanceData *hit_instance =
                        world_info.instances + instance_idx;
                    float opacity = surfaceOpacity(
                        hit_instance, hit_info,
                        trace_info.lodDistance + hit_info.tHit / t_scale);
                    if (opacity > kTransparentOpacity) {
                        t_max = hit_info.tHit;
                        tri_hit = hit_info;
                        tri_hit.instanceIdx = instance_idx;
                    }
                }

                triangle_grp = unsetPresentBit(triangle_grp, local_node_tri_idx);
            }
        }

        // If the current node group is empty, make sure to pop something or
        // break out of the tracing loop
        if (getPresentBits(current_grp) == 0) {
            if (stack_size == 0) {
                t_max = t_max / t_scale;
                t_scale = 1.f;

                break;
            } else {
                if (getGroupType(current_grp) == GroupType::BottomLevelRoot) {
                    t_max = t_max / t_scale;
                    t_scale = 1.f;

                    // Just restore all the trace info.
                    ray_o = trace_info.rayOrigin;
                    ray_d = trace_info.rayDirection;
                    inv_ray_d = Diag3x3::fromVec(ray_d).inv();
                    instance_idx = -1;

                    node_buffer = world_info.nodes;
                }

                current_grp = stack[--stack_size];
            }
        }
    }

    if (tri_hit.hit) {
        tri_hit.tHit = t_max;

        InstanceData *instance = world_info.instances + tri_hit.instanceIdx;
        float lod_dist = trace_info.lodDistance + tri_hit.tHit;

        if (bvhParams.raycastRGBD && (!trace_info.dOnly)) {
            SurfaceSample surface = sampleSurface(instance, tri_hit, lod_dist);
            result.opacity = surface.opacity;
            result.color = surface.color;
            if (surface.materialIdx != -1) {
                const Material *mat = &bvhParams.materials[surface.materialIdx];
                result.metalness = mat->metalness;
                result.roughness = mat->roughness;
            }
            result.normal = instance->rotation.rotateVec(tri_hit.normal);
            result.segmentation = instance->objectID;
        } else {
            result.opacity = surfaceOpacity(instance, tri_hit, lod_dist);
        }

        result.depth = tri_hit.tHit;
        result.hit = true;
    }

    return result;
}

static __device__ void writeRGB(uint32_t pixel_byte_offset, const Vector3 &color)
{
    uint8_t *rgb_out = (uint8_t *)bvhParams.rgbOutput + pixel_byte_offset;

    *(rgb_out + 0) = (color.x) * 255;
    *(rgb_out + 1) = (color.y) * 255;
    *(rgb_out + 2) = (color.z) * 255;
    *(rgb_out + 3) = 255;
}

static __device__ void writeDepth(uint32_t pixel_byte_offset, float depth)
{
    float *depth_out = (float *)((uint8_t *)bvhParams.depthOutput + pixel_byte_offset);
    *depth_out = depth;
}

static __device__ void writeNormal(uint32_t pixel_byte_offset, const Vector3 &normal)
{
    uint8_t *normal_out = (uint8_t *)bvhParams.normalOutput + pixel_byte_offset;

    *(normal_out + 0) = (normal.x) * 255;
    *(normal_out + 1) = (normal.y) * 255;
    *(normal_out + 2) = (normal.z) * 255;
    *(normal_out + 3) = 255;
}

static __device__ void writeSegmentation(uint32_t pixel_byte_offset, int32_t segmentation)
{
    int32_t *segmentation_out = (int32_t *)((uint8_t *)bvhParams.segmentationOutput + pixel_byte_offset);
    *segmentation_out = segmentation;
}

static __device__ float linearToSRGB(float color) {
    if (color <= 0.00031308f) {
        return 12.92f * color;
    } else {
        return 1.055f*pow(color,(1.f / 2.4f)) - 0.055f;
    }
}

static __device__ Vector3 linearToSRGB(Vector3 color) {
    return Vector3(
        linearToSRGB(color.x),
        linearToSRGB(color.y),
        linearToSRGB(color.z)
    );
}

static __device__ float compute_attenuating(float d, float attenuation) {
    return 1.0 / (1.0 + attenuation * d * d);
}

struct FragmentResult {
    // Whether a surface covers the pixel (see 'kCoverageOpacity'), which the
    // depth, the normal and the segmentation are those of
    bool hit;
    // The color of the pixel, which translucent surfaces that cover it in no
    // other output still tint
    Vector3 color;
    Vector3 normal;
    int segmentation;
    float depth;
};

// The distance a ray continuing past a surface hit at some depth starts beyond
// that surface, so that it does not hit it again.
static __device__ inline float continuationDistance(float depth)
{
    return depth + 1e-4f * (1.f + depth);
}

// Move a ray past the surface it hit at some depth.
static __device__ inline void continueRay(TraceInfo &info, float depth)
{
    float dist = continuationDistance(depth);
    info.rayOrigin += dist * info.rayDirection;
    info.tMax -= dist;
    info.lodDistance += dist;
}

// The fraction of the light leaving a point toward a light that reaches it,
// each translucent surface on the way letting through one minus its opacity,
// and an opaque one none.
static __device__ float shadowTransmittance(Vector3 origin,
                                            Vector3 to_light,
                                            float max_dist,
                                            float lod_distance,
                                            TraceWorldInfo world_info)
{
    float transmittance = 1.f;
    TraceInfo info = {
        .rayOrigin = origin,
        .rayDirection = to_light,
        .tMin = 0.000001f,
        .tMax = max_dist,
        .dOnly = true,
        .lodDistance = lod_distance,
    };
    for (int layer = 0; layer < kMaxTranslucentLayers; ++layer) {
        TraceResult hit = traceRay(info, world_info);
        if (!hit.hit) {
            return transmittance;
        }
        transmittance *= 1.f - hit.opacity;
        if (transmittance < 1e-3f) {
            return 0.f;
        }
        continueRay(info, hit.depth);
    }
    return transmittance;
}

// The linear radiance of the surface a camera ray hit, lit by the lights of
// its world.
static __device__ Vector3 shadeHit(const TraceInfo &trace_info,
                                   const TraceResult &hit,
                                   TraceWorldInfo world_info)
{
    Vector3 hit_pos = trace_info.rayOrigin + hit.depth * trace_info.rayDirection;
    float hit_lod_distance = trace_info.lodDistance + hit.depth;

    Vector3 totalLighting = {0.f, 0.f, 0.f};

    for (int i = 0; i < world_info.numLights; ++i) {
        const LightDesc& light = world_info.lights[i];
        if(!light.active) {
            continue;
        }

        Vector3 ray_dir;
        float attenuating_factor = 1.f;
        if (light.type == LightDesc::Directional) {
            ray_dir = light.direction.normalize();
        } else {
            attenuating_factor = compute_attenuating((hit_pos - light.position).length(), light.attenuation);
            ray_dir = (hit_pos - light.position).normalize();
            if(light.cutoffAngle >= 0) {
                float angle = acosf(ray_dir.dot(light.direction.normalize()));
                if (std::abs(angle) > light.cutoffAngle) {
                    continue;
                }
            }
        }

        // A surface facing the light receives what the surfaces between them
        // let through
        if (light.castShadow) {
            if (ray_dir.dot(hit.normal) < 0.f) {
                attenuating_factor *= shadowTransmittance(
                    hit_pos, -ray_dir, 10000.f, hit_lod_distance, world_info);
                if (attenuating_factor <= 0.f) {
                    continue;
                }
            }
        }

        float n_dot_l = max(0.0, dot(hit.normal, -ray_dir));
        totalLighting += attenuating_factor * n_dot_l * hexToRgb(light.color) * light.intensity;
    }

    Vector3 color = {
        totalLighting.x * hit.color.x,
        totalLighting.y * hit.color.y,
        totalLighting.z * hit.color.z
    };

    // Add ambient term
    float ambient = 0.05;   // TODO: Ambient light is now hardcoded.
    color += hit.color * ambient;
    return color;
}

static __device__ FragmentResult computeFragment(
    TraceInfo trace_info,
    TraceWorldInfo world_info)
{
    // Direct hit first
    TraceResult layer_hit = traceRay(trace_info, world_info);

    __syncwarp();

    FragmentResult result = {
        .hit = false,
        .color = { 0.f, 0.f, 0.f },
    };

    // The ray goes through the surfaces it hits, front to back. The first one
    // whose opacity reaches the coverage cutoff gives the depth, the normal
    // and the segmentation. Each one adds its color in proportion to its
    // opacity and to what the surfaces before it let through, the black
    // background where the ray leaves the scene, and the last one counting as
    // opaque.
    bool is_rgb = bvhParams.raycastRGBD;
    Vector3 color = { 0.f, 0.f, 0.f };
    float transmittance = 1.f;
    TraceInfo layer_info = trace_info;
    for (int layer = 0; layer < kMaxTranslucentLayers; ++layer) {
        if (!layer_hit.hit) {
            break;
        }
        if (!result.hit && layer_hit.opacity >= kCoverageOpacity) {
            result.hit = true;
            result.depth = layer_info.lodDistance - trace_info.lodDistance +
                layer_hit.depth;
            result.normal = layer_hit.normal;
            result.segmentation = layer_hit.segmentation;
        }
        if (is_rgb && transmittance >= 1.f / 512.f) {
            float opacity = layer == kMaxTranslucentLayers - 1 ?
                1.f : layer_hit.opacity;
            color += (transmittance * opacity) *
                shadeHit(layer_info, layer_hit, world_info);
            transmittance *= 1.f - opacity;
        }
        if (result.hit && (!is_rgb || transmittance < 1.f / 512.f)) {
            break;
        }
        continueRay(layer_info, layer_hit.depth);
        layer_hit = traceRay(layer_info, world_info);
    }

    if (is_rgb) {
        // Convert to sRGB and clamp to 0-1
        color = linearToSRGB(color);
        result.color = Vector3::max(Vector3::zero(), Vector3::min(Vector3::one(), color));
    }
    return result;
}

extern "C" __global__ void bvhRaycastEntry()
{
    uint32_t pixels_per_block = blockDim.x;

    const uint32_t total_num_views = bvhParams.internalData->numViews;

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;
    uint32_t bytes_per_view = bvhParams.renderOutputWidth * bvhParams.renderOutputHeight * 4;
    uint32_t num_processed_pixels = 0;
    uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
    uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        PerspectiveCameraData *view_data = &bvhParams.views[current_view_offset];
        uint32_t world_idx = (uint32_t)view_data->worldID;
        Vector3 ray_start = view_data->position;
        Vector3 ray_dir = calculateOutRay(view_data, pixel_x, pixel_y);

        uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];

        // This does both the tracing / lighting, etc... just like a fragment
        // shader does in GLSL.
        FragmentResult result = computeFragment(
            TraceInfo {
                .rayOrigin = ray_start,
                .rayDirection = ray_dir,
                .tMin = view_data->zNear,
                .tMax = view_data->zFar,
                .dOnly = false,
                .lodDistance = 0.f,
            },
            TraceWorldInfo {
                .nodes = bvhParams.internalData->traversalNodes + internal_nodes_offset,
                .instances = bvhParams.instances + internal_nodes_offset,
                .lights = &bvhParams.lights[bvhParams.lightOffsets[world_idx]],
                .numLights = (uint32_t)bvhParams.lightCounts[world_idx]
            }
        );

        uint32_t linear_pixel_idx = 4 * (pixel_x + pixel_y * bvhParams.renderOutputWidth);
        uint32_t global_pixel_byte_off = current_view_offset * bytes_per_view + linear_pixel_idx;

        if (bvhParams.raycastRGBD) {
            // Write both depth and color information
            writeRGB(global_pixel_byte_off, result.color);
            if (result.hit) {
                writeDepth(global_pixel_byte_off, result.depth);
                writeNormal(global_pixel_byte_off, (result.normal + 1.0) * 0.5);
                writeSegmentation(global_pixel_byte_off, result.segmentation);
            } else {
                writeDepth(global_pixel_byte_off, INFINITY);
                writeNormal(global_pixel_byte_off, { 0.f, 0.f, 0.f });
                writeSegmentation(global_pixel_byte_off, -1);
            }
        } else {
            // Only write depth information
            if (result.hit) {
                writeDepth(global_pixel_byte_off, result.depth);
            } else {
                writeDepth(global_pixel_byte_off, INFINITY);
            }
        }

        current_view_offset += num_resident_views;

        num_processed_pixels++;

        __syncwarp();
    }
}

