#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

// The alpha from which a surface covers a pixel of a view: the texels of zero
// alpha of a cutout cover none, and a surface of alpha a covers a fraction a
// of the pixels of each 4x4 block (ordered dither), showing what lies behind
// through the others. The ray tracer picks the surface giving the depth, the
// normal and the segmentation of each pixel with the same threshold (see
// 'ditheredCoverage' in bvh_raycast.cpp), so both pipelines report the same
// surface at each pixel.
float ditheredAlphaThreshold(uint2 pixel)
{
    static const float bayer[16] = {
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0,
    };
    uint2 cell = pixel & 3u;
    return (bayer[cell.y * 4u + cell.x] + 0.5) / 16.0;
}

float4 composeQuats(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 rotateVec(float4 q, float3 v)
{
    float3 pure = q.xyz;
    float scalar = q.w;
    
    float3 pure_x_v = cross(pure, v);
    float3 pure_x_pure_x_v = cross(pure, pure_x_v);
    
    return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
}

PerspectiveCameraData unpackViewData(PackedViewData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];
    const float4 d3 = packed.data[3];

    PerspectiveCameraData cam;
    cam.pos = d0.xyz;
    cam.rot = float4(d1.xyz, d0.w);
    cam.xScale = d1.w;
    cam.yScale = d2.x;
    cam.zNear = d2.y;
    cam.zFar = d2.z;
    cam.worldID = asint(d2.w);
    cam.halfFov = d3.x;
    cam.projectionType = asuint(d3.y);

    return cam;
}

Vertex unpackVertex(PackedVertex packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];

    uint3 packed_normal_tangent = uint3(asuint(d0.w), asuint(d1.x), asuint(d1.y));
    float3 normal;
    float4 tangent_and_sign;
    decodeNormalTangent(packed_normal_tangent, normal, tangent_and_sign);

    Vertex vert;
    vert.position = float3(d0.x, d0.y, d0.z);
    vert.normal = normal;
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = float2(d1.z, d1.w);

    return vert;
}

EngineInstanceData unpackEngineInstanceData(PackedInstanceData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];
    const float4 d3 = packed.data[3];

    EngineInstanceData o;
    o.position = d0.xyz;
    o.rotation = float4(d1.xyz, d0.w);
    o.scale = float3(d1.w, d2.xy);
    o.matID = asint(d2.z);
    o.objectID = asint(d2.w);
    o.worldID = asint(d3.x);

    return o;
}

ShaderLightData unpackLightData(PackedLightData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];

    ShaderLightData o;
    o.position = d0.xyz;
    o.direction = float3(d0.w, d1.xy);
    o.color = asuint(d1.z);
    o.cutoffAngle = d1.w;
    o.attenuation = d2.x;
    o.intensity = d2.y;
    const uint32_t rest = asuint(d2.z);
    o.isDirectional = (rest & 0xFF) != 0;
    o.castShadow = (rest & 0xFF00) != 0;
    o.active = (rest & 0xFF0000) != 0;

    return o;
}

float3x3 toMat(float4 r)
{
    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    return float3x3(
        float3(
            1.f - 2.f * (y2 + z2),
            2.f * (xy - wz),
            2.f * (xz + wy)),
        float3(
            2.f * (xy + wz),
            1.f - 2.f * (x2 + z2),
            2.f * (yz - wx)),
        float3(
            2.f * (xz - wy),
            2.f * (yz + wx),
            1.f - 2.f * (x2 + y2)));
}

void computeCompositeTransform(float3 obj_t,
                               float4 obj_r,
                               float3 cam_t,
                               float4 cam_r_inv,
                               out float3 to_view_translation,
                               out float4 to_view_rotation)
{
    to_view_translation = rotateVec(cam_r_inv, obj_t - cam_t);
    to_view_rotation = normalize(composeQuats(cam_r_inv, obj_r));
}

float4 projectToClip(PerspectiveCameraData view_data,
                     float3 view_pos,
                     float clip_z)
{
    return float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        clip_z,
        view_pos.y);
}

// We are basically packing 3 uints into 2. 21 bits per uint except for 22 
// for the instance ID
uint2 packVizBufferData(uint primitive_id, uint mesh_id, uint instance_id)
{
    primitive_id += 1;
    mesh_id += 1;
    instance_id += 1;

    uint d0 = primitive_id << 11;
    d0 |= 0x7FF & (instance_id >> 11);
    uint d1 = mesh_id << 11;
    d1 |= 0x7FF & instance_id;
    return uint2(d0, d1);
}

uint3 unpackVizBufferData(in uint2 data)
{
    uint primitive_id = data.x >> 11;
    uint mesh_id = data.y >> 11;
    uint instance_id = ((data.x & 0x7FF) << 11) | (data.y & 0x7FF);

    return uint3(primitive_id-1, mesh_id-1, instance_id-1);
}

float4 intToColor(uint inCol)
{
    if(inCol == 0xffffffff) return float4(1,1,1,1);

    float a = ((inCol & 0xff000000) >>24);
    float r = ((inCol & 0xff0000) >> 16);
    float g = ((inCol & 0xff00) >> 8);
    float b = ((inCol & 0xff));
    return float4(r,g,b,255.0)/255.0;
}

uint hash(uint x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}
