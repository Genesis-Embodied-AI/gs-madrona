#include "shader_utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

// This is an array of all the textures
[[vk::binding(0, 0)]]
RWTexture2DArray<float> rgbInBuffer[];

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint32_t> rgbOutputBuffer;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> depthOutputBuffer;

[[vk::binding(3, 0)]]
Texture2D<float> depthInBuffer[];

[[vk::binding(4, 0)]]
SamplerState linearSampler;

[[vk::binding(0, 1)]]
StructuredBuffer<uint> indexBuffer;

// Instances and views
[[vk::binding(0, 2)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 2)]]
StructuredBuffer<uint32_t> instanceOffsets;


// Lighting
[[vk::binding(0, 3)]]
StructuredBuffer<PackedLightData> lights;

[[vk::binding(1, 3)]]
Texture2D<float4> transmittanceLUT;

[[vk::binding(2, 3)]]
Texture2D<float4> irradianceLUT;

[[vk::binding(3, 3)]]
Texture3D<float4> scatteringLUT;

[[vk::binding(4, 3)]]
StructuredBuffer<SkyData> skyBuffer;


#include "lighting.h"

uint zeroDummy()
{
    uint zero_dummy = min(asuint(viewDataBuffer[0].data[2].w), 0) +
                      min(asuint(engineInstanceBuffer[0].data[0].x), 0) +
                      min(indexBuffer[0], 0) +
                      min(instanceOffsets[0], 0) +
                      min(0.0, abs(transmittanceLUT.SampleLevel(
                          linearSampler, float2(0.0, 0.0f), 0).x)) +
                      min(0.0, abs(irradianceLUT.SampleLevel(
                          linearSampler, float2(0.0, 0.0f), 0).x)) +
                      min(0.0, abs(scatteringLUT.SampleLevel(
                          linearSampler, float3(0.0, 0.0f, 0.0f), 0).x)) +
                      min(0.0, abs(skyBuffer[0].solarIrradiance.x)) +
                      min(0.0, abs(float(rgbInBuffer[0][uint3(0,0,0)].x))) + 
                      min(0.0, abs(viewDataBuffer[0].data[0].x)) +
                      min(0.0, abs(engineInstanceBuffer[0].data[0].x)) +
                      min(0.0, abs(float(indexBuffer[0]))) +
                      min(0.0, abs(depthInBuffer[0].SampleLevel(linearSampler, float2(0,0), 0).x));


    return zero_dummy;
}

float linearToSRGB(float v)
{
    if (v <= 0.00031308f) {
        return 12.92f * v;
    } else {
        return 1.055f*pow(v,(1.f / 2.4f)) - 0.055f;
    }
}

uint32_t linearToSRGB8(float3 rgb)
{
    float3 srgb = float3(
        linearToSRGB(rgb.x), 
        linearToSRGB(rgb.y), 
        linearToSRGB(rgb.z));

    uint3 quant = (uint3)(255 * clamp(srgb, 0.f, 1.f));

    return quant.r | (quant.g << 8) | (quant.b << 16) | ((uint32_t)255 << 24);
}

// idx.x is the x coordinate of the image
// idx.y is the y coordinate of the image
// idx.z is the global view index
[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint view_idx = idx.z;

    uint num_views_per_image = pushConst.maxImagesXPerTarget * 
                               pushConst.maxImagesYPerTarget;

    // Figure out which image to render to
    uint target_idx = view_idx / num_views_per_image;

    // View index within that target
    uint target_view_idx = view_idx % num_views_per_image;

    uint target_view_idx_x = target_view_idx %
                             pushConst.maxImagesXPerTarget;
    uint target_view_idx_y = target_view_idx /
                             pushConst.maxImagesXPerTarget;

    float x_pixel_offset = target_view_idx_x * pushConst.viewWidth;
    float y_pixel_offset = target_view_idx_y * pushConst.viewHeight;

    if (idx.x >= pushConst.viewWidth || idx.y >= pushConst.viewHeight) {
        return;
    }

    uint3 vbuffer_pixel = uint3(idx.x, idx.y, 0);

    float2 vbuffer_pixel_clip =
        float2(float(vbuffer_pixel.x) + 0.5f, float(vbuffer_pixel.y) + 0.5f) /
        float2(pushConst.viewWidth, pushConst.viewHeight);

    vbuffer_pixel_clip = vbuffer_pixel_clip * 2.0f - float2(1.0f, 1.0f);
    vbuffer_pixel_clip.y *= -1.0;

    uint2 sample_uv_u32 = vbuffer_pixel.xy + uint2(x_pixel_offset, y_pixel_offset);

    float2 total_res = float2(pushConst.viewWidth * pushConst.maxImagesXPerTarget, pushConst.viewHeight * pushConst.maxImagesYPerTarget);

    float2 sample_uv = float2(sample_uv_u32) / total_res;
    sample_uv.y = 1.0 - sample_uv.y;

    float depth = rgbInBuffer[target_idx][vbuffer_pixel + 
                     uint3(x_pixel_offset, y_pixel_offset, 0)];

    float3 out_color = float3(depth, depth, depth);

    out_color.x += zeroDummy();

    uint32_t out_pixel_idx =
        view_idx * pushConst.viewWidth * pushConst.viewHeight +
        idx.y * pushConst.viewWidth + idx.x;

    rgbOutputBuffer[out_pixel_idx] = linearToSRGB8(float3(0 + zeroDummy(), 0, 0)); 
    depthOutputBuffer[out_pixel_idx] = depth;
}
