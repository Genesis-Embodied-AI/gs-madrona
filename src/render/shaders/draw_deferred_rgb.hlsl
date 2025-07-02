#include "shader_utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

// This is an array of all the textures
[[vk::binding(0, 0)]]
RWTexture2DArray<float4> vizBuffer[];

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint32_t> rgbOutputBuffer;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> depthOutputBuffer;

[[vk::binding(3, 0)]]
Texture2D<float> depthInBuffer[];

[[vk::binding(4, 0)]]
Texture2D<float2> shadowMapBuffer[];

[[vk::binding(5, 0)]]
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

[[vk::binding(5, 3)]]
StructuredBuffer<RenderOptions> renderOptionsBuffer;


#include "lighting.h"

#define SHADOW_BIAS 0.002f

float calculateLinearDepth(float depth_in)
{
    // Calculate linear depth with reverse-z buffer
    PerspectiveCameraData cam_data = unpackViewData(viewDataBuffer[0]);
    float z_near = cam_data.zNear;
    float z_far = cam_data.zFar;
    float linear_depth = z_far * z_near / (z_near - depth_in * (z_near - z_far));

    return linear_depth;
}

uint32_t float3ToUint32(float3 v)
{
    return (uint32_t)(v.x * 255.0f) | ((uint32_t)(v.y * 255.0f) << 8) | ((uint32_t)(v.z * 255.0f) << 16) | (255 << 24);
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

    return float3ToUint32(srgb);
}

float3 getShadowMapPixelOffset(uint view_idx) {
    uint num_views_per_image = pushConst.maxShadowMapXYPerTarget * 
                               pushConst.maxShadowMapXYPerTarget;

    uint target_idx = view_idx / num_views_per_image;

    uint target_view_idx = view_idx % num_views_per_image;

    uint target_view_idx_x = target_view_idx % pushConst.maxShadowMapXYPerTarget;
    uint target_view_idx_y = target_view_idx / pushConst.maxShadowMapXYPerTarget;

    float x_pixel_offset = target_view_idx_x * pushConst.shadowMapSize;
    float y_pixel_offset = target_view_idx_y * pushConst.shadowMapSize;

    return float3(x_pixel_offset, y_pixel_offset, target_idx);
}

/* Shadowing is done using variance shadow mapping. */
float shadowFactorVSM(float3 world_pos)
{
    float3 shadow_map_pixel_offset = getShadowMapPixelOffset(view_idx);
    uint shadow_map_target_idx = shadow_map_pixel_offset.z;

    /* Light space position */
    float4 world_pos_v4 = float4(world_pos.xyz, 1.f);
    float4 ls_pos = mul(shadowViewDataBuffer[pushConst.viewIdx].viewProjectionMatrix, 
                        world_pos_v4);
    ls_pos.xyz /= ls_pos.w;
    ls_pos.z += SHADOW_BIAS;

    /* UV to use when sampling in the shadow map. */
    float2 uv = ls_pos.xy * 0.5 + float2(0.5, 0.5);

    /* Only deal with points which are within the shadow map. */
    if (uv.x > 1.0 || uv.x < 0.0 || uv.y > 1.0 || uv.y < 0.0 ||
        ls_pos.z > 1.0 || ls_pos.z < 0.0)
        return 1.0;

    uint2 shadow_map_dim = shadowMapBuffer[shadow_map_target_idx].GetDimensions();
    float2 texel_size = float2(1.f, 1.f) / float2(shadow_map_dim);
    float2 shadow_map_uv = (uv + shadow_map_pixel_offset.xy) / float2(shadow_map_dim);
    float2 moment = shadowMapBuffer[shadow_map_target_idx].SampleLevel(linearSampler, shadow_map_uv, 0);

    float occlusion = 0.0f;

    // PCF
    float pcf_count = 1;

    for (int x = int(-pcf_count); x <= int(pcf_count); ++x) {
        for (int y = int(-pcf_count); y <= int(pcf_count); ++y) {
            float2 moment = shadowMap.SampleLevel(linearSampler, 
                                                  uv + float2(x, y) * texel_size, 0).rg;

            // Chebychev's inequality
            float p = (ls_pos.z > moment.x);
            float sigma = max(moment.y - moment.x * moment.x, 0.0);

            float dist_from_mean = (ls_pos.z - moment.x);

            float pmax = linear_step(0.9, 1.0, sigma / (sigma + dist_from_mean * dist_from_mean));
            float occ = min(1.0f, max(pmax, p));

            occlusion += occ;
        }
    }

    occlusion /= (pcf_count * 2.0f + 1.0f) * (pcf_count * 2.0f + 1.0f);

    return occlusion;
}

float3 getPixelOffset(uint view_idx) {
    uint num_views_per_image = pushConst.maxImagesXPerTarget * 
                               pushConst.maxImagesYPerTarget;

    uint target_idx = view_idx / num_views_per_image;

    uint target_view_idx = view_idx % num_views_per_image;

    uint target_view_idx_x = target_view_idx % pushConst.maxImagesXPerTarget;
    uint target_view_idx_y = target_view_idx / pushConst.maxImagesXPerTarget;

    float x_pixel_offset = target_view_idx_x * pushConst.viewWidth;
    float y_pixel_offset = target_view_idx_y * pushConst.viewHeight;

    return float3(x_pixel_offset, y_pixel_offset, target_idx);
}

// idx.x is the x coordinate of the image
// idx.y is the y coordinate of the image
// idx.z is the global view index
[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint view_idx = idx.z;
    float3 pixel_offset = getPixelOffset(view_idx);
    uint target_idx = pixel_offset.z;

    if (idx.x >= pushConst.viewWidth || idx.y >= pushConst.viewHeight) {
        return;
    }

    uint3 vbuffer_pixel = uint3(idx.x, idx.y, 0);
    uint32_t out_pixel_idx =
        view_idx * pushConst.viewWidth * pushConst.viewHeight +
        idx.y * pushConst.viewWidth + idx.x;

    if (renderOptionsBuffer[0].outputRGB) {

        float4 color = vizBuffer[target_idx][vbuffer_pixel + 
                         uint3(pixel_offset.xy, 0)];
        float3 out_color = color.rgb;

        rgbOutputBuffer[out_pixel_idx] = linearToSRGB8(out_color); 
    }

    if (renderOptionsBuffer[0].outputDepth) 
    {
        uint2 depth_dim;
        depthInBuffer[target_idx].GetDimensions(depth_dim.x, depth_dim.y);
        float2 depth_uv = float2(vbuffer_pixel.x + pixel_offset.x + 0.5, 
                                vbuffer_pixel.y + pixel_offset.y + 0.5) / 
                        float2(depth_dim.x, depth_dim.y);

        float depth_in = depthInBuffer[target_idx].SampleLevel(
                         linearSampler, depth_uv, 0).x;

        float linear_depth = calculateLinearDepth(depth_in);

        depthOutputBuffer[out_pixel_idx] = linear_depth;
    }
}
