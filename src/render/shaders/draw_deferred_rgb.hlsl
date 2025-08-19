#include "shader_utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

// This is an array of all the textures
[[vk::binding(0, 0)]]
Texture2D<float4> rgbInBuffer[];

[[vk::binding(1, 0)]]
Texture2D<float> depthInBuffer[];

[[vk::binding(2, 0)]]
Texture2D<float4> normalInBuffer[];

[[vk::binding(3, 0)]]
Texture2D<int> segmentationInBuffer[];


[[vk::binding(4, 0)]]
RWStructuredBuffer<uint32_t> rgbOutputBuffer;

[[vk::binding(5, 0)]]
RWStructuredBuffer<float> depthOutputBuffer;

[[vk::binding(6, 0)]]
RWStructuredBuffer<uint32_t> normalOutputBuffer;

[[vk::binding(7, 0)]]
RWStructuredBuffer<int> segmentationOutputBuffer;


[[vk::binding(8, 0)]]
SamplerState linearSampler;

// Instances and views
[[vk::binding(0, 1)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 1)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 1)]]
StructuredBuffer<uint32_t> instanceOffsets;

// Lighting
[[vk::binding(0, 2)]]
StructuredBuffer<RenderOptions> renderOptionsBuffer;


#include "lighting.h"

float calculateLinearDepth(float depth_in, uint view_idx)
{
    // Calculate linear depth with reverse-z buffer
    PerspectiveCameraData cam_data = unpackViewData(viewDataBuffer[view_idx]);
    float z_near = cam_data.zNear;
    float z_far = cam_data.zFar;
    float linear_depth = (z_near * z_far) / (z_far + depth_in * (z_near - z_far));
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

uint3 getPixelOffset(uint view_idx) {
    uint num_views_per_image = pushConst.maxImagesXPerTarget * 
                               pushConst.maxImagesYPerTarget;

    uint target_idx = view_idx / num_views_per_image;

    uint target_view_idx = view_idx % num_views_per_image;

    uint target_view_idx_x = target_view_idx % pushConst.maxImagesXPerTarget;
    uint target_view_idx_y = target_view_idx / pushConst.maxImagesXPerTarget;

    uint x_pixel_offset = target_view_idx_x * pushConst.viewWidth;
    uint y_pixel_offset = target_view_idx_y * pushConst.viewHeight;

    return uint3(x_pixel_offset, y_pixel_offset, target_idx);
}

// idx.x is the x coordinate of the image
// idx.y is the y coordinate of the image
// idx.z is the global view index
[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint view_idx = idx.z;
    uint3 pixel_offset = getPixelOffset(view_idx);
#if 1
    uint something = 0;
    something += engineInstanceBuffer[0].data[0].x;
    something += instanceOffsets[0];
    something += viewDataBuffer[0].data[0].x;

    pixel_offset.x += float(min(0, something)) * 0.000000001;
#endif

    uint target_idx = pixel_offset.z;

    if (idx.x >= pushConst.viewWidth || idx.y >= pushConst.viewHeight) {
        return;
    }

    uint32_t out_pixel_idx =
        view_idx * pushConst.viewWidth * pushConst.viewHeight +
        idx.y * pushConst.viewWidth + idx.x;

    if (renderOptionsBuffer[0].outputs[0]) {    // RGB
        float4 color = rgbInBuffer[target_idx][idx.xy + pixel_offset.xy];
        float3 out_color = color.rgb;
        rgbOutputBuffer[out_pixel_idx] = linearToSRGB8(out_color); 
    }

    if (renderOptionsBuffer[0].outputs[1]) {    // Depth
        uint2 depth_dim;
        depthInBuffer[target_idx].GetDimensions(depth_dim.x, depth_dim.y);
        float2 depth_uv = (float2(idx.xy) + float2(pixel_offset.xy) + 0.5) / float2(depth_dim.xy);

        float depth_in = depthInBuffer[target_idx].SampleLevel(
                         linearSampler, depth_uv, 0).x;

        float linear_depth = calculateLinearDepth(depth_in, view_idx);

        depthOutputBuffer[out_pixel_idx] = linear_depth;
    }

    if (renderOptionsBuffer[0].outputs[2]) {    // Normal
        uint2 normal_dim;
        normalInBuffer[target_idx].GetDimensions(normal_dim.x, normal_dim.y);

        float2 normal_uv = (float2(idx.xy) + float2(pixel_offset.xy) + 0.5) / float2(normal_dim.xy);
        float3 normal_in = normalInBuffer[target_idx].SampleLevel(
                           linearSampler, normal_uv, 0).xyz;
        normalOutputBuffer[out_pixel_idx] = float3ToUint32(normal_in);
    }
    
    if (renderOptionsBuffer[0].outputs[3]) {    // Segmentation
        int out_segmentation = segmentationInBuffer[target_idx][idx.xy + pixel_offset.xy];
        segmentationOutputBuffer[out_pixel_idx] = out_segmentation;
    }
}
