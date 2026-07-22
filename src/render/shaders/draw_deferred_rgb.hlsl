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
    // Deferred RGB is LDR; tone-map HDR lighting before packing it into bytes.
    uint3 quant = (uint3)(255.0f * clamp(v, 0.0f, 1.0f));
    return quant.r | (quant.g << 8) | (quant.b << 16) | ((uint32_t)255 << 24);
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

uint3 getPixelOffset(uint view_idx)
{
    uint num_views_per_image =
        pushConst.maxImagesXPerTarget * pushConst.maxImagesYPerTarget;

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

#if 1   // Used to suppress validation layer errors by referencing unused bound resources
    uint something = 0;
    something += engineInstanceBuffer[0].data[0].x;
    something += instanceOffsets[0];
    something += viewDataBuffer[0].data[0].x;
#endif

    if (idx.x >= pushConst.viewWidth || idx.y >= pushConst.viewHeight) {
        return;
    }

    uint32_t out_pixel_idx =
        pushConst.viewWidth * pushConst.viewHeight * view_idx +
        pushConst.viewWidth * idx.y + idx.x;
    PerspectiveCameraData view_data = unpackViewData(viewDataBuffer[view_idx]);

    uint3 pixel_offset = getPixelOffset(view_idx);
    uint target_idx = pixel_offset.z;
    float2 view_pixel_offset = float2(pixel_offset.xy);
    float2 view_span = float2(pushConst.viewWidth, pushConst.viewHeight);
    float2 sample_px = float2(idx.xy);
    bool sample_valid = true;

    if (view_data.projectionType == MADRONA_PROJECTION_FISHEYE_EQUIDISTANT) {
        float2 uv = (sample_px + 0.5f) / view_span;
        float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
        float aspect = -view_data.yScale / view_data.xScale;
        float2 aspect_scale = aspect >= 1.0f ? float2(aspect, 1.0f) : float2(1.0f, 1.0f / aspect);
        float2 scale_ndc = ndc * aspect_scale;
        float rho = length(scale_ndc);

        if (rho > 1.0f) {
            sample_valid = false;
        } else {
            float theta = rho * view_data.halfFov;
            float tan_theta = tan(theta);
            float2 plane_dir = rho > 1e-5f ? scale_ndc / rho : float2(0.0f, 0.0f);
            float3 dir = float3(plane_dir.x * tan_theta, 1.0f, plane_dir.y * tan_theta);

            float2 image_plane_ndc = float2(dir.x, dir.z);    // intersection with y=1
            float2 persp_ndc = image_plane_ndc * float2(view_data.xScale, view_data.yScale);   // yScale filped ndc
            sample_px = (persp_ndc * 0.5f + 0.5f) * (view_span - 1.0f);
        }
    }

    float2 atlas_px = sample_px + view_pixel_offset;

    if (renderOptionsBuffer[0].outputs[0]) {
        float3 out_color = float3(0.0f, 0.0f, 0.0f);
        if (sample_valid) {
            uint2 rgb_dims;
            rgbInBuffer[target_idx].GetDimensions(rgb_dims.x, rgb_dims.y);
            float2 rgb_atlas_uv = (atlas_px + 0.5f) / float2(rgb_dims);
            float4 color = rgbInBuffer[target_idx].SampleLevel(linearSampler, rgb_atlas_uv, 0);
            out_color = color.rgb;
        }
        rgbOutputBuffer[out_pixel_idx] = linearToSRGB8(out_color);
    }

    if (renderOptionsBuffer[0].outputs[1]) {
        float linear_depth = view_data.zFar;
        if (sample_valid) {
            uint2 depth_dims;
            depthInBuffer[target_idx].GetDimensions(depth_dims.x, depth_dims.y);
            float2 depth_atlas_uv = (atlas_px + 0.5f) / float2(depth_dims);
            float depth_in = depthInBuffer[target_idx].SampleLevel(linearSampler, depth_atlas_uv, 0).x;
            linear_depth = calculateLinearDepth(depth_in, view_idx);
        }
        depthOutputBuffer[out_pixel_idx] = linear_depth;
    }

    if (renderOptionsBuffer[0].outputs[2]) {
        float3 normal_sample = float3(0.0f, 0.0f, 0.0f);
        if (sample_valid) {
            uint2 normal_dims;
            normalInBuffer[target_idx].GetDimensions(normal_dims.x, normal_dims.y);
            float2 normal_atlas_uv = (atlas_px + 0.5f) / float2(normal_dims);
            float4 normal_tex = normalInBuffer[target_idx].SampleLevel(linearSampler, normal_atlas_uv, 0);
            normal_sample = normal_tex.xyz;
        }
        normalOutputBuffer[out_pixel_idx] = float3ToUint32(normal_sample);
    }
    
    if (renderOptionsBuffer[0].outputs[3]) {
        int segmentation = -1;
        if (sample_valid) {
            int2 segmentation_atlas_px = int2(atlas_px + 0.5f);
            segmentation = segmentationInBuffer[target_idx].Load(int3(segmentation_atlas_px, 0));
        }
        segmentationOutputBuffer[out_pixel_idx] = segmentation;
    }
}
