#include "shader_utils.hlsl"

[[vk::push_constant]]
BatchDrawPushConst pushConst;

// Instances and views
[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> instanceOffsets;

// TODO: Make this part of lighting shader
[[vk::binding(3, 0)]]
StructuredBuffer<PackedLightData> lightDataBuffer;

[[vk::binding(4, 0)]]
StructuredBuffer<RenderOptions> renderOptionsBuffer;

[[vk::binding(5, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

// Draw information
[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(1, 1)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(2, 1)]]
RWStructuredBuffer<DrawDataBR> drawDataBuffer;

// Asset descriptor bindings
[[vk::binding(0, 2)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<MeshData> meshDataBuffer;

[[vk::binding(2, 2)]]
StructuredBuffer<MaterialData> materialBuffer;

[[vk::binding(0, 3)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 3)]]
SamplerState linearSampler;

[[vk::binding(2, 3)]]
StructuredBuffer<int> materialTexturesIndices;

[[vk::binding(0, 4)]]
Texture2D<float2> shadowMapTextures[];

// TODO: Ambient intensity is hardcoded for now. Shadow bias is hardcoded for now.
// Will implement in the future.
static const float ambient = 0.05;

struct V2F {
    [[vk::location(0)]] float4 position : SV_Position;
    [[vk::location(1)]] float3 worldPos : TEXCOORD0;
    [[vk::location(2)]] float2 uv : TEXCOORD1;
    [[vk::location(3)]] int materialIdx : TEXCOORD2;
    [[vk::location(4)]] float3 worldNormal : TEXCOORD3;
    [[vk::location(5)]] uint worldIdx : TEXCOORD4;
    [[vk::location(6)]] uint viewIdx : TEXCOORD5;
    [[vk::location(7)]] uint objectIdx : TEXCOORD6;
};


[shader("vertex")]
void vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f)
{
    DrawDataBR draw_data = drawDataBuffer[draw_id + pushConst.drawDataOffset];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    uint instance_id = draw_data.instanceID;
    PerspectiveCameraData view_data = unpackViewData(viewDataBuffer[draw_data.viewID]);
    EngineInstanceData instance_data = unpackEngineInstanceData(engineInstanceBuffer[instance_id]);

    float3 to_view_translation;
    float4 to_view_rotation;
    computeCompositeTransform(
        instance_data.position, instance_data.rotation,
        view_data.pos, view_data.rot,
        to_view_translation, to_view_rotation
    );

    float3 view_pos = rotateVec(to_view_rotation, instance_data.scale * vert.position) + to_view_translation;
    float z_far = view_data.zFar;
    float z_near = view_data.zNear;
    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        z_far / (z_far - z_near) * view_pos.y + (z_far * z_near) / (z_near - z_far),
        view_pos.y);

#if 1
    uint something = min(0, instanceOffsets[0]) +
                     min(0, drawCount[0]) +
                     min(0, drawCommandBuffer[0].vertexOffset) +
                     min(0, int(ceil(meshDataBuffer[0].vertexOffset)));
#endif

    clip_pos.x += min(0.0, abs(float(draw_data.meshID))) +
                  min(0.0, abs(float(draw_data.instanceID))) +
                  something;

    v2f.worldPos = rotateVec(instance_data.rotation, instance_data.scale * vert.position) + instance_data.position;
    v2f.position = clip_pos;
    v2f.uv = vert.uv;
    v2f.worldNormal = rotateVec(instance_data.rotation, vert.normal);
    v2f.worldIdx = instance_data.worldID;
    v2f.viewIdx = draw_data.viewID;
    v2f.objectIdx = instance_data.objectID;

    if (instance_data.matID == -1) {
        v2f.materialIdx = meshDataBuffer[draw_data.meshID].materialIndex;
    } else {
        v2f.materialIdx = instance_data.matID;
    }
}

float calculateLightAttenuating(ShaderLightData light, float3 worldPos)
{
    if (light.isDirectional) { // Directional light
        return 1.0f;
    } else { // Spot light
        float d = length(worldPos.xyz - light.position.xyz);
        return 1.0f / (1.0f + light.attenuation * d * d);
    }
}

float3 calculateRayDirection(ShaderLightData light, float3 worldPos)
{
    if (light.isDirectional) { // Directional light
        return normalize(light.direction.xyz);
    } else { // Spot light
        float3 ray_dir = normalize(worldPos.xyz - light.position.xyz);
        if(light.cutoffAngle >= 0) {
            float angle = acos(dot(normalize(ray_dir), normalize(light.direction.xyz)));
            if (abs(angle) > light.cutoffAngle) {
                return float3(0, 0, 0); // Return zero vector to indicate light should be skipped
            }
        }
        return ray_dir;
    }
}

float4 getShadowMapPixelScaleOffset(uint view_idx, uint2 shadow_map_dim)
{
    uint num_views_per_image = pushConst.maxShadowMapsXPerTarget * 
                               pushConst.maxShadowMapsYPerTarget;

    uint target_view_idx = view_idx % num_views_per_image;

    uint target_view_idx_x = target_view_idx % pushConst.maxShadowMapsXPerTarget;
    uint target_view_idx_y = target_view_idx / pushConst.maxShadowMapsXPerTarget;

    float x_pixel_offset = target_view_idx_x * pushConst.shadowMapWidth;
    float y_pixel_offset = target_view_idx_y * pushConst.shadowMapHeight;

    float2 scale = float2(pushConst.shadowMapWidth, pushConst.shadowMapHeight) / shadow_map_dim;
    float2 offset = float2(x_pixel_offset, y_pixel_offset) / shadow_map_dim;
    return float4(scale, offset);
}

float linear_step(float low, float high, float v)
{
    return clamp((v - low) / (high - low), 0, 1);
}

float samplePCF(uint shadow_map_target_idx, float2 uv, float z)
{
    float2 moment = shadowMapTextures[shadow_map_target_idx].SampleLevel(linearSampler, uv, 0).rg;

    // Chebychev's inequality
    float p = (z > moment.x);
    float sigma = max(moment.y - moment.x * moment.x, 0.0);

    float dist_from_mean = (z - moment.x);

    float pmax = linear_step(0.9, 1.0, sigma / (sigma + dist_from_mean * dist_from_mean));
    float occ = min(1.0f, max(pmax, p));

    return occ;
}

float4 calculuateLightSpacePosition(float3 world_pos, uint view_idx)
{
    float4 world_pos_v4 = float4(world_pos.xyz, 1.f);
    float4 ls_pos = mul(shadowViewDataBuffer[view_idx].viewProjectionMatrix, world_pos_v4);
    ls_pos.xyz /= ls_pos.w;

    return ls_pos;
}

bool isZOutOfRange(float z)
{
    return z > 1.0 || z < 0.0;
}

bool isUVOutOfRange(float2 uv, float4 shadow_map_uv_bounds)
{
    return uv.x < shadow_map_uv_bounds.x || uv.y < shadow_map_uv_bounds.y ||
           uv.x >= shadow_map_uv_bounds.z || uv.y >= shadow_map_uv_bounds.w;
}

/* Shadowing is done using variance shadow mapping. */
float shadowFactorVSM(float3 world_pos, uint view_idx)
{
    uint shadow_map_target_idx = view_idx / (pushConst.maxShadowMapsXPerTarget * pushConst.maxShadowMapsYPerTarget);
    uint2 shadow_map_dim;
    shadowMapTextures[shadow_map_target_idx].GetDimensions(shadow_map_dim.x, shadow_map_dim.y);

    /* Get the scale and offset to transform the shadow map UV to the shadow map texture. */
    float4 uv_scale_offset = getShadowMapPixelScaleOffset(view_idx, shadow_map_dim);

    /* Light space position */
    float4 ls_pos = calculuateLightSpacePosition(world_pos, view_idx);

    /* UV to use when sampling in the shadow map. */
    float2 uv = ls_pos.xy * 0.5 + float2(0.5, 0.5);

    /* Only deal with points which are within the shadow map. */
    if (isZOutOfRange(ls_pos.z))
        return 1.0;

    // PCF
    float pcf_count = 1;
    float occlusion = 0.0f;
    float2 shadow_map_uv = uv * uv_scale_offset.xy + uv_scale_offset.zw;
    float4 shadow_map_uv_bounds = float4(0.0, 0.0, 1.0, 1.0) * uv_scale_offset.xyxy + uv_scale_offset.zwzw;

    // Leave 1-pixel gap at each edge of the shadow map to avoid sampling to neighboring shadow maps.
    float2 texel_size = float2(1.f, 1.f) / float2(shadow_map_dim);
    shadow_map_uv_bounds.xy += texel_size;
    shadow_map_uv_bounds.zw -= texel_size;

    for (int x = int(-pcf_count); x <= int(pcf_count); ++x) {
        for (int y = int(-pcf_count); y <= int(pcf_count); ++y) {
            float2 offset_sample_uv = shadow_map_uv + float2(x, y) * texel_size;
            if(isUVOutOfRange(offset_sample_uv, shadow_map_uv_bounds))
                occlusion += 1.0;
            else
                occlusion += samplePCF(shadow_map_target_idx, offset_sample_uv, ls_pos.z);
        }
    }

    occlusion /= (pcf_count * 2.0f + 1.0f) * (pcf_count * 2.0f + 1.0f);

    return occlusion;
}

struct PixelOutput {
    float4 rgbOut : SV_Target0;
    float4 normalOut : SV_Target1;
    int segmentationOut : SV_Target2;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f, in uint prim_id : SV_PrimitiveID)
{
    PixelOutput output;
    RenderOptions renderOptions = renderOptionsBuffer[0];
    float3 normal = normalize(v2f.worldNormal);

    if (!renderOptions.outputs[0]) {
        output.rgbOut = float4(0.0, 0.0, 0.0, 1.0);
    }
    else {
        MaterialData mat_data = materialBuffer[v2f.materialIdx];
        float4 color = mat_data.color;

        int texture_idx = -1;
        uint texture_count = mat_data.numTextures;
        if (texture_count > 0) {
            uint texture_start = mat_data.textureOffset;
            texture_idx = materialTexturesIndices[texture_start + v2f.worldIdx % texture_count];
        }
        if (texture_idx != -1) {
            color *= materialTexturesArray[texture_idx].Sample(linearSampler, v2f.uv);
        }

        float3 totalLighting = float3(0.f, 0.f, 0.f);
        uint numLights = pushConst.numLights;
        float shadowFactor = shadowFactorVSM(v2f.worldPos, v2f.viewIdx);

        [unroll(1)]
        for (uint i = 0; i < numLights; i++) {
            ShaderLightData light = unpackLightData(lightDataBuffer[v2f.worldIdx * numLights + i]);
            if (!light.active) {
                continue;
            }
            
            float3 ray_dir = calculateRayDirection(light, v2f.worldPos);
            if (all(ray_dir == float3(0, 0, 0))) {
                continue;
            }

            float n_dot_l = max(0.0, dot(normal, -ray_dir));
            float attenuating_factor = calculateLightAttenuating(light, v2f.worldPos);
            totalLighting += attenuating_factor * hexToRgb(light.color).rgb * n_dot_l * light.intensity;

            // Apply shadow to the shadowed light. Only support one shadow per view for now. 
            if (i == shadowViewDataBuffer[v2f.viewIdx].lightIdx) {
                totalLighting *= shadowFactor;
            }
        }
        
        color.rgb = (totalLighting + ambient) * color.rgb;
        output.rgbOut = color;
    }

    if (renderOptions.outputs[2]) {
        output.normalOut = float4(0.5 * (normal + 1.0), 1.0);
    }

    if (renderOptions.outputs[3]) {
        output.segmentationOut = v2f.objectIdx;
    }

    return output;
}
