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

[[vk::binding(3, 3)]]
Texture2D<float2> shadowMapTextures[];

struct V2F {
    [[vk::location(0)]] float4 position : SV_Position;
    [[vk::location(1)]] float3 worldPos : TEXCOORD0;
    [[vk::location(2)]] float2 uv : TEXCOORD1;
    [[vk::location(3)]] int materialIdx : TEXCOORD2;
    [[vk::location(4)]] uint color : TEXCOORD3;
    [[vk::location(5)]] float3 worldNormal : TEXCOORD4;
    [[vk::location(6)]] uint worldIdx : TEXCOORD5;
    [[vk::location(7)]] uint viewIdx : TEXCOORD6;
};

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
float shadowFactorVSM(float3 world_pos, uint view_idx)
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

    uint2 shadow_map_dim = shadowMapTextures[shadow_map_target_idx].GetDimensions();
    float2 texel_size = float2(1.f, 1.f) / float2(shadow_map_dim);
    float2 shadow_map_uv = (uv + shadow_map_pixel_offset.xy) / float2(shadow_map_dim);
    float2 moment = shadowMapTextures[shadow_map_target_idx].SampleLevel(linearSampler, shadow_map_uv, 0);

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

// TODO: Ambient intensity is hardcoded for now.  Will implement in the future.
static const float ambient = 0.2;

[shader("vertex")]
void vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f)
{
    DrawDataBR draw_data = drawDataBuffer[draw_id + pushConst.drawDataOffset];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    uint instance_id = draw_data.instanceID;

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[draw_data.viewID]);

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float3 to_view_translation;
    float4 to_view_rotation;
    computeCompositeTransform(instance_data.position, instance_data.rotation,
        view_data.pos, view_data.rot,
        to_view_translation, to_view_rotation);

    float3 view_pos =
        rotateVec(to_view_rotation, instance_data.scale * vert.position) +
            to_view_translation;

    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);

    v2f.worldPos = rotateVec(instance_data.rotation, instance_data.scale * vert.position) + instance_data.position;
    v2f.position = clip_pos;
    v2f.uv = vert.uv;
    v2f.worldNormal = rotateVec(instance_data.rotation, vert.normal);
    v2f.worldIdx = instance_data.worldID;
    v2f.viewIdx = draw_data.viewID;

    if (instance_data.matID == -2) {
        v2f.materialIdx = -2;
        v2f.color = instance_data.color;
    } else if (instance_data.matID == -1) {
        v2f.materialIdx = meshDataBuffer[draw_data.meshID].materialIndex;
        v2f.color = 0;
    } else {
        v2f.materialIdx = instance_data.matID;
        v2f.color = 0;
    }
}

float3 calculateRayDirection(ShaderLightData light, float3 worldPos) {
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

struct PixelOutput {
    float4 rgbOut : SV_Target0;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f,
                 in uint prim_id : SV_PrimitiveID)
{
    PixelOutput output;

    RenderOptions renderOptions = renderOptionsBuffer[0];

    float3 normal = normalize(v2f.worldNormal);

    if (!renderOptions.outputRGB) {
        output.rgbOut = float4(0.0, 0.0, 0.0, 1.0);
    }
    else {

        if (v2f.materialIdx == -2) {
            output.rgbOut = hexToRgb(v2f.color);
        } else {
            MaterialData mat_data = materialBuffer[v2f.materialIdx];
            float4 color = mat_data.color;
            
            if (mat_data.textureIdx != -1) {
                color *= materialTexturesArray[mat_data.textureIdx].Sample(
                        linearSampler, v2f.uv);
            }

            float3 totalLighting = 0;
            uint numLights = pushConst.numLights;
            float shadowFactor = shadowFactorVSM(v2f.worldPos, v2f.viewIdx);

            [unroll(1)]
            for (uint i = 0; i < numLights; i++) {
                ShaderLightData light = unpackLightData(lightDataBuffer[v2f.worldIdx * numLights + i]);
                if(!light.active) {
                    continue;
                }
                
                float3 ray_dir = calculateRayDirection(light, v2f.worldPos);
                if (all(ray_dir == float3(0, 0, 0))) {
                    continue;
                }

                float n_dot_l = max(0.0, dot(normal, -ray_dir));
                // Only apply shadow to the first light
                if (i == 0) {
                    totalLighting += n_dot_l * light.intensity * shadowFactor;
                } else {
                    totalLighting += n_dot_l * light.intensity;
                }
            }

            float3 lighting = totalLighting * color.rgb;
            lighting += color.rgb * ambient;
            
            color.rgb = lighting;
            output.rgbOut = color;
        }
    }

    return output;
}
