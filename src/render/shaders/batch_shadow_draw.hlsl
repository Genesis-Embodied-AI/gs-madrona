#include "shader_utils.hlsl"

[[vk::push_constant]]
ShadowDrawPushConst pushConst;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<DrawData> drawDataBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

// Asset descriptor bindings
[[vk::binding(0, 1)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;


[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID) : SV_Position
{
    DrawDataBR draw_data = drawDataBuffer[draw_id + pushConst.drawDataOffset];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    uint instance_id = draw_data.instanceID;

    float4x4 shadow_matrix = shadowViewDataBuffer[draw_data.viewID].viewProjectionMatrix;

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float4 world_space_pos = float4(
        instance_data.position + mul(toMat(instance_data.rotation), (instance_data.scale * vert.position)), 
        1.f);

    float4 clip_pos = mul(shadow_matrix, world_space_pos);

    return clip_pos;
}

[shader("pixel")]
float2 frag(in float4 position : SV_Position) : SV_Target0
{
    float depth = position.z;

    // VSM
    float dx = ddx(depth);
    float dy = ddy(depth);
    float sigma = depth * depth + 0.25 * (dx * dx + dy * dy);

    return float2(depth, sigma);
}
