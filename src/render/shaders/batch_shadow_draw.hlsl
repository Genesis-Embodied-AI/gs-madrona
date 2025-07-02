#include "shader_utils.hlsl"

[[vk::push_constant]]
ShadowDrawPushConst pushConst;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(1, 0)]]
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


struct V2F {
    [[vk::location(0)]] float4 position : SV_Position;
};

float getSomethingVS()
{
    // This is a hack to reference unused bound resources to avoid errors.
    // TODO: Remove this once we have a proper way to handle this.
    uint something = 0;
    something += drawCount[0];
    something += drawCommandBuffer[0].vertexOffset;
    something += meshDataBuffer[0].vertexOffset;

    return float(min(0, something)) * 0.000000001;
}

float getSomethingPS()
{
    // This is a hack to reference unused bound resources to avoid errors.
    // TODO: Remove this once we have a proper way to handle this.
    uint something = min(0, int(ceil(materialBuffer[0].color.w)));

    return float(min(0, something)) * 0.000000001;
}

[shader("vertex")]
void vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f)
{
    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    DrawDataBR draw_data = drawDataBuffer[draw_id + pushConst.drawDataOffset];
    uint instance_id = draw_data.instanceID;

    float4x4 shadow_matrix = shadowViewDataBuffer[draw_data.viewID].viewProjectionMatrix;

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float4 world_space_pos = float4(
        rotateVec(instance_data.rotation, instance_data.scale * vert.position) + instance_data.position,
        1.0);
    
    float4 clip_pos = mul(shadow_matrix, world_space_pos);
#if 1
    clip_pos.x += getSomethingVS();
#endif

    v2f.position = clip_pos;
}

struct PixelOutput {
    float2 shadowOut : SV_Target0;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f)
{
    float depth = v2f.position.z;
#if 1
    depth += getSomethingPS();
#endif

    // VSM
    float dx = ddx(depth);
    float dy = ddy(depth);
    float sigma = depth * depth + 0.25 * (dx * dx + dy * dy);

    PixelOutput output;
    output.shadowOut = float2(depth, sigma);
    return output;
}