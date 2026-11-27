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

// The textures of the materials, the set the RGB pass binds
[[vk::binding(0, 3)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 3)]]
SamplerState linearSampler;

[[vk::binding(2, 3)]]
StructuredBuffer<int> materialTexturesIndices;


struct V2F {
    [[vk::location(0)]] float4 position : SV_Position;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] int materialIdx : TEXCOORD1;
    [[vk::location(3)]] uint worldIdx : TEXCOORD2;
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

// The opacity of the surface at a point, its material alpha times the alpha
// of its texture, the texture of the world for a material holding one per
// world (as the RGB pass reads it)
float surfaceAlpha(V2F v2f)
{
    MaterialData mat_data = materialBuffer[v2f.materialIdx];
    float alpha = mat_data.color.a;
    uint texture_count = mat_data.numTextures;
    if (texture_count > 0) {
        uint texture_start = mat_data.textureOffset;
        int texture_idx = materialTexturesIndices[texture_start + v2f.worldIdx % texture_count];
        if (texture_idx != -1) {
            alpha *= materialTexturesArray[texture_idx].Sample(linearSampler, v2f.uv).a;
        }
    }
    return alpha;
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

    EngineInstanceData instance_data = unpackEngineInstanceData(engineInstanceBuffer[instance_id]);

    float4 world_space_pos = float4(
        rotateVec(instance_data.rotation, instance_data.scale * vert.position) + instance_data.position,
        1.0);
    
    float4 clip_pos = mul(shadow_matrix, world_space_pos);
#if 1
    clip_pos.x += getSomethingVS();
#endif

    v2f.position = clip_pos;
    v2f.uv = vert.uv;
    v2f.worldIdx = instance_data.worldID;
    if (instance_data.matID == -1) {
        v2f.materialIdx = meshDataBuffer[draw_data.meshID].materialIndex;
    } else {
        v2f.materialIdx = instance_data.matID;
    }
}

struct PixelOutput {
    float2 shadowOut : SV_Target0;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f)
{
    // A surface casts the shadow of the pixels of the shadow map it covers,
    // so that a cutout casts its pattern
    if (surfaceAlpha(v2f) < ditheredAlphaThreshold(uint2(v2f.position.xy))) {
        discard;
    }

    float depth = v2f.position.z;

    // VSM
    float dx = ddx(depth);
    float dy = ddy(depth);
    float sigma = depth * depth + 0.25 * (dx * dx + dy * dy);

    PixelOutput output;
    output.shadowOut = float2(depth, sigma);
    return output;
}