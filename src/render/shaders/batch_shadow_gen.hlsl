#include "shader_utils.hlsl"

[[vk::push_constant]]
BatchShadowGenPushConst pushConst;

[[vk::binding(0, 0)]]
RWStructuredBuffer<ShadowViewData> shadowViewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedLightData> lights;

[[vk::binding(2, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

float4 invQuat(float4 rot)
{
    return float4(-rot.x, -rot.y, -rot.z, rot.w);
}

PerspectiveCameraData getCameraData(uint32_t view_idx)
{
    return unpackViewData(viewDataBuffer[view_idx]);
}

void computeOrthogonalProjectionBounds(PerspectiveCameraData unpackedView,
                                       float3 cam_pos, float3 cam_fwd, float3 cam_right, float3 cam_up,
                                       float3x3 to_light, out float3 min_bounds, out float3 max_bounds)
{
    float far_width, near_width, far_height, near_height;
    float tan_half_fov = -1.0f / unpackedView.yScale;
    float aspect = -unpackedView.yScale / unpackedView.xScale;
    float near = unpackedView.zNear;
    float far = unpackedView.zFar;

    // Only shadow the first 10% of the far plane
    // TODO: Support Cascade shadow maps to shadow the entire far plane
    far *= 0.1f;

    far_height = 2.0f * far * tan_half_fov;
    near_height = 2.0f * near * tan_half_fov;
    far_width = far_height * aspect;
    near_width = near_height * aspect;

    float3 center_near = cam_pos + cam_fwd * near;
    float3 center_far = cam_pos + cam_fwd * far;

    float far_width_half = far_width / 2.0f;
    float near_width_half = near_width / 2.0f;
    float far_height_half = far_height / 2.0f;
    float near_height_half = near_height / 2.0f;

    // f = far, n = near, l = left, r = right, t = top, b = bottom
    enum OrthoCorner {
        flt, flb,
        frt, frb,
        nlt, nlb,
        nrt, nrb
    };    

    float3 ls_corners[8];

    ls_corners[flt] = mul(to_light, center_far - cam_right * far_width_half + cam_up * far_height_half);
    ls_corners[flb] = mul(to_light, center_far - cam_right * far_width_half - cam_up * far_height_half);
    ls_corners[frt] = mul(to_light, center_far + cam_right * far_width_half + cam_up * far_height_half);
    ls_corners[frb] = mul(to_light, center_far + cam_right * far_width_half - cam_up * far_height_half);
    ls_corners[nlt] = mul(to_light, center_near - cam_right * near_width_half + cam_up * near_height_half);
    ls_corners[nlb] = mul(to_light, center_near - cam_right * near_width_half - cam_up * near_height_half);
    ls_corners[nrt] = mul(to_light, center_near + cam_right * near_width_half + cam_up * near_height_half);
    ls_corners[nrb] = mul(to_light, center_near + cam_right * near_width_half - cam_up * near_height_half);

    float x_min, x_max, y_min, y_max, z_min, z_max;

    x_min = x_max = ls_corners[0].x;
    y_min = y_max = ls_corners[0].y;
    z_min = z_max = ls_corners[0].z;

    for (uint32_t i = 1; i < 8; ++i) {
        if (x_min > ls_corners[i].x) x_min = ls_corners[i].x;
        if (x_max < ls_corners[i].x) x_max = ls_corners[i].x;

        if (y_min > ls_corners[i].y) y_min = ls_corners[i].y;
        if (y_max < ls_corners[i].y) y_max = ls_corners[i].y;

        if (z_min > ls_corners[i].z) z_min = ls_corners[i].z;
        if (z_max < ls_corners[i].z) z_max = ls_corners[i].z;
    }

    {
        float tmp = y_max;
        y_max = y_min;
        y_min = tmp;
    }

    min_bounds = float3(x_min, y_min, z_min);
    max_bounds = float3(x_max, y_max, z_max);
}

struct SharedData {
    uint32_t shadowedLightIndex;
};

groupshared SharedData sm;

[numThreads(256, 1, 1)]
[shader("compute")]
void shadowGen(uint3 idx : SV_DispatchThreadID)
{
    // Find the first shadowed light.
    if (idx.x == 0) {
        sm.shadowedLightIndex = -1;
        for (uint32_t i = 0; i < pushConst.numLights; ++i) {
            ShaderLightData light = unpackLightData(lights[i]);
            if (light.active && light.castShadow) {
                sm.shadowedLightIndex = i;
                break;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // If no shadowed light is found, skip.
    if (idx.x >= pushConst.maxNumViews)
        return;

    uint32_t view_idx = idx.x;
    shadowViewDataBuffer[view_idx].lightIdx = sm.shadowedLightIndex;
    if (sm.shadowedLightIndex == -1)
        return;

    // Get camera data
    PerspectiveCameraData unpackedView = getCameraData(view_idx);
    float3 cam_pos = unpackedView.pos;
    float4 cam_rot = invQuat(unpackedView.rot);
    float3 cam_fwd = rotateVec(cam_rot, float3(0.0f, 1.0f, 0.0f));
    float3 cam_up = rotateVec(cam_rot, float3(0.0f, 0.0f, 1.0f));
    float3 cam_right = rotateVec(cam_rot, float3(1.0f, 0.0f, 0.0f));

    // Construct orthonormal basis
    ShaderLightData light = unpackLightData(lights[sm.shadowedLightIndex]);
    float3 light_fwd = normalize(light.direction.xyz);
    float3 light_right = (light_fwd.z < 0.9999f) ?
        cross(light_fwd, float3(0.f, 0.f, 1.f)) :
        cross(light_fwd, float3(0.f, 1.f, 0.f));
    float3 light_up = cross(light_right, light_fwd);

    // Note that we use the basis vectors as the *rows* of the to_light
    // transform matrix, because we want the inverse of the light to world
    // matrix (which is just the transpose for rotation matrices).
    float3x3 to_light = float3x3(
        light_right,
        light_up,
        -light_fwd
    );

    float4x4 view = float4x4(
        float4(to_light[0], 0.f),
        float4(to_light[1], 0.f),
        float4(to_light[2], 0.f),
        float4(0.f, 0.f, 0.f, 1.f)
    );

    float4x4 projection;
    if(light.isDirectional) {
        float3 min_bounds, max_bounds;
        computeOrthogonalProjectionBounds(unpackedView, cam_pos, cam_fwd, cam_right, cam_up, to_light, min_bounds, max_bounds);
        projection = float4x4(
        float4(2.0f / (max_bounds.x - min_bounds.x), 0.0f,                                  0.0f,                                   -(max_bounds.x + min_bounds.x) / (max_bounds.x - min_bounds.x)),
        float4(0.0f,                                 2.0f / (max_bounds.y - min_bounds.y),  0.0f,                                   -(max_bounds.y + min_bounds.y) / (max_bounds.y - min_bounds.y)),
        float4(0.0f,                                 0.0f,                                  1.0f / (max_bounds.z - min_bounds.z),  -(min_bounds.z) / (max_bounds.z - min_bounds.z)),
        float4(0.0f,                                 0.0f,                                  0.0f,                                   1.0f));
    } else {
        // TODO: Support light cutoff distance to replace hardcoded radius
        float radius = 100.0f;
        float z_min = max(0.1f, radius * 0.01f);
        float z_max = radius;

        // Offset by light position
        float3 view_space_light_pos = mul(to_light, light.position.xyz);
        view[0][3] = -view_space_light_pos.x;
        view[1][3] = -view_space_light_pos.y;
        view[2][3] = view_space_light_pos.z;

        projection = float4x4(
        float4(1.0f / tan(light.cutoffAngle),      0.0f,                           0.0f,                     0.0f),
        float4(0.0f,                               1.0f / tan(light.cutoffAngle),  0.0f,                     0.0f),
        float4(0.0f,                               0.0f,                           z_max / (z_max - z_min),  -(z_max * z_min) / (z_max - z_min)),
        float4(0.0f,                               0.0f,                           1.0f,                     0.0f)
        );
    }

    shadowViewDataBuffer[view_idx].viewProjectionMatrix = mul(
        projection, view
    );

    {
        shadowViewDataBuffer[view_idx].cameraRight = float4(cam_right, 1.f);
        shadowViewDataBuffer[view_idx].cameraUp = float4(cam_up, 1.f);
        shadowViewDataBuffer[view_idx].cameraForward = float4(cam_fwd, 1.f);
        shadowViewDataBuffer[view_idx].lightIdx = sm.shadowedLightIndex;
    }
}
