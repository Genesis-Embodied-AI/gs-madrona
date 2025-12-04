#include "shader_utils.hlsl"

// Post-processing push constants
struct PostProcessPushConst {
    uint32_t viewWidth;
    uint32_t viewHeight;
    uint32_t totalViews;
    uint32_t blurRadius; // blur kernel radius
};

[[vk::push_constant]]
PostProcessPushConst pushConst;

// Input/Output buffers
[[vk::binding(0, 0)]]
RWStructuredBuffer<uint32_t> rgbBuffer; // Input and output RGB buffer

// ------------------------------------------------------------------------------------------------
// Sample RGB from buffer with bounds checking
uint32_t sampleRGB(uint32_t view_idx, int2 coord) {
    // Clamp coordinates to view bounds
    coord = clamp(coord, int2(0, 0), int2(pushConst.viewWidth - 1, pushConst.viewHeight - 1));
    
    uint32_t pixel_idx = view_idx * pushConst.viewWidth * pushConst.viewHeight +
                         coord.y * pushConst.viewWidth + coord.x;
    
    return rgbBuffer[pixel_idx];
}

// ------------------------------------------------------------------------------------------------
float3 unpackRGB8(uint32_t packed) {
    uint3 rgb;
    rgb.r = (packed) & 0xFF;
    rgb.g = (packed >> 8) & 0xFF;
    rgb.b = (packed >> 16) & 0xFF;
    
    return float3(rgb) / 255.0f;
}

// ------------------------------------------------------------------------------------------------
uint32_t packRGB8(float3 rgb) {
    uint3 quant = (uint3)(255 * clamp(rgb, 0.f, 1.f));
    return quant.r | (quant.g << 8) | (quant.b << 16) | ((uint32_t)255 << 24);
}

// ------------------------------------------------------------------------------------------------
// Sample RGB as float3 with bounds checking
float3 sampleRGBFloat(uint32_t view_idx, int2 coord) {
    return unpackRGB8(sampleRGB(view_idx, coord));
}

// ------------------------------------------------------------------------------------------------
// Bilinear interpolation helper
float3 bilerp(float3 a, float3 b, float3 c, float3 d, float2 t) {
    float3 top = lerp(a, b, t.x);    // Interpolate between top-left and top-right
    float3 bottom = lerp(c, d, t.x); // Interpolate between bottom-left and bottom-right
    return lerp(top, bottom, t.y);   // Interpolate between top and bottom
}

// ------------------------------------------------------------------------------------------------
// Texture sampling with bilinear filtering
// uvs are normalized coordinates [0,1]
// offset is in pixels
// Equivalent to GLSL textureOffset() function
float3 sampleBilinearColor(uint32_t view_idx, float2 uv, float2 offset = float2(0.f, 0.f)) {
    // Convert pixel offset to normalized coordinates
    float2 offsetUV = float2(offset) / float2(pushConst.viewWidth, pushConst.viewHeight);

    // Apply offset to normalized coordinates
    uv += offsetUV;

    // Convert normalized coordinates to pixel coordinates
    float2 texelCoord = uv * float2(pushConst.viewWidth, pushConst.viewHeight);
    
    // Get the integer coordinates of the four surrounding texels
    int2 coord00 = int2(floor(texelCoord));
    int2 coord10 = coord00 + int2(1, 0);
    int2 coord01 = coord00 + int2(0, 1);
    int2 coord11 = coord00 + int2(1, 1);
    
    // Calculate fractional part for interpolation
    float2 fract = texelCoord - float2(coord00);
    
    // Sample the four surrounding texels
    float3 sample00 = sampleRGBFloat(view_idx, coord00); // Top-left
    float3 sample10 = sampleRGBFloat(view_idx, coord10); // Top-right
    float3 sample01 = sampleRGBFloat(view_idx, coord01); // Bottom-left
    float3 sample11 = sampleRGBFloat(view_idx, coord11); // Bottom-right
    
    // Perform bilinear interpolation
    return bilerp(sample00, sample10, sample01, sample11, fract);
}

// ------------------------------------------------------------------------------------------------
// Texture sampling with guassian filtering
float3 sampleGaussianColor(uint view_idx, float2 uv, float2 offset = float2(0.f, 0.f)) {
    float2 resolution = float2(pushConst.viewWidth, pushConst.viewHeight);

    // Convert pixel offset to normalized coordinates
    float2 offsetUV = float2(offset) / resolution;
    // Apply offset to normalized coordinates
    uv += offsetUV;

    float2 texelCoord = uv * resolution + 0.5;
    int2 baseCoord = int2(floor(texelCoord));
    
    // Fractional part is ignored here — this is a centered filter.
    float3 result = float3(0.0, 0.0, 0.0);
    
    float weightSum = 0;
    
    const int2 offsets[9] = {
        int2(-1, -1), int2(0, -1), int2(1, -1),
        int2(-1,  0), int2(0,  0), int2(1,  0),
        int2(-1,  1), int2(0,  1), int2(1,  1)
    };
    
    const float weights[9] = {
        0.5, 0.7, 0.5,
        0.7, 2.5, 0.7,
        0.5, 0.7, 0.5
    };
    
    for (int i = 0; i < 9; ++i) {
        int2 coord = baseCoord + offsets[i];
        float3 color = sampleRGBFloat(view_idx, coord);
        result += weights[i] * color;
        weightSum += weights[i];
    }
    
    return result / float(weightSum);
}

// ------------------------------------------------------------------------------------------------
float3 sampleColor(uint32_t view_idx, float2 uv, float2 offset = float2(0.f, 0.f)) {
    return sampleBilinearColor(view_idx, uv, offset);
}

// ------------------------------------------------------------------------------------------------
// Similar to textureOffset but return luminance directly
float sampleLuminance(uint32_t view_idx, float2 uv, float2 offset = float2(0.f, 0.f)) {
    return rgbToLuminance(sampleColor(view_idx, uv, offset));
}

// ------------------------------------------------------------------------------------------------
// Edge detection result
struct EdgeResult {
    bool isEdge;
    float lumaRange;
};

// given center and neighbor luminances, is anti-aliasing needed?
EdgeResult detectEdge(float lumaM, float lumaN, float lumaS, float lumaE, float lumaW) {
    // Find the minimum and maximum luma around the current pixel
    float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));

    EdgeResult result;

    // Compute the delta
    result.lumaRange = lumaMax - lumaMin;
    
    // If the luma variation is lower than a threshold, we are not on an edge
    const float fxaaQualityEdgeThreshold = 0.02;
    const float fxaaQualityEdgeMax = 0.125;
    
    result.isEdge = (result.lumaRange >= max(fxaaQualityEdgeThreshold, lumaMax * fxaaQualityEdgeMax));

    return result;
}

// ------------------------------------------------------------------------------------------------
// is horizontal or vertical edge?
bool isHorizontal(float lumaM, float lumaN, float lumaS, float lumaE, float lumaW, 
                  float lumaNE, float lumaNW, float lumaSE, float lumaSW) {
   // Combine the four edge lumas
    float lumaNS = lumaN + lumaS;
    float lumaWE = lumaW + lumaE;
    
    // Combine the four corner lumas
    float lumaW_ = lumaNW + lumaSW;
    float lumaE_ = lumaNE + lumaSE;
    float lumaN_ = lumaNW + lumaNE;
    float lumaS_ = lumaSW + lumaSE;

    // Compute an estimation of the gradient along the horizontal and vertical axis.
	float edgeHoriz	= abs(-2.0 * lumaW + lumaW_) + abs(-2.0 * lumaM + lumaNS) * 2.0 + abs(-2.0 * lumaE + lumaE_);
	float edgeVert 	= abs(-2.0 * lumaS + lumaS_) + abs(-2.0 * lumaM + lumaWE) * 2.0 + abs(-2.0 * lumaN + lumaN_);
    
    // Is the local edge horizontal or vertical?
    return edgeHoriz >= edgeVert;
}

// ------------------------------------------------------------------------------------------------
float3 applyFXAA(uint32_t view_idx, int2 coord) {    
    // Sample the center pixel
    float2 texelSize = float2(1.0 / float(pushConst.viewWidth), 1.0 / float(pushConst.viewHeight));
    float2 coordf = float2(coord) * texelSize;// + 0.5 * texelSize;
    float3 rgbM = sampleColor(view_idx, coordf);
    float lumaM = rgbToLuminance(rgbM);

    // 1.Detecting where to apply AA
    
    // Sample neighbors
    float lumaS = sampleLuminance(view_idx, coordf, float2( 0.f, -1.f));  // South
    float lumaN = sampleLuminance(view_idx, coordf, float2( 0.f,  1.f));  // North
    float lumaE = sampleLuminance(view_idx, coordf, float2(-1.f,  0.f));  // East
    float lumaW = sampleLuminance(view_idx, coordf, float2( 1.f,  0.f));  // West

    EdgeResult edgeDetection = detectEdge(lumaM, lumaN, lumaS, lumaE, lumaW);
    if (!edgeDetection.isEdge) {
        return sampleRGBFloat(view_idx, coord); // No anti-aliasing needed
    }
    float lumaRange = edgeDetection.lumaRange;

    // 2. Estimating gradient and choosing edge direction

    // Sample the corners
    float lumaNW = sampleLuminance(view_idx, coordf, float2( 1.f,  1.f));
    float lumaNE = sampleLuminance(view_idx, coordf, float2(-1.f,  1.f));
    float lumaSW = sampleLuminance(view_idx, coordf, float2( 1.f, -1.f));
    float lumaSE = sampleLuminance(view_idx, coordf, float2(-1.f, -1.f));
    
    // Is the local edge horizontal or vertical?
    bool horzSpan = isHorizontal(lumaM, lumaN, lumaS, lumaE, lumaW, lumaNE, lumaNW, lumaSE, lumaSW);
    
    // Select the two neighboring texels lumas in the opposite direction to the local edge
    float luma1 = horzSpan ? lumaS : lumaE;
    float luma2 = horzSpan ? lumaN : lumaW;
    
    // Compute gradients in this direction
    float gradient1 = luma1 - lumaM;
    float gradient2 = luma2 - lumaM;
    
    // Which direction is the steepest?
    bool is1Steepest = abs(gradient1) >= abs(gradient2);
    
    // Gradient in the corresponding direction, normalized
    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));
    
    // Choose the step size (how far to go on each iteration) according to the edge direction
    float stepLength = horzSpan ? texelSize.y : texelSize.x;

    // Average luma in the correct direction
    float lumaLocalAverage = 0.0;
    if (is1Steepest) {
        // Switch the direction
        stepLength = -stepLength;
        lumaLocalAverage = 0.5 * (luma1 + lumaM);
    } else {
        lumaLocalAverage = 0.5 * (luma2 + lumaM);
    }

    // Shift UV in the correct direction
    float2 currentUv = float2(coordf);
    if (horzSpan) {
        currentUv.y += stepLength * 0.5;
    } else {
        currentUv.x += stepLength * 0.5;
    }
    
    // Compute offset (for each iteration step) in the right direction
    float2 offset = horzSpan ? float2(texelSize.x, 0.0) : float2(0.0, texelSize.y);
    
    // Compute UVs to explore on each side of the edge, orthogonally
    float2 uv1 = currentUv - offset;
    float2 uv2 = currentUv + offset;
    
    // Read the lumas at both current extremities of the exploration segment
    float lumaEnd1 = sampleLuminance(view_idx, uv1);
    float lumaEnd2 = sampleLuminance(view_idx, uv2);

    lumaEnd1 -= lumaLocalAverage;
    lumaEnd2 -= lumaLocalAverage;
    
    // If the luma deltas at the current extremities are larger than the local gradient, we have reached the side of the edge
    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;
    bool reachedBoth = reached1 && reached2;
    
    // If the side is not reached, we continue to explore in this direction
    if (!reached1) {
        uv1 -= offset;
    }
    if (!reached2) {
        uv2 += offset;
    }
    
    // If both sides have not been reached, continue to explore
    #define ITERATIONS 10
	const float QUALITY[ITERATIONS] = {1.5f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
    if (!reachedBoth) {
        for (int i = 0; i < ITERATIONS; i++) { // Maximum 12 iterations
            // If needed, read luma in 1st direction, compute delta
            if (!reached1) {
                lumaEnd1 = sampleLuminance(view_idx, uv1);
                lumaEnd1 = lumaEnd1 - lumaLocalAverage;
            }
            // If needed, read luma in opposite direction, compute delta
            if (!reached2) {
                lumaEnd2 = sampleLuminance(view_idx, uv2);
                lumaEnd2 = lumaEnd2 - lumaLocalAverage;
            }
            // If the luma deltas at the current extremities is larger than the local gradient, we have reached the side of the edge
            reached1 = abs(lumaEnd1) >= gradientScaled;
            reached2 = abs(lumaEnd2) >= gradientScaled;
            reachedBoth = reached1 && reached2;
            
            // If the side is not reached, we continue to explore in this direction, with a variable quality
            if (!reached1) {
                uv1 -= offset * QUALITY[i];
            }
            if (!reached2) {
                uv2 += offset * QUALITY[i];
            }
            
            // If both sides have been reached, stop the exploration
            if (reachedBoth) {
                break;
            }
        }
    }
    
    // Compute the distances to each extremity of the edge
    float distance1 = horzSpan ? (coordf.x - uv1.x) : (coordf.y - uv1.y);
    float distance2 = horzSpan ? (uv2.x - coordf.x) : (uv2.y - coordf.y);
    
    // In which direction is the extremity of the edge closer?
    bool isDirection1 = distance1 < distance2;
    float distanceFinal = min(distance1, distance2);
    
    // Length of the edge
    float edgeLength = (distance1 + distance2);
    
    // UV offset: read in the direction of the closest side of the edge
    float pixelOffset = -distanceFinal / edgeLength + 0.5;
    
    // Is the luma at center smaller than the local average?
    bool isLumaMLowerThanAvg = lumaM < lumaLocalAverage;
    
    // If the luma at center is smaller than at its neighbor, the delta luma at each end should be positive (same variation)
    bool correctVariation = ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaMLowerThanAvg;
    
    // If the luma variation is incorrect, do not offset
    float finalOffset = correctVariation ? pixelOffset : 0.0;
    
    // Sub-pixel shifting
    // Full weighted average of the luma over the 3x3 neighborhood
    float lumaAverage = (1.0/12.0) * (2.0 * (lumaN + lumaE + lumaS + lumaW) + lumaNE + lumaNW + lumaSE + lumaSW);
    
    // Ratio of the delta between the global average and the center luma, over the luma range in the 3x3 neighborhood
    float subPixelOffset1 = clamp(abs(lumaAverage - lumaM) / lumaRange, 0.0, 1.0);
    float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
    
    // Compute a sub-pixel offset based on this delta
    const float fxaaQualitySubpixel = 0.75;
    float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * fxaaQualitySubpixel; // Sub-pixel quality
    
    // Pick the biggest of the two offsets
    finalOffset = max(finalOffset, subPixelOffsetFinal);
    
    // Compute the final UV coordinates
    float2 finalUv = float2(coordf);
    if (horzSpan) {
        finalUv.y += finalOffset * stepLength;
    } else {
        finalUv.x += finalOffset * stepLength;
    }
    
    // Read the color at the new UV coordinates, and return it
    return sampleColor(view_idx, finalUv);
}

// ------------------------------------------------------------------------------------------------
[numThreads(32, 32, 1)]
[shader("compute")]
void main(uint3 idx : SV_DispatchThreadID)
{
    if (idx.x >= pushConst.viewWidth || idx.y >= pushConst.viewHeight || idx.z >= pushConst.totalViews) {
        return;
    }
    
    uint view_idx = idx.z;
    int2 coord = int2(idx.x, idx.y);
    
    float3 final_color;
    
    final_color = applyFXAA(view_idx, coord);
    
    uint32_t out_pixel_idx = view_idx * pushConst.viewWidth * pushConst.viewHeight +
                             idx.y * pushConst.viewWidth + idx.x;
    
    rgbBuffer[out_pixel_idx] = packRGB8(final_color);
}
