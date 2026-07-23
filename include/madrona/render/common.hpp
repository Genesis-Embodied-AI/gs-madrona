#pragma once

#include <madrona/math.hpp>

namespace madrona::render {

struct APILib {};
struct APIBackend {};
struct GPUDevice {};

inline float srgbToLinear(float srgb);
inline math::Vector4 srgb8ToFloat(uint8_t r, uint8_t g, uint8_t b);

}

#include "common.inl"
