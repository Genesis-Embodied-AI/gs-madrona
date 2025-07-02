add_library(madrona_render_shader_compiler SHARED
    ${MADRONA_INC_DIR}/render/shader_compiler.hpp compiler.cpp
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(madrona_render_shader_compiler PRIVATE
        SHADER_DEBUG
    )
    message(STATUS "Building with shader debug")
endif()

target_link_libraries(madrona_render_shader_compiler PRIVATE
    madrona_common
    madrona_dxcompiler
    spv_reflect
)
