set(SIMULATOR_SRCS
    types.hpp sim.hpp sim.inl sim.cpp geometry.hpp
)

add_library(madgs_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(madgs_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

option(LOAD_VULKAN "Load Vulkan libraries" ON)

add_library(madgs_mgr SHARED
    mgr.hpp mgr.cpp
)

if (NOT LOAD_VULKAN)
    message(INFO "Disabling vulkan!")
    target_compile_definitions(madgs_mgr
        PRIVATE
            -DMGR_DISABLE_VULKAN)
else ()
    message(INFO "Vulkan was enabled")
endif ()

target_link_libraries(madgs_mgr
    PRIVATE
        madrona_python_utils
        madgs_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
        madrona_render_asset_processor
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madgs_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madgs_mgr PRIVATE
        madgs_gpu_srcs
        madrona_mw_gpu
        madrona_cuda
    )
endif ()

target_compile_definitions(madgs_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(_gs_madrona_batch_renderer
    bindings.cpp
)

target_link_libraries(_gs_madrona_batch_renderer PRIVATE
    madgs_mgr
)

if (TARGET madrona_viz)
    madrona_python_module(_gs_madrona_visualizer
        viewer.cpp
    )

    target_link_libraries(_gs_madrona_visualizer PRIVATE 
        madrona_viz madgs_mgr
    )

    if (TARGET madrona_cuda)
        target_link_libraries(_gs_madrona_visualizer PRIVATE
            madrona_cuda
        )
    endif()
endif()
