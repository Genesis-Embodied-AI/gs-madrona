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

if (SKBUILD)
    if (CUDA_REQUIRED_ARG)
        find_library(CUDA_NVJITLINK_LIBRARY nvJitLink
            PATHS ${CUDAToolkit_LIBRARY_DIR}
            REQUIRED)
        get_filename_component(CUDA_NVJITLINK_LIBRARY_REALPATH "${CUDA_NVJITLINK_LIBRARY}" REALPATH)
        get_filename_component(CUDA_NVJITLINK_NAME "${CUDA_NVJITLINK_LIBRARY}" NAME)
        install(FILES ${CUDA_NVJITLINK_LIBRARY_REALPATH} DESTINATION gs_madrona
            RENAME "${CUDA_NVJITLINK_NAME}.${CUDAToolkit_VERSION_MAJOR}")
    endif()

    if (CUDAToolkit_VERSION_MINOR LESS 5)
        set(CCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/cccl")
        install(DIRECTORY "${CCCL_DIR}/cub" "${CCCL_DIR}/thrust"
            DESTINATION gs_madrona/cccl)
        install(DIRECTORY "${CCCL_DIR}/libcudacxx/include"
            DESTINATION gs_madrona/cccl/libcudacxx)
    endif()

    find_package(embree REQUIRED)
    get_target_property(madrona_embree_shlib embree IMPORTED_LOCATION_RELEASE)
    install(FILES ${madrona_embree_shlib} DESTINATION gs_madrona)

    get_target_property(madrona_dxcompiler_shlib madrona_dxcompiler IMPORTED_LOCATION)
    install(FILES ${madrona_dxcompiler_shlib} DESTINATION gs_madrona)

    install(TARGETS madgs_mgr LIBRARY DESTINATION gs_madrona)
    install(TARGETS madrona_std_mem LIBRARY DESTINATION gs_madrona)
    install(TARGETS madrona_render_shader_compiler LIBRARY DESTINATION gs_madrona)

    file(GLOB_RECURSE MADRONA_PY_FILES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/gs_madrona/
        ${CMAKE_CURRENT_SOURCE_DIR}/gs_madrona/*.py
    )
    foreach(file ${MADRONA_PY_FILES})
        get_filename_component(file_dir ${file} DIRECTORY)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/gs_madrona/${file}"
            DESTINATION gs_madrona/${file_dir})
    endforeach()

    file(GLOB_RECURSE MADRONA_INC_FILES
        RELATIVE ${MADRONA_INC_DIR}
        ${MADRONA_INC_DIR}/*.hpp
        ${MADRONA_INC_DIR}/*.inl)
    foreach(file ${MADRONA_INC_FILES})
        get_filename_component(file_dir ${file} DIRECTORY)
        install(FILES "${MADRONA_INC_DIR}/${file}"
            DESTINATION gs_madrona/include/madrona/${file_dir})
    endforeach()

    set(MADRONA_DEVICE_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mw/device/include")
    file(GLOB_RECURSE MADRONA_DEVICE_INC_FILES
        RELATIVE ${MADRONA_DEVICE_INC_DIR}
        ${MADRONA_DEVICE_INC_DIR}/*)
    foreach(file ${MADRONA_DEVICE_INC_FILES})
        get_filename_component(file_dir ${file} DIRECTORY)
        install(FILES "${MADRONA_DEVICE_INC_DIR}/${file}"
            DESTINATION gs_madrona/src/mw/device/include/${file_dir})
    endforeach()

    file(GLOB_RECURSE MADRONA_SRC_FILES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../*.inl
        ${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp
    )
    file(GLOB_RECURSE MADRONA_DATA_FILES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../render/*
    )
    set(MADRONA_SRC_AND_DATA_FILES ${MADRONA_SRC_FILES} ${MADRONA_DATA_FILES})
    list(REMOVE_DUPLICATES MADRONA_SRC_AND_DATA_FILES)
    foreach(file ${MADRONA_SRC_AND_DATA_FILES})
        get_filename_component(file_dir ${file} DIRECTORY)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../${file}"
            DESTINATION gs_madrona/src/${file_dir})
    endforeach()
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../data DESTINATION gs_madrona)

    install(DIRECTORY ${CUDAToolkit_INCLUDE_DIRS} DESTINATION gs_madrona/cuda)
endif()
