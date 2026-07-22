
include(madrona-deps/cmake/sys-detect.cmake)
set(MADRONA_LINUX ${MADRONA_LINUX} PARENT_SCOPE)
set(MADRONA_MACOS ${MADRONA_MACOS} PARENT_SCOPE)
set(MADRONA_WINDOWS ${MADRONA_WINDOWS} PARENT_SCOPE)
set(MADRONA_IOS ${MADRONA_IOS} PARENT_SCOPE)
set(MADRONA_ANDROID ${MADRONA_ANDROID} PARENT_SCOPE)
set(MADRONA_X64 ${MADRONA_X64} PARENT_SCOPE)
set(MADRONA_ARM ${MADRONA_ARM} PARENT_SCOPE)

unset(CMAKE_CXX_VISIBILITY_PRESET)

if (NOT WIN32) #FIX
    add_subdirectory(madrona-toolchain)
else()
    add_library(madrona_libcxx INTERFACE)
endif()

add_library(madrona_noexceptrtti INTERFACE)
if (FRONTEND_GCC OR FRONTEND_CLANG)
    target_compile_options(madrona_noexceptrtti INTERFACE
        -fno-exceptions -fno-rtti)
elseif (FRONTEND_MSVC)
    target_compile_options(madrona_noexceptrtti INTERFACE
        /GR-
    )
else ()
    message(FATAL_ERROR "Unsupported compiler frontend")
endif ()

install(TARGETS madrona_noexceptrtti EXPORT madronaExternal)
install(EXPORT madronaExternal DESTINATION ${CMAKE_INSTALL_PREFIX})

set(MADRONA_TOOLCHAIN_ROOT_OVERRIDE "${CMAKE_CURRENT_BINARY_DIR}/madrona-toolchain")
add_subdirectory(madrona-deps)

add_library(madrona_sys_defns INTERFACE)
if (MADRONA_LINUX)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_LINUX=(1)")
elseif (MADRONA_WINDOWS)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_WINDOWS=(1)")
elseif (MADRONA_MACOS)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_MACOS=(1)")
elseif (MADRONA_IOS)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_IOS=(1)")
elseif (MADRONA_ANDROID)
    target_compile_definitions(madrona_sys_defns INTERFACE "MADRONA_ANDROID=(1)")
endif ()

add_subdirectory(meshoptimizer EXCLUDE_FROM_ALL)
set_property(TARGET meshoptimizer PROPERTY
    POSITION_INDEPENDENT_CODE TRUE)

if (NOT WIN32) #FIX
    target_link_libraries(meshoptimizer PRIVATE
        madrona_libcxx madrona_noexceptrtti)
endif()

add_subdirectory(fast_float EXCLUDE_FROM_ALL)


if (Python_FOUND)
    add_subdirectory(nanobind EXCLUDE_FROM_ALL)
endif ()

add_library(spv_reflect STATIC
    SPIRV-Reflect/spirv_reflect.cpp SPIRV-Reflect/spirv_reflect.h
)
target_include_directories(spv_reflect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Reflect/
)
set_target_properties(spv_reflect PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(spv_reflect PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)

