cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
cmake_policy(VERSION 3.26)

include(cmake/madrona_init.cmake)

project(madrona LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)

if (MADRONA_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
