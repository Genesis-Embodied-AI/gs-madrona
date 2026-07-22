set(IMPORTER_SOURCES
    ${MADRONA_INC_DIR}/importer.hpp importer.cpp
    obj.hpp obj.cpp
    stb_read.cpp img.cpp
)

add_library(madrona_importer STATIC
    ${IMPORTER_SOURCES}
)

target_link_libraries(madrona_importer PRIVATE
    madrona_common
    meshoptimizer
    fast_float
    stb
)
