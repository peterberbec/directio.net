cmake_minimum_required(VERSION 3.20)

project(DirectioNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------

set(SOURCES
    src/main.cpp
)

# ------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------

add_executable(directio.net
    ${SOURCES}
)

# ------------------------------------------------------------------
# Headers
# ------------------------------------------------------------------

target_include_directories(directio.net PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include_3rd_party
)

# ------------------------------------------------------------------
# Library search path
# ------------------------------------------------------------------

target_link_directories(directio.net PRIVATE
    ${PROJECT_SOURCE_DIR}/lib
)

# ------------------------------------------------------------------
# Llama / GGML
# ------------------------------------------------------------------

target_link_libraries(directio.net PRIVATE
    llama
    ggml
    ggml-base
    ggml-cpu
    sqlite3
)

# ------------------------------------------------------------------
# Platform-specific libraries
# ------------------------------------------------------------------

if(UNIX AND NOT APPLE)
    target_link_libraries(directio.net PRIVATE
        pthread
        dl
        m
    )
endif()

if(WIN32)
    target_link_libraries(directio.net PRIVATE
        ws2_32
    )
endif()

# ------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------

set_target_properties(directio.net PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/bin
)
