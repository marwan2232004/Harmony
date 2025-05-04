cmake_minimum_required(VERSION 3.16)
project(AudioML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# ────────────────────────────────────────────────────────────────────────────────
# Set output directories for binaries and libraries
# ────────────────────────────────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")

# ────────────────────────────────────────────────────────────────────────────────
# Find BLAS and LAPACK first since they're key dependencies
# ────────────────────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")

# Explicitly set library flags if needed
set(MATH_LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

# ────────────────────────────────────────────────────────────────────────────────
# Find external libraries: Essentia, Dlib, Armadillo, MLPACK, Boost, OpenMP
# ────────────────────────────────────────────────────────────────────────────────
find_library(ESSENTIA_LIB essentia)
find_path(ESSENTIA_INCLUDE essentia/essentia.h)

find_package(Armadillo REQUIRED)

find_package(dlib REQUIRED)

set(MLPACK_INCLUDE_DIR "/usr/local/include")

find_package(Boost REQUIRED COMPONENTS serialization system)
find_package(OpenMP REQUIRED)

# ────────────────────────────────────────────────────────────────────────────────
# Collect all source files
# ────────────────────────────────────────────────────────────────────────────────
file(GLOB SOURCES "core/*.cpp")
file(GLOB EXTRACTORS "core/feature_extractors/*.cpp")
file(GLOB STACKING "core/stacking/*.cpp")
file(GLOB MODELS "core/model/*.cpp")
file(GLOB CLEANER "core/cleaning/*.cpp")
file(GLOB AUDIO "core/audio/*.cpp")
file(GLOB PREPROCESSOR "core/preprocessing/*.cpp")
file(GLOB UTILS "utils/*.cpp")

# IMPORTANT: Collect tools but PROPERLY exclude clean_dataset.cpp
file(GLOB TOOLS_ALL "tools/*.cpp")
set(TOOLS "")
foreach(TOOL_FILE ${TOOLS_ALL})
    if(NOT ${TOOL_FILE} MATCHES "clean_dataset.cpp" AND NOT ${TOOL_FILE} MATCHES "process_dataset.cpp" AND NOT ${TOOL_FILE} MATCHES "extract_features.cpp" AND NOT ${TOOL_FILE} MATCHES "stacking.cpp" AND NOT ${TOOL_FILE} MATCHES "inference.cpp")
        list(APPEND TOOLS ${TOOL_FILE})
    endif()
endforeach()

# Header files
file(GLOB HEADERS_INCLUDE "include/*.h")
file(GLOB HEADERS_AUDIO "core/audio/*.hpp")
file(GLOB HEADERS_CLEANING "core/cleaning/*.hpp")
file(GLOB HEADERS_TOOLS "tools/*.hpp")
file(GLOB HEADERS_PREPROCESSOR "core/preprocessing/*.hpp")
file(GLOB HEADERS_STACKING "core/stacking/*.hpp")
file(GLOB HEADERS_UTILS "core/utils/*.hpp")

# ────────────────────────────────────────────────────────────────────────────────
# Main executable - WITHOUT clean_dataset.cpp
# ────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────────
# Add clean_dataset as a separate executable
# ────────────────────────────────────────────────────────────────────────────────
# add_executable(clean_dataset 
#     tools/clean_dataset.cpp
#     ${CLEANER}
#     ${TOOLS}
#     ${AUDIO} 
#     ${HEADERS_AUDIO}
#     ${HEADERS_CLEANING}
#     ${HEADERS_TOOLS}
#     ${HEADERS_UTILS}
# )

# # Include directories for clean_dataset
# target_include_directories(clean_dataset PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${ESSENTIA_INCLUDE}
#     ${dlib_INCLUDE_DIRS}
#     ${MLPACK_INCLUDE_DIR}
# )

# # Link libraries for clean_dataset
# target_link_libraries(clean_dataset PRIVATE
#     ${ESSENTIA_LIB}
#     ${dlib_LIBRARIES}
#     stdc++fs  # Add filesystem library
# )


# # ────────────────────────────────────────────────────────────────────────────────
# # Add process_dataset as a separate executable
# # ────────────────────────────────────────────────────────────────────────────────
# add_executable(process_dataset 
#     tools/process_dataset.cpp
#     ${PREPROCESSOR}
#     ${TOOLS}
#     ${UTILS}
#     ${HEADERS_PREPROCESSOR} 
#     ${HEADERS_TOOLS}
#     ${HEADERS_UTILS}
# )


# target_compile_options(process_dataset PRIVATE
#     ${OpenMP_CXX_FLAGS}
# )

# # Include directories for process_dataset
# target_include_directories(process_dataset PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${ESSENTIA_INCLUDE}
#     ${OpenMP_CXX_INCLUDE_DIRS}
#     ${dlib_INCLUDE_DIRS}
#     ${MLPACK_INCLUDE_DIR}
# )

# # Link libraries for process_dataset
# target_link_libraries(process_dataset PRIVATE
#     OpenMP::OpenMP_CXX
#     ${ESSENTIA_LIB}
#     ${dlib_LIBRARIES}
#     ${OpenMP_CXX_LIBRARIES}
#     stdc++fs  # Add filesystem library
# )

# # ────────────────────────────────────────────────────────────────────────────────
# # Add extract_features as a separate executable
# # ────────────────────────────────────────────────────────────────────────────────
# add_executable(extract_features 
#     tools/extract_features.cpp
#     ${EXTRACTORS}
#     ${TOOLS}
#     ${UTILS}
#     ${HEADERS_INCLUDE}
#     ${HEADERS_TOOLS}
#     ${HEADERS_UTILS}
# )

# # Include directories for extract_features
# target_include_directories(extract_features PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${ESSENTIA_INCLUDE}
#     ${dlib_INCLUDE_DIRS}
#     ${MLPACK_INCLUDE_DIR}
# )

# # Link libraries for extract_features
# target_link_libraries(extract_features PRIVATE
#     ${ESSENTIA_LIB}
#     ${dlib_LIBRARIES}
#     stdc++fs  # Add filesystem library
# )

# # ────────────────────────────────────────────────────────────────────────────────
# # Add stacking as a separate executable
# # ────────────────────────────────────────────────────────────────────────────────
# add_executable(stacking 
#     tools/stacking.cpp
#     ${STACKING}
#     ${MODELS}
#     ${TOOLS}
#     ${UTILS}
#     ${HEADERS_INCLUDE}
#     ${HEADERS_STACKING}
#     ${HEADERS_TOOLS}
#     ${HEADERS_UTILS}
# )

# target_compile_options(stacking PRIVATE
#     ${OpenMP_CXX_FLAGS}
# )

# # Include directories for stacking
# target_include_directories(stacking PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${ESSENTIA_INCLUDE}
#     ${dlib_INCLUDE_DIRS}
#     ${OpenMP_CXX_INCLUDE_DIRS}
#     ${MLPACK_INCLUDE_DIR}
#     ${Armadillo_INCLUDE_DIRS}
# )

# # Find BLAS and LAPACK packages more explicitly
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)

# # Add these variables for direct linking
# set(BLAS_LIB "-lblas -lopenblas")
# set(LAPACK_LIB "-llapack")

# # Make the BLAS_LIBRARIES variable more explicit if needed
# if(NOT BLAS_LIBRARIES)
#     set(BLAS_LIBRARIES ${BLAS_LIB})
# endif()

# # Link libraries for stacking
# target_link_libraries(stacking
#     OpenMP::OpenMP_CXX
#     ${ESSENTIA_LIB}
#     ${dlib_LIBRARIES}
#     ${Boost_LIBRARIES} 
#     stdc++fs
#     # The ordering of these libraries is critical for resolving dependencies
#     ${Armadillo_LIBRARIES}
#     ${LAPACK_LIBRARIES}
#     ${BLAS_LIBRARIES}
#     ${OpenMP_CXX_LIBRARIES}
#     # Explicit linking as a fallback
#     -larmadillo
#     -llapack 
#     -lopenblas
#     -lblas
# )

# ────────────────────────────────────────────────────────────────────────────────
# Add inference as a separate executable
# ────────────────────────────────────────────────────────────────────────────────
add_executable(inference
    tools/inference.cpp 
    ${EXTRACTORS}
    ${STACKING}
    ${PREPROCESSOR}
    ${MODELS}
    ${TOOLS}
    ${UTILS}
    ${HEADERS_INCLUDE}
    ${HEADERS_STACKING}
    ${HEADERS_PREPROCESSOR} 
    ${HEADERS_TOOLS}
    ${HEADERS_UTILS}
)

target_compile_options(inference PRIVATE
    ${OpenMP_CXX_FLAGS}
)

# Include directories for stacking
target_include_directories(inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ESSENTIA_INCLUDE}
    ${dlib_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
    ${MLPACK_INCLUDE_DIR}
    ${Armadillo_INCLUDE_DIRS}
)

# Link libraries for stacking
target_link_libraries(inference
    OpenMP::OpenMP_CXX
    ${ESSENTIA_LIB}
    ${dlib_LIBRARIES}
    ${Boost_LIBRARIES} 
    stdc++fs
    # The ordering of these libraries is critical for resolving dependencies
    ${Armadillo_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    # Explicit linking as a fallback
    -larmadillo
    -llapack 
    -lopenblas
    -lblas
)

# ────────────────────────────────────────────────────────────────────────────────
# Add infer - a wrapper for inference with execution time measurement
# ────────────────────────────────────────────────────────────────────────────────
add_executable(infer
    infer.cpp
)

# Include directories for infer
target_include_directories(infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries for infer
target_link_libraries(infer
    stdc++fs  # For filesystem support
)

# Ensure infer depends on inference so it's built first
add_dependencies(infer inference)