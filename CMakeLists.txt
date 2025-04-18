cmake_minimum_required(VERSION 3.16)
project(AudioML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_library(ESSENTIA_LIB essentia)
find_path(ESSENTIA_INCLUDE essentia/essentia.h)

find_package(dlib REQUIRED)

file(GLOB SOURCES "core/*.cpp")
file(GLOB EXTRACTORS "core/src_files/feature_extractors/*.cpp")
file(GLOB MODELS "core/model/*.cpp")

add_executable(main ${SOURCES} ${EXTRACTORS} ${MODELS})

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ESSENTIA_INCLUDE}
    ${dlib_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    ${ESSENTIA_LIB}
    ${dlib_LIBRARIES}
)



# Global dependencies (shared across all executables)
find_package(Threads REQUIRED)

add_executable(getBestThresholdFiles
        core/src_files/data/getBestThresholdFiles.cpp)
target_link_libraries(getBestThresholdFiles PRIVATE)