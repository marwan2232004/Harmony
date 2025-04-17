cmake_minimum_required(VERSION 3.16)
project(AudioML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_library(ESSENTIA_LIB essentia)
find_path(ESSENTIA_INCLUDE essentia/essentia.h)

file(GLOB SOURCES "core/*.cpp")
file(GLOB EXTRACTORS "core/feature_extractors/*.cpp")

add_executable(main ${SOURCES} ${EXTRACTORS})

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ESSENTIA_INCLUDE}
)

target_link_libraries(main PRIVATE
    ${ESSENTIA_LIB}
)
