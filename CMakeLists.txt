cmake_minimum_required(VERSION 3.16)
project(AudioML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_library(ESSENTIA_LIB essentia)
find_path(ESSENTIA_INCLUDE essentia/essentia.h)

find_package(dlib REQUIRED)

# Collect all source files
file(GLOB SOURCES "core/*.cpp")
file(GLOB EXTRACTORS "core/feature_extractors/*.cpp")
file(GLOB MODELS "core/model/*.cpp")
file(GLOB CLEANER "core/cleaning/*.cpp")
file(GLOB AUDIO "core/audio/*.cpp")
file(GLOB PREPROCESSOR "core/preprocessing/*.cpp")

# IMPORTANT: Collect tools but PROPERLY exclude clean_dataset.cpp
file(GLOB TOOLS_ALL "tools/*.cpp")
set(TOOLS "")
foreach(TOOL_FILE ${TOOLS_ALL})
    if(NOT ${TOOL_FILE} MATCHES "clean_dataset.cpp" AND NOT ${TOOL_FILE} MATCHES "process_dataset.cpp")
        list(APPEND TOOLS ${TOOL_FILE})
    endif()
endforeach()

# Header files
file(GLOB HEADERS_INCLUDE "include/*.hpp")
file(GLOB HEADERS_AUDIO "core/audio/*.hpp")
file(GLOB HEADERS_CLEANING "core/cleaning/*.hpp")
file(GLOB HEADERS_TOOLS "tools/*.hpp")
file(GLOB HEADERS_PREPROCESSOR "core/preprocessing/*.hpp")

# Main executable - WITHOUT clean_dataset.cpp
add_executable(main 
    ${SOURCES} 
    ${EXTRACTORS} 
    ${MODELS} 
    ${CLEANER} 
    ${AUDIO} 
    ${TOOLS}
    ${HEADERS_INCLUDE}
    ${HEADERS_AUDIO}
    ${HEADERS_CLEANING}
    ${HEADERS_TOOLS}
)

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ESSENTIA_INCLUDE}
    ${dlib_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    ${ESSENTIA_LIB}
    ${dlib_LIBRARIES}
    stdc++fs  # Add filesystem library
)

# Add clean_dataset as a separate executable
add_executable(clean_dataset 
    tools/clean_dataset.cpp
    ${CLEANER}
    ${TOOLS}
    ${AUDIO} 
    ${HEADERS_AUDIO}
    ${HEADERS_CLEANING}
    ${HEADERS_TOOLS}
)

# Include directories for clean_dataset
target_include_directories(clean_dataset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ESSENTIA_INCLUDE}
    ${dlib_INCLUDE_DIRS}
)

# Link libraries for clean_dataset
target_link_libraries(clean_dataset PRIVATE
    ${ESSENTIA_LIB}
    ${dlib_LIBRARIES}
    stdc++fs  # Add filesystem library
)


# Add clean_dataset as a separate executable
add_executable(process_dataset 
    tools/process_dataset.cpp
    ${PREPROCESSOR}
    ${TOOLS}
    ${HEADERS_PREPROCESSOR} 
    ${HEADERS_TOOLS}
)

# Include directories for clean_dataset
target_include_directories(process_dataset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ESSENTIA_INCLUDE}
    ${dlib_INCLUDE_DIRS}
)

# Link libraries for clean_dataset
target_link_libraries(process_dataset PRIVATE
    ${ESSENTIA_LIB}
    ${dlib_LIBRARIES}
    stdc++fs  # Add filesystem library
)