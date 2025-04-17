cmake_minimum_required(VERSION 3.20)
project(AudioML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

if(EXISTS "C:/Users/Karim/vcpkg/installed/x64-windows/share/taglib")
    set(TagLib_DIR "C:/Users/Karim/vcpkg/installed/x64-windows/share/taglib")
endif()

if(EXISTS "C:/Users/Karim/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/Karim/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

# Global dependencies (shared across all executables)
find_package(Threads REQUIRED)
find_package(TagLib)
add_executable(getAudioLength
        core/src_files/data/getAudioLength.cpp)
target_link_libraries(getAudioLength PRIVATE
        TagLib::TagLib
        Threads::Threads
)
if(EXISTS "C:/Users/Karim/vcpkg/installed/x64-windows/bin/tag.dll")
    add_custom_command(TARGET getAudioLength POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Users/Karim/vcpkg/installed/x64-windows/bin/tag.dll"
            $<TARGET_FILE_DIR:getAudioLength>
    )
endif()

add_executable(getBestThresholdFiles
        core/src_files/data/getBestThresholdFiles.cpp)
target_link_libraries(getBestThresholdFiles PRIVATE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(getAudioLength PROPERTIES FOLDER "core/data")
set_target_properties(getBestThresholdFiles PROPERTIES FOLDER "core/data")