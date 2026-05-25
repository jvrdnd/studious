cmake_minimum_required(VERSION 3.15...4.3)
project(studious LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE STUDIOUS_SOURCES CONFIGURE_DEPENDS
    lib/core/*.cpp
    lib/cpu/*.cpp
    lib/bindings.cpp
)

if(APPLE)
    file(GLOB_RECURSE METAL_SOURCES CONFIGURE_DEPENDS lib/metal/*.cpp)
    list(REMOVE_ITEM METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lib/metal/devices_.cpp)
    list(APPEND STUDIOUS_SOURCES ${METAL_SOURCES})
else()
    list(APPEND STUDIOUS_SOURCES lib/metal/devices_.cpp)
endif()

nanobind_add_module(_lib NB_SUPPRESS_WARNINGS ${STUDIOUS_SOURCES})

target_compile_features(_lib PRIVATE cxx_std_20)
target_compile_options(_lib PRIVATE -Wall -Wextra -Wpedantic -Wconversion -g -O3 -march=native)

if(APPLE)
    include(FetchContent)
    FetchContent_Declare(
        metal_cpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_26.4.zip
        URL_HASH SHA256=4e7587c0f351d782f1f3e7c67c649b40d0ba780c051b2f6729f00ddc61ada28c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(metal_cpp)

    add_library(metal_cpp_headers INTERFACE)

    target_include_directories(metal_cpp_headers SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})
    target_link_libraries(metal_cpp_headers INTERFACE "-framework Foundation" "-framework Metal")
    target_link_libraries(_lib PRIVATE metal_cpp_headers)
endif()

set_target_properties(_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/studious)

nanobind_add_stub(
    _lib_stub
    MODULE studious._lib
    OUTPUT ${CMAKE_SOURCE_DIR}/studious/_lib.pyi
    PYTHON_PATH ${CMAKE_SOURCE_DIR}
    DEPENDS _lib
    MARKER_FILE ${CMAKE_SOURCE_DIR}/studious/py.typed
)