cmake_minimum_required(VERSION 3.15...4.3)
project(studious LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE STUDIOUS_SOURCES CONFIGURE_DEPENDS
    lib/core/*.cpp
    lib/cpu/*.cpp
    lib/bindings/*.cpp
)

if(APPLE)
    file(GLOB_RECURSE METAL_SOURCES CONFIGURE_DEPENDS lib/metal/*.cpp)
    list(REMOVE_ITEM METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lib/metal/devices_.cpp)
    list(APPEND STUDIOUS_SOURCES ${METAL_SOURCES})
    file(GLOB_RECURSE METAL_KERNEL_SOURCES CONFIGURE_DEPENDS lib/metal/kernels/*.metal)
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

    set(METALLIB_FILENAME "_kernels.metallib")
    target_compile_definitions(metal_cpp_headers INTERFACE METALLIB_FILENAME="${METALLIB_FILENAME}")

    target_include_directories(metal_cpp_headers SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})
    target_link_libraries(metal_cpp_headers INTERFACE "-framework Foundation" "-framework Metal")
    target_link_libraries(_lib PRIVATE metal_cpp_headers)

    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/studious_kernels.air)
    set(METAL_LIB ${CMAKE_SOURCE_DIR}/studious/${METALLIB_FILENAME})

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal
            -c ${METAL_KERNEL_SOURCES}
            -o ${METAL_AIR}
        DEPENDS ${METAL_KERNEL_SOURCES}
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/studious
        COMMAND xcrun -sdk macosx metallib
            ${METAL_AIR}
            -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        VERBATIM
    )

    add_custom_target(studious_metallib ALL
        DEPENDS ${METAL_LIB}
    )

    add_dependencies(_lib studious_metallib)
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