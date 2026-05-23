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

nanobind_add_module(_lib NB_SUPPRESS_WARNINGS lib/core/array/array.cpp lib/core/array/numpy.cpp lib/core/bindings.cpp lib/core/buffer.cpp)

target_compile_features(_lib PRIVATE cxx_std_20)
target_compile_options(_lib PRIVATE -Wall -Wextra -Wpedantic -Wconversion -g -O0)

set_target_properties(_lib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/studious)

nanobind_add_stub(
    _lib_stub
    MODULE studious._lib
    OUTPUT ${CMAKE_SOURCE_DIR}/studious/_lib.pyi
    PYTHON_PATH ${CMAKE_SOURCE_DIR}
    DEPENDS _lib
    MARKER_FILE ${CMAKE_SOURCE_DIR}/studious/py.typed
)