cmake_minimum_required(VERSION 3.15...4.3)
project(studious LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_SUPPRESS_WARNINGS core/tensor.cpp core/bindings.cpp)

target_compile_features(_core PRIVATE cxx_std_20)
target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -g -O0)

set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/studious)