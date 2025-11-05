cmake_minimum_required(VERSION 3.10)

project(mechanize CXX)

file(GLOB_RECURSE src_files "src/*")
add_executable(mechanize ${src_files})

find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(hwy CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

option(NO_PAPI "Disable PAPI performance measurements" OFF)

# Find and link PAPI
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI IMPORTED_TARGET papi)

if(NO_PAPI)
  target_compile_definitions(mechanize PRIVATE NO_PAPI)
endif()
target_link_libraries(mechanize PRIVATE PkgConfig::PAPI)

target_link_libraries(
  mechanize PRIVATE argparse::argparse nlohmann_json::nlohmann_json hwy::hwy
                    OpenMP::OpenMP_CXX)
