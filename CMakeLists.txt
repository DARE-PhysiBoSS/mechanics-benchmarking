cmake_minimum_required(VERSION 3.10)

project(mechanize CXX)

file(GLOB_RECURSE src_files "src/*")
add_executable(mechanize ${src_files})

find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(hwy CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

target_link_libraries(
  mechanize PRIVATE argparse::argparse nlohmann_json::nlohmann_json hwy::hwy
                    OpenMP::OpenMP_CXX)
