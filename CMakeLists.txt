cmake_minimum_required(VERSION 3.10)

project(mechanize CXX)

add_executable(mechanize src/main.cpp)

find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(mechanize PRIVATE argparse::argparse
                                        nlohmann_json::nlohmann_json)
