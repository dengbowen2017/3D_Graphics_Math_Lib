cmake_minimum_required (VERSION 3.8)

project ("GMathLib")

add_executable(${PROJECT_NAME} "src/examples.cpp" "include/GMath.h")

target_include_directories(${PROJECT_NAME} PUBLIC "include")
