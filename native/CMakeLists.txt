cmake_minimum_required(VERSION 3.14)
project(logitlens-native CXX C)

set(CMAKE_CXX_STANDARD 17)

set(LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/wllama/llama.cpp)
set(WLLAMA_CPP ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/wllama/cpp)

set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_ROOT} llama.cpp-build)

add_executable(lens-cli lens-cli.cpp)
target_include_directories(lens-cli PRIVATE ${WLLAMA_CPP} ${LLAMA_ROOT}/include)
target_link_libraries(lens-cli PRIVATE llama ggml)
