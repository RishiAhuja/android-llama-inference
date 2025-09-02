cmake_minimum_required(VERSION 3.22.1)
project("gemma_app")

# FIX: Set compiler flags for ARM64 to enable NEON FP16 instructions.
# This is required to compile certain optimized parts of llama.cpp.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    message(STATUS "ARM64-v8a build detected, enabling FP16 NEON support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+fp16")
endif()

# Add llama.cpp as a subdirectory. It will inherit the flags set above.
add_subdirectory(llama.cpp)

# Define our native library that bridges C++ to Dart.
add_library(native-lib SHARED native-lib.cpp)

# Find the log library required for Android logging
find_library(log-lib log)

# Link our native library against the compiled llama library and Android log library.
target_link_libraries(native-lib 
    llama 
    ${log-lib}
)

