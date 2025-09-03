cmake_minimum_required(VERSION 3.22.1)
project("gemma_app")

# FIX: Set compiler flags for ARM64 to enable NEON FP16 instructions.
# This is required to compile certain optimized parts of llama.cpp.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    message(STATUS "ARM64-v8a build detected, enabling FP16 NEON support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+fp16")
endif()

# --- GPU OPTIMIZATION ---
# This is the key line. It tells llama.cpp's build system to include the Vulkan backend,
# which allows it to use the phone's GPU for acceleration.
# This MUST come BEFORE add_subdirectory(llama.cpp).
set(LLAMA_VULKAN ON CACHE BOOL "Enable Vulkan support")
message(STATUS "Vulkan GPU support for llama.cpp has been enabled.")

# Enable other GPU-related optimizations
set(LLAMA_NATIVE OFF CACHE BOOL "Disable native optimizations for broader compatibility")
set(LLAMA_LTO OFF CACHE BOOL "Disable LTO for faster compilation")

# Add llama.cpp as a subdirectory. It will now be built with Vulkan support.
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

