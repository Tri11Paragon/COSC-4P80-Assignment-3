cmake_minimum_required(VERSION 3.25)
project(COSC-4P80-Assignment-3 VERSION 0.0.24)
include(FetchContent)

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)

FetchContent_Declare(implot
        GIT_REPOSITORY https://github.com/epezent/implot
        GIT_TAG 419a8a0f5fcb77e1e7c19ab540441686bfe21bca
        FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(implot)

add_subdirectory(lib/blt-with-graphics)

include_directories(include/)
include_directories(${implot_SOURCE_DIR})
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB IM_PLOT_FILES "${implot_SOURCE_DIR}/*.cpp")

add_executable(COSC-4P80-Assignment-3 ${PROJECT_BUILD_FILES} ${IM_PLOT_FILES})

target_compile_options(COSC-4P80-Assignment-3 PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)
target_link_options(COSC-4P80-Assignment-3 PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)

target_link_libraries(COSC-4P80-Assignment-3 PRIVATE BLT_WITH_GRAPHICS)

if (${ENABLE_ADDRSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-3 PRIVATE -fsanitize=address)
    target_link_options(COSC-4P80-Assignment-3 PRIVATE -fsanitize=address)
endif ()

if (${ENABLE_UBSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-3 PRIVATE -fsanitize=undefined)
    target_link_options(COSC-4P80-Assignment-3 PRIVATE -fsanitize=undefined)
endif ()

if (${ENABLE_TSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-3 PRIVATE -fsanitize=thread)
    target_link_options(COSC-4P80-Assignment-3 PRIVATE -fsanitize=thread)
endif ()

if (EMSCRIPTEN)
    message("Linking Emscripten")
    set(BLT_PRELOAD_PATH ../data)
    include(lib/blt-with-graphics/cmake/link_flags.cmake)
endif ()