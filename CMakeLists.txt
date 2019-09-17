cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

SET( CMAKE_EXPORT_COMPILE_COMMANDS ON )
aux_source_directory(./src SRC_LIST)

find_package( OpenCV 3 REQUIRED PATHS "/homes/thuxohl/tools/opencv_install")
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_executable( eval src/eval.cpp src/measures.cpp src/measures.h )
target_link_libraries( eval ${Boost_LIBRARIES} ${OpenCV_LIBS})

