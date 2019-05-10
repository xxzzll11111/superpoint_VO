# cmake needs this line
cmake_minimum_required(VERSION 2.8)
project( vo  )

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -mcpu=cortex-a53")

find_package( OpenCV REQUIRED )
message(${OpenCV_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# find_package(Eigen3)
# include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ./include )
#add_subdirectory( src )
add_executable( main src/visodo.cpp )


file(GLOB_RECURSE DPU_CODES ${CMAKE_SOURCE_DIR}/model/dpu_superpoint.elf)
target_link_libraries(main ${DPU_CODES} ${OpenCV_LIBS})
target_link_libraries(main pthread hineon n2cube dputils)
