cmake_minimum_required(VERSION 3.10)
#set(CMAKE_CXX_STANDARD 14)
project (quadsim LANGUAGES CXX)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Include OpenCV header files
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT)


link_directories(${OPENGL_LIBRARIES_DIRS} ${OPENGL_LIBRARIES_DIRS})
include_directories(${CMAKE_SOURCE_DIR} ${OPENGL_INCLUDE_DIRS})

add_subdirectory(pybind11)

pybind11_add_module(quadsim quadsim.cpp)
target_link_libraries(quadsim 
  PRIVATE ${GLUT_LIBRARY} ${OPENGL_LIBRARY}
)

# add_executable(main main.cpp)
# target_link_libraries(main 
#   PRIVATE OpenGL::OpenGL
#   PRIVATE GLUT::GLUT
# )
