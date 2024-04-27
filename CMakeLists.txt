cmake_minimum_required(VERSION 3.0.0)
project(iLQR_CppAD)
set(CMAKE_CXX_STANDARD 11)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) # set the output path of the executable
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) # set the output path of the library

include_directories(${PROJECT_SOURCE_DIR}/include) # include the header files

link_directories(${PROJECT_SOURCE_DIR}/lib) # link the executable with the library

add_library(iLQR_CppAD SHARED 
    ${PROJECT_SOURCE_DIR}/src/iLQR_CppAD.cpp) # add the shared library

add_executable(Mujoco_Acrobot
    ${PROJECT_SOURCE_DIR}/src/Mujoco_Acrobot.cpp) # add the executable

target_link_libraries(Mujoco_Acrobot iLQR_CppAD) # link the executable with the library