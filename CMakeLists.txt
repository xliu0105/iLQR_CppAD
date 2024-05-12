cmake_minimum_required(VERSION 3.0.0)
project(iLQR_CppAD)
set(CMAKE_CXX_STANDARD 11)
find_package(glfw3 REQUIRED)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) # set the output path of the executable
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) # set the output path of the library
set(MUJOCO_FOLDER /home/liu_xu/.mujoco/mujoco-3.0.0/)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${MUJOCO_FOLDER}/include
                    ${MUJOCO_FOLDER}/include/mujoco
                    ${glfw3_INCLUDE_DIR}) # include the header files

link_directories(${PROJECT_SOURCE_DIR}/lib ${MUJOCO_FOLDER}/lib) # link the executable with the library

add_library(iLQR_CppAD SHARED 
    ${PROJECT_SOURCE_DIR}/src/iLQR_CppAD.cpp) # add the shared library

add_executable(Acrobot_iLQR
    ${PROJECT_SOURCE_DIR}/src/Acrobot_iLQR.cpp) # add the executable

add_executable(Mujoco_visualize
    ${PROJECT_SOURCE_DIR}/src/Mujoco_visualize.cpp) # add the executable

target_link_libraries(Acrobot_iLQR iLQR_CppAD mujoco glfw) # link the executable with the library
target_link_libraries(Mujoco_visualize mujoco glfw) # link the executable with the library