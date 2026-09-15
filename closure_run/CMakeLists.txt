cmake_minimum_required(VERSION 3.21)
project(DL_LES_Training CXX)
set(CMAKE_CXX_STANDARD 17)

####- Fix this path to your libtorch installation -####
set(CMAKE_PREFIX_PATH "/Users/raj/WORK_RAJ/gits/nekRS/3rd_party/libtorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(MPI REQUIRED)


#Training driver
add_executable(train_driver main_training.cpp)
target_link_libraries(train_driver "${TORCH_LIBRARIES}" MPI::MPI_CXX)
target_include_directories(train_driver PUBLIC "${TORCH_INCLUDE_DIRS}" ${MPI_CXX_INCLUDE_DIRS})
set_property(TARGET train_driver PROPERTY CXX_STANDARD 17)




# Full gradient check driver
add_executable(grad_check_debug grad_check_debug.cpp)
target_link_libraries(grad_check_debug "${TORCH_LIBRARIES}" MPI::MPI_CXX)
target_include_directories(grad_check_debug PUBLIC "${TORCH_INCLUDE_DIRS}" ${MPI_CXX_INCLUDE_DIRS})
set_property(TARGET grad_check_debug PROPERTY CXX_STANDARD 17)


