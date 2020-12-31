cmake_minimum_required(VERSION 3.2)

project(improc)

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")

include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

add_library(improc MODULE ${SRC})
target_link_libraries(improc ${Python3_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

if(SKBUILD)
python_extension_module(improc)
endif()
install(TARGETS improc LIBRARY DESTINATION "niftypet/${CMAKE_PROJECT_NAME}/prc")
