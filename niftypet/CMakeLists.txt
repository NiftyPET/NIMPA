cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
if("${NIMPA_BUILD_VERSION}" STREQUAL "")
  set(NIMPA_BUILD_VERSION 2 CACHE STRING "version" FORCE)
endif()
project(nimpa LANGUAGES C CXX CUDA VERSION "${NIMPA_BUILD_VERSION}")

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)  # <PackageName>_ROOT hints for find_package
cmake_policy(SET CMP0104 NEW)  # CMAKE_CUDA_ARCHITECTURES
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import cuvec; print(cuvec.include_path)"
  OUTPUT_VARIABLE CUVEC_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${CUVEC_INCLUDE_DIRS}" STREQUAL "")
  message(WARNING "Could not find cuvec includes")
else()
  message(STATUS "Found cuvec includes: ${CUVEC_INCLUDE_DIRS}")
endif()
if(SKBUILD)
  find_package(PythonExtensions REQUIRED)
  set(LIB_TYPE "MODULE")
else()
  set(LIB_TYPE "SHARED")
endif()
cmake_policy(POP)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(CUVEC_DEBUG "Print out CUDA malloc & free operations" OFF)
if(CUVEC_DEBUG)
  add_compile_definitions(CUVEC_DEBUG)
endif(CUVEC_DEBUG)
message(STATUS "cuvec debugging: ${CUVEC_DEBUG}")
if("${NIMPA_CU_THREADS}" STREQUAL "")
  set(NIMPA_CU_THREADS 1024 CACHE STRING
    "Maximum number of CUDA threads per block (should be less than cudaDeviceProp::maxThreadsDim)" FORCE)
endif()
add_compile_definitions(NIMPA_CU_THREADS=${NIMPA_CU_THREADS})
if("${NIMPA_KERNEL_RADIUS}" STREQUAL "")
  set(NIMPA_KERNEL_RADIUS 8 CACHE STRING "Maximum kernel half-width (excluding central voxel)" FORCE)
endif()
add_compile_definitions(NIMPA_KERNEL_RADIUS=${NIMPA_KERNEL_RADIUS})

add_subdirectory(nimpa)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION niftypet/${CMAKE_PROJECT_NAME}/cmake)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${CMAKE_PROJECT_VERSION}" COMPATIBILITY AnyNewerVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/NiftyPET${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION niftypet/${CMAKE_PROJECT_NAME}/cmake)
