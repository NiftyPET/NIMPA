cmake_minimum_required(VERSION 3.2)

project(nimpa)

set(PYTHON_PREFIX_PATH)
set(PYTHON_INCLUDE_DIRS)
set(PREINSTALL_NAME "_preinstall")

# find cuda, maybe set a minimum version
find_package(CUDA REQUIRED)

set(PYTHON_LIBRARIES)
find_library(PYTHON_LIBRARY
              NAMES python27 libpython2.7.so
              PATHS ${PYTHON_PREFIX_PATH}/libs/
              ${PYTHON_PREFIX_PATH}/lib/
            )

# find_package(PythonLibs 2.7 REQUIRED )
# find_package(PythonInterp 2.7 REQUIRED )

add_subdirectory(${CMAKE_PROJECT_NAME}/dinf)
add_subdirectory(${CMAKE_PROJECT_NAME}/prc)

set(CMAKE_BUILD_TYPE Release)

#configure_file(${CMAKE_SOURCE_DIR}/setup.py     ${CMAKE_SOURCE_DIR}/${PREINSTALL_NAME}/setup.py COPYONLY)
#configure_file(${CMAKE_SOURCE_DIR}/MANIFEST.in  ${CMAKE_SOURCE_DIR}/${PREINSTALL_NAME}/MANIFEST.in COPYONLY)
#configure_file(${CMAKE_SOURCE_DIR}/cudasetup.py  ${CMAKE_SOURCE_DIR}/${PREINSTALL_NAME}/cudasetup.py COPYONLY)
#configure_file(${CMAKE_SOURCE_DIR}/LICENCE      ${CMAKE_SOURCE_DIR}/${PREINSTALL_NAME}/LICENCE COPYONLY)
#configure_file(${CMAKE_SOURCE_DIR}/README.md    ${CMAKE_SOURCE_DIR}/${PREINSTALL_NAME}/README.md COPYONLY)

execute_process ( COMMAND python -c "import sys; print sys.prefix" OUTPUT_VARIABLE PYTHON_ENV_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX
    ${PYTHON_ENV_ROOT} CACHE PATH "Set the path of the anaconda environment to which you wish to install this package to" FORCE
    )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

#when the install project is run, call the python setup.py script to install the python to site-packages
#install(CODE "execute_process(COMMAND python setup.py install --prefix ${PYTHON_ENV_ROOT} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${PREINSTALL_NAME})")
