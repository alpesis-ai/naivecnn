CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

# ----------------------------------------------------------------------------
# Project settings

# Project name
PROJECT ( NCNN CXX C )

# project version
SET ( NCNN_MAJOR_VERSION 0 )
SET ( NCNN_MINOR_VERSION 0 )
SET ( NCNN_PATCH_VERSION 1 )
SET ( NCNN_VERSION ${NCNN_MAJOR_VERSION}.${NCNN_MINOR_VERSION}.${NCNN_PATCH_VERSION} )

# ----------------------------------------------------------------------------
# CMake settings

# cmake packages
SET ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" )

# include libraries
INCLUDE ( EnableCXX11 )
INCLUDE ( Eigen )

# ----------------------------------------------------------------------------
# Build options

# build options
OPTION ( WITH_TEST "Compile and run unittest for naive cnn" ON )

IF ( WITH_TEST )
    INCLUDE ( GTest )
    ENABLE_TESTING ()
ENDIF ()

# ----------------------------------------------------------------------------
# Project sources

# project root
SET ( PROJECT_ROOT ${CMAKE_SOURCE_DIR} )
INCLUDE_DIRECTORIES ( "${PROJECT_ROOT}" )

# project modules
ADD_SUBDIRECTORY ( core )
ADD_SUBDIRECTORY ( models )
ADD_SUBDIRECTORY ( examples )

# data
FILE ( COPY ${PROJECT_ROOT}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# ----------------------------------------------------------------------------
