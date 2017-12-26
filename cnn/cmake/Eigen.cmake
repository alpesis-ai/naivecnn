# We need thread support
find_package(Threads REQUIRED)

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install Eigen
ExternalProject_Add(
    eigen
    URL http://bitbucket.org/eigen/eigen/get/3.3.3.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/lib/eigen
    # Disable install step
    INSTALL_COMMAND ""
)

# Get Eigen source and binary directories from CMake project
# ExternalProject_Get_Property(eigen source_dir binary_dir)

# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${CMAKE_CURRENT_BINARY_DIR}/lib/eigen/src/eigen")
