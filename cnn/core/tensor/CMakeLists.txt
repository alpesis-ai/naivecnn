# ----------------------------------------------------------------------------
# Sub-directories

ADD_SUBDIRECTORY ( tests )

# ----------------------------------------------------------------------------

FILE ( GLOB TENSOR_HEADERS . *.h )
# FILE ( GLOB TENSOR_SOURCES . *.cpp )

# ADD_LIBRARY ( ncnn_tensor SHARED ${TENSOR_SOURCES} )
# TARGET_LINK_LIBRARIES ( ncnn_tensor )
