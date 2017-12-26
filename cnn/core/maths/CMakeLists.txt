# ----------------------------------------------------------------------------

FILE ( GLOB MATHS_HEADERS . *.h )
FILE ( GLOB MATHS_SOURCES . *.cpp )

ADD_LIBRARY ( ncnn_maths SHARED ${MATHS_SOURCES} )
TARGET_LINK_LIBRARIES ( ncnn_maths )
