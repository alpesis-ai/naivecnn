# ----------------------------------------------------------------------------

FILE ( GLOB MODEL_HEADERS . *.h )
FILE ( GLOB MODEL_SOURCES . *.cpp )

ADD_LIBRARY ( ncnn_model SHARED ${MODEL_SOURCES} )
TARGET_LINK_LIBRARIES ( ncnn_model ncnn_maths )
