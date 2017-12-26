# ----------------------------------------------------------------------------
# modules

ADD_SUBDIRECTORY ( tensor )
ADD_SUBDIRECTORY ( maths )
ADD_SUBDIRECTORY ( layers )
ADD_SUBDIRECTORY ( model )

# ----------------------------------------------------------------------------


FILE ( GLOB NCNN_HEADERS . *.h )
FILE ( GLOB NCNN_SOURCES . *.cpp )

ADD_LIBRARY ( ncnn STATIC ${NCNN_SOURCES} )
TARGET_LINK_LIBRARIES ( ncnn 
    ncnn_maths
    ncnn_model
)
