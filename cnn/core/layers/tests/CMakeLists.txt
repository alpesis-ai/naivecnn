# ----------------------------------------------------------------------------

FILE ( GLOB LAYERS_PADDING_HEADERS . test_padding.h )
FILE ( GLOB LAYERS_PADDING_SOURCES . test_padding.cpp )

ADD_EXECUTABLE ( test_padding ${LAYERS_PADDING_SOURCES} )
TARGET_LINK_LIBRARIES ( test_padding libgtest )

ADD_TEST ( NAME test_padding COMMAND test_padding )
SET ( UNIT_TEST test_padding )
ADD_CUSTOM_COMMAND (
    TARGET ${UNIT_TEST}
    COMMENT "Run padding tests"
    POST_BUILD
    COMMAND ${UNIT_TEST}
)
