# ----------------------------------------------------------------------------
# TensorTests

FILE ( GLOB TENSOR_TESTS_HEADERS . test_tensor.h )
FILE ( GLOB TENSOR_TESTS_SOURCES . test_tensor.cpp )

ADD_EXECUTABLE ( test_tensor ${TENSOR_TESTS_SOURCES} )
TARGET_LINK_LIBRARIES ( test_tensor libgtest )

ADD_TEST ( NAME test_tensor COMMAND test_tensor )
SET ( UNIT_TEST test_tensor )
ADD_CUSTOM_COMMAND (
    TARGET ${UNIT_TEST}
    COMMENT "Run tensor tests"
    POST_BUILD
    COMMAND ${UNIT_TEST} 
)
