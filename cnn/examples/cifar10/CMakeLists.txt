# ----------------------------------------------------------------------------

FILE ( GLOB CIFAR10_HEADERS . *.hpp )
FILE ( GLOB CIFAR10_SOURCES . *.cpp )

ADD_EXECUTABLE ( cifar10_cnn ${CIFAR10_SOURCES} )
TARGET_LINK_LIBRARIES ( cifar10_cnn ncnn_model )

# run cifar10_cnn after building
# ADD_TEST ( NAME cifar10_cnn COMMAND cifar10_cnn )
# SET ( UNIT_TEST cifar10_cnn )
# ADD_CUSTOM_COMMAND (
#    TARGET ${UNIT_TEST}
#    COMMENT "Run cifar10_cnn"
#    POST_BUILD
#    COMMAND ${UNIT_TEST}
# )

