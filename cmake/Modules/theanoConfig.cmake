INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_THEANO theano)

FIND_PATH(
    THEANO_INCLUDE_DIRS
    NAMES theano/api.h
    HINTS $ENV{THEANO_DIR}/include
        ${PC_THEANO_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    THEANO_LIBRARIES
    NAMES gnuradio-theano
    HINTS $ENV{THEANO_DIR}/lib
        ${PC_THEANO_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(THEANO DEFAULT_MSG THEANO_LIBRARIES THEANO_INCLUDE_DIRS)
MARK_AS_ADVANCED(THEANO_LIBRARIES THEANO_INCLUDE_DIRS)

