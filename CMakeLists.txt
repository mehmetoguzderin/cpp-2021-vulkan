# cpp-2021-vulkan

# https://github.com/apache/arrow/blob/master/cpp/cmake_modules/Usevcpkg.cmake

message(STATUS "Using vcpkg to find dependencies")

if(DEFINED CMAKE_TOOLCHAIN_FILE)
  get_filename_component(_VCPKG_DOT_CMAKE "${CMAKE_TOOLCHAIN_FILE}" NAME)
  if(EXISTS "${CMAKE_TOOLCHAIN_FILE}" AND _VCPKG_DOT_CMAKE STREQUAL "vcpkg.cmake")
    get_filename_component(_VCPKG_BUILDSYSTEMS_DIR "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(VCPKG_ROOT "${_VCPKG_BUILDSYSTEMS_DIR}/../.." ABSOLUTE)
  else()
    message(FATAL_ERROR "vcpkg toolchain file not found at path specified in -DCMAKE_TOOLCHAIN_FILE"
    )
  endif()
else()
  if(DEFINED VCPKG_ROOT)
    find_program(_VCPKG_BIN vcpkg
                 PATHS "${VCPKG_ROOT}"
                 NO_DEFAULT_PATH)
    if(NOT _VCPKG_BIN)
      message(FATAL_ERROR "vcpkg not found in directory specified in -DVCPKG_ROOT")
    endif()
  elseif(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_ROOT $ENV{VCPKG_ROOT})
    find_program(_VCPKG_BIN vcpkg
                 PATHS "${VCPKG_ROOT}"
                 NO_DEFAULT_PATH)
    if(NOT _VCPKG_BIN)
      message(FATAL_ERROR "vcpkg not found in directory in environment variable VCPKG_ROOT"
      )
    endif()
  else()
    find_program(_VCPKG_BIN vcpkg)
    if(_VCPKG_BIN)
      get_filename_component(_VCPKG_REAL_BIN "${_VCPKG_BIN}" REALPATH)
      get_filename_component(VCPKG_ROOT "${_VCPKG_REAL_BIN}" DIRECTORY)
    else()
      if(CMAKE_HOST_WIN32)
        set(_VCPKG_PATH_TXT "$ENV{LOCALAPPDATA}/vcpkg/vcpkg.path.txt")
      else()
        set(_VCPKG_PATH_TXT "$ENV{HOME}/.vcpkg/vcpkg.path.txt")
      endif()
      if(EXISTS "${_VCPKG_PATH_TXT}")
        file(READ "${_VCPKG_PATH_TXT}" VCPKG_ROOT)
      else()
        message(FATAL_ERROR "vcpkg not found. Install vcpkg if not installed, "
                            "then run vcpkg integrate install or set environment variable VCPKG_ROOT."
        )
      endif()
      find_program(_VCPKG_BIN vcpkg
                   PATHS "${VCPKG_ROOT}"
                   NO_DEFAULT_PATH)
      if(NOT _VCPKG_BIN)
        message(FATAL_ERROR "vcpkg not found. Re-run vcpkg integrate install "
                            "or set environment variable VCPKG_ROOT.")
      endif()
    endif()
  endif()
  set(CMAKE_TOOLCHAIN_FILE
      "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE FILEPATH "Path to vcpkg CMake toolchain file")
endif()
message(STATUS "Using CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "Using VCPKG_ROOT: ${VCPKG_ROOT}")

# cpp-2021-vulkan
