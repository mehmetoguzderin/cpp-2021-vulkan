# cpp-2021-vulkan

# https://github.com/apache/arrow/blob/master/cpp/cmake_modules/Usevcpkg.cmake

message(STATUS "Using vcpkg to find dependencies")

macro(list_subdirs SUBDIRS DIR)
  file(GLOB children_
       RELATIVE ${DIR}
       ${DIR}/*)
  set(subdirs_ "")
  foreach(child_ ${children_})
    if(IS_DIRECTORY "${DIR}/${child_}")
      list(APPEND subdirs_ ${child_})
    endif()
  endforeach()
  set("${SUBDIRS}" ${subdirs_})
  unset(children_)
  unset(subdirs_)
endmacro()

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

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}")
endif()
if(NOT DEFINED VCPKG_MANIFEST_MODE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json")
  set(VCPKG_MANIFEST_MODE
      ON
      CACHE BOOL "Use vcpkg.json manifest")
  message(STATUS "vcpkg.json manifest found. Using VCPKG_MANIFEST_MODE: ON")
endif()
set(_INST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed") # try here first
set(_INST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed") # try here second
set(_INST_VCPKG_ROOT "${VCPKG_ROOT}/installed")
foreach(_INST_DIR IN LISTS _INST_BUILD_DIR _INST_SOURCE_DIR _INST_VCPKG_ROOT "notfound")
  if(_INST_DIR STREQUAL "notfound")
    message(FATAL_ERROR "vcpkg installed libraries directory not found. "
                        "Install packages with vcpkg before executing cmake.")
  elseif(NOT EXISTS "${_INST_DIR}")
    continue()
  elseif((_INST_DIR STREQUAL _INST_BUILD_DIR OR _INST_DIR STREQUAL _INST_SOURCE_DIR)
         AND NOT VCPKG_MANIFEST_MODE)
    message(STATUS "Skipped looking for installed packages in ${_INST_DIR} "
                   "because -DVCPKG_MANIFEST_MODE=OFF")
    continue()
  else()
    message(STATUS "Looking for installed packages in ${_INST_DIR}")
  endif()
  if(DEFINED VCPKG_TARGET_TRIPLET)
    if(EXISTS "${_INST_DIR}/${VCPKG_TARGET_TRIPLET}")
      set(_VCPKG_INSTALLED_DIR "${_INST_DIR}")
      break()
    endif()
  else()
    list_subdirs(_VCPKG_TRIPLET_SUBDIRS "${_INST_DIR}")
    list(REMOVE_ITEM _VCPKG_TRIPLET_SUBDIRS "vcpkg")
    list(LENGTH _VCPKG_TRIPLET_SUBDIRS _NUM_VCPKG_TRIPLET_SUBDIRS)
    if(_NUM_VCPKG_TRIPLET_SUBDIRS EQUAL 1)
      list(GET _VCPKG_TRIPLET_SUBDIRS 0 VCPKG_TARGET_TRIPLET)
      set(_VCPKG_INSTALLED_DIR "${_INST_DIR}")
      break()
    endif()
  endif()
endforeach()
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  message(FATAL_ERROR "Could not infer VCPKG_TARGET_TRIPLET. "
                      "Specify triplet with -DVCPKG_TARGET_TRIPLET.")
elseif(NOT DEFINED _VCPKG_INSTALLED_DIR)
  message(FATAL_ERROR "Could not find installed vcpkg packages for triplet ${VCPKG_TARGET_TRIPLET}. "
                      "Install packages with vcpkg before executing cmake.")
endif()

set(VCPKG_TARGET_TRIPLET
    "${VCPKG_TARGET_TRIPLET}"
    CACHE STRING "vcpkg triplet for the target environment")

if(NOT DEFINED VCPKG_BUILD_TYPE)
  set(VCPKG_BUILD_TYPE
      "${LOWERCASE_BUILD_TYPE}"
      CACHE STRING "vcpkg build type (release|debug)")
endif()

if(NOT DEFINED VCPKG_LIBRARY_LINKAGE)
  if(PROJECT_DEPENDENCY_USE_SHARED)
    set(VCPKG_LIBRARY_LINKAGE "dynamic")
  else()
    set(VCPKG_LIBRARY_LINKAGE "static")
  endif()
  set(VCPKG_LIBRARY_LINKAGE
      "${VCPKG_LIBRARY_LINKAGE}"
      CACHE STRING "vcpkg preferred library linkage (static|dynamic)")
endif()

message(STATUS "Using vcpkg installed libraries directory: ${_VCPKG_INSTALLED_DIR}")
message(STATUS "Using VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "Using VCPKG_BUILD_TYPE: ${VCPKG_BUILD_TYPE}")
message(STATUS "Using VCPKG_LIBRARY_LINKAGE: ${VCPKG_LIBRARY_LINKAGE}")

# cpp-2021-vulkan
