# Install script for directory: /home/chenzan/Wing/ext/sytorch/ext/sci/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/chenzan/Wing/ext/sytorch/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-FloatML.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-OT.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-HE.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-FloatingPoint.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-BuildingBlocks.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-LinearOT.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-LinearHE.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-Math.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chenzan/Wing/ext/sytorch/lib/libSCI-GC.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets.cmake"
         "/home/chenzan/Wing/ext/sytorch/ext/sci/src/CMakeFiles/Export/lib/cmake/SCI/SCITargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI" TYPE FILE FILES "/home/chenzan/Wing/ext/sytorch/ext/sci/src/CMakeFiles/Export/lib/cmake/SCI/SCITargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI" TYPE FILE FILES "/home/chenzan/Wing/ext/sytorch/ext/sci/src/CMakeFiles/Export/lib/cmake/SCI/SCITargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/utils"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/OT"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/GC"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/Millionaire"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/NonLinear"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/BuildingBlocks"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/LinearOT"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/LinearHE"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/Math"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/FloatingPoint"
    FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/defines.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/defines_uniform.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/defines_float.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/globals.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/globals_float.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/library_fixed.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/library_fixed_uniform.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/library_float.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/cleartext_library_fixed.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/cleartext_library_fixed_uniform.h"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/cleartext_library_float.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI" TYPE FILE FILES
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/utils/cmake/FindGMP.cmake"
    "/home/chenzan/Wing/ext/sytorch/ext/sci/src/utils/cmake/source_of_randomness.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/utils/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/OT/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/GC/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/Millionaire/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/BuildingBlocks/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/LinearOT/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/LinearHE/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/NonLinear/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/Math/cmake_install.cmake")
  include("/home/chenzan/Wing/ext/sytorch/ext/sci/src/FloatingPoint/cmake_install.cmake")

endif()

