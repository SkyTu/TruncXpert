# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chenzan/Wing/ext/sytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chenzan/Wing/ext/sytorch

# Include any dependencies generated for this target.
include ext/sci/src/CMakeFiles/SCI-FloatML.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ext/sci/src/CMakeFiles/SCI-FloatML.dir/compiler_depend.make

# Include the progress variables for this target.
include ext/sci/src/CMakeFiles/SCI-FloatML.dir/progress.make

# Include the compile flags for this target's objects.
include ext/sci/src/CMakeFiles/SCI-FloatML.dir/flags.make

ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.o: ext/sci/src/CMakeFiles/SCI-FloatML.dir/flags.make
ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.o: ext/sci/src/library_float.cpp
ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.o: ext/sci/src/CMakeFiles/SCI-FloatML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenzan/Wing/ext/sytorch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.o"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.o -MF CMakeFiles/SCI-FloatML.dir/library_float.cpp.o.d -o CMakeFiles/SCI-FloatML.dir/library_float.cpp.o -c /home/chenzan/Wing/ext/sytorch/ext/sci/src/library_float.cpp

ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SCI-FloatML.dir/library_float.cpp.i"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenzan/Wing/ext/sytorch/ext/sci/src/library_float.cpp > CMakeFiles/SCI-FloatML.dir/library_float.cpp.i

ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SCI-FloatML.dir/library_float.cpp.s"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenzan/Wing/ext/sytorch/ext/sci/src/library_float.cpp -o CMakeFiles/SCI-FloatML.dir/library_float.cpp.s

ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o: ext/sci/src/CMakeFiles/SCI-FloatML.dir/flags.make
ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o: ext/sci/src/globals_float.cpp
ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o: ext/sci/src/CMakeFiles/SCI-FloatML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenzan/Wing/ext/sytorch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o -MF CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o.d -o CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o -c /home/chenzan/Wing/ext/sytorch/ext/sci/src/globals_float.cpp

ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SCI-FloatML.dir/globals_float.cpp.i"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenzan/Wing/ext/sytorch/ext/sci/src/globals_float.cpp > CMakeFiles/SCI-FloatML.dir/globals_float.cpp.i

ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SCI-FloatML.dir/globals_float.cpp.s"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenzan/Wing/ext/sytorch/ext/sci/src/globals_float.cpp -o CMakeFiles/SCI-FloatML.dir/globals_float.cpp.s

ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o: ext/sci/src/CMakeFiles/SCI-FloatML.dir/flags.make
ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o: ext/sci/src/cleartext_library_float.cpp
ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o: ext/sci/src/CMakeFiles/SCI-FloatML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenzan/Wing/ext/sytorch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o -MF CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o.d -o CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o -c /home/chenzan/Wing/ext/sytorch/ext/sci/src/cleartext_library_float.cpp

ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.i"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenzan/Wing/ext/sytorch/ext/sci/src/cleartext_library_float.cpp > CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.i

ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.s"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenzan/Wing/ext/sytorch/ext/sci/src/cleartext_library_float.cpp -o CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.s

# Object files for target SCI-FloatML
SCI__FloatML_OBJECTS = \
"CMakeFiles/SCI-FloatML.dir/library_float.cpp.o" \
"CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o" \
"CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o"

# External object files for target SCI-FloatML
SCI__FloatML_EXTERNAL_OBJECTS =

lib/libSCI-FloatML.a: ext/sci/src/CMakeFiles/SCI-FloatML.dir/library_float.cpp.o
lib/libSCI-FloatML.a: ext/sci/src/CMakeFiles/SCI-FloatML.dir/globals_float.cpp.o
lib/libSCI-FloatML.a: ext/sci/src/CMakeFiles/SCI-FloatML.dir/cleartext_library_float.cpp.o
lib/libSCI-FloatML.a: ext/sci/src/CMakeFiles/SCI-FloatML.dir/build.make
lib/libSCI-FloatML.a: ext/sci/src/CMakeFiles/SCI-FloatML.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenzan/Wing/ext/sytorch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library ../../../lib/libSCI-FloatML.a"
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && $(CMAKE_COMMAND) -P CMakeFiles/SCI-FloatML.dir/cmake_clean_target.cmake
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SCI-FloatML.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ext/sci/src/CMakeFiles/SCI-FloatML.dir/build: lib/libSCI-FloatML.a
.PHONY : ext/sci/src/CMakeFiles/SCI-FloatML.dir/build

ext/sci/src/CMakeFiles/SCI-FloatML.dir/clean:
	cd /home/chenzan/Wing/ext/sytorch/ext/sci/src && $(CMAKE_COMMAND) -P CMakeFiles/SCI-FloatML.dir/cmake_clean.cmake
.PHONY : ext/sci/src/CMakeFiles/SCI-FloatML.dir/clean

ext/sci/src/CMakeFiles/SCI-FloatML.dir/depend:
	cd /home/chenzan/Wing/ext/sytorch && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenzan/Wing/ext/sytorch /home/chenzan/Wing/ext/sytorch/ext/sci/src /home/chenzan/Wing/ext/sytorch /home/chenzan/Wing/ext/sytorch/ext/sci/src /home/chenzan/Wing/ext/sytorch/ext/sci/src/CMakeFiles/SCI-FloatML.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ext/sci/src/CMakeFiles/SCI-FloatML.dir/depend

