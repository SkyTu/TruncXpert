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
include CMakeFiles/bert.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/bert.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bert.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bert.dir/flags.make

CMakeFiles/bert.dir/examples/bert.cpp.o: CMakeFiles/bert.dir/flags.make
CMakeFiles/bert.dir/examples/bert.cpp.o: examples/bert.cpp
CMakeFiles/bert.dir/examples/bert.cpp.o: CMakeFiles/bert.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenzan/Wing/ext/sytorch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bert.dir/examples/bert.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/bert.dir/examples/bert.cpp.o -MF CMakeFiles/bert.dir/examples/bert.cpp.o.d -o CMakeFiles/bert.dir/examples/bert.cpp.o -c /home/chenzan/Wing/ext/sytorch/examples/bert.cpp

CMakeFiles/bert.dir/examples/bert.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bert.dir/examples/bert.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenzan/Wing/ext/sytorch/examples/bert.cpp > CMakeFiles/bert.dir/examples/bert.cpp.i

CMakeFiles/bert.dir/examples/bert.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bert.dir/examples/bert.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenzan/Wing/ext/sytorch/examples/bert.cpp -o CMakeFiles/bert.dir/examples/bert.cpp.s

# Object files for target bert
bert_OBJECTS = \
"CMakeFiles/bert.dir/examples/bert.cpp.o"

# External object files for target bert
bert_EXTERNAL_OBJECTS =

bert: CMakeFiles/bert.dir/examples/bert.cpp.o
bert: CMakeFiles/bert.dir/build.make
bert: libsytorch.a
bert: /usr/local/cuda-12.4/lib64/libcudart.so
bert: ext/llama/libLLAMA.a
bert: ext/bitpack/libbitpack.a
bert: ext/cryptoTools/libcryptoTools.a
bert: lib/libSCI-FloatML.a
bert: lib/libSCI-FloatingPoint.a
bert: lib/libSCI-BuildingBlocks.a
bert: lib/libSCI-Math.a
bert: lib/libSCI-GC.a
bert: lib/libSCI-LinearOT.a
bert: lib/libSCI-OT.a
bert: lib/libSCI-BuildingBlocks.a
bert: lib/libSCI-Math.a
bert: lib/libSCI-GC.a
bert: lib/libSCI-LinearOT.a
bert: lib/libSCI-OT.a
bert: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
bert: /usr/lib/x86_64-linux-gnu/libpthread.a
bert: /usr/lib/x86_64-linux-gnu/libssl.so
bert: /usr/lib/x86_64-linux-gnu/libcrypto.so
bert: /usr/lib/x86_64-linux-gnu/libgmp.so
bert: CMakeFiles/bert.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenzan/Wing/ext/sytorch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bert"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bert.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bert.dir/build: bert
.PHONY : CMakeFiles/bert.dir/build

CMakeFiles/bert.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bert.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bert.dir/clean

CMakeFiles/bert.dir/depend:
	cd /home/chenzan/Wing/ext/sytorch && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenzan/Wing/ext/sytorch /home/chenzan/Wing/ext/sytorch /home/chenzan/Wing/ext/sytorch /home/chenzan/Wing/ext/sytorch /home/chenzan/Wing/ext/sytorch/CMakeFiles/bert.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bert.dir/depend

