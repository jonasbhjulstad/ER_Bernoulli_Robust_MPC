# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.24

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\msys64\mingw64\bin\cmake.exe

# The command to remove a file.
RM = C:\msys64\mingw64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\jonas\Documents\Network_Robust_MPC\Cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\jonas\Documents\Network_Robust_MPC\build

# Include any dependencies generated for this target.
include test/CMakeFiles/monomials.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/monomials.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/monomials.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/monomials.dir/flags.make

test/CMakeFiles/monomials.dir/monomials.cpp.obj: test/CMakeFiles/monomials.dir/flags.make
test/CMakeFiles/monomials.dir/monomials.cpp.obj: test/CMakeFiles/monomials.dir/includes_CXX.rsp
test/CMakeFiles/monomials.dir/monomials.cpp.obj: C:/Users/jonas/Documents/Network_Robust_MPC/Cpp/test/monomials.cpp
test/CMakeFiles/monomials.dir/monomials.cpp.obj: test/CMakeFiles/monomials.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\jonas\Documents\Network_Robust_MPC\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/monomials.dir/monomials.cpp.obj"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\test && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/monomials.dir/monomials.cpp.obj -MF CMakeFiles\monomials.dir\monomials.cpp.obj.d -o CMakeFiles\monomials.dir\monomials.cpp.obj -c C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\test\monomials.cpp

test/CMakeFiles/monomials.dir/monomials.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/monomials.dir/monomials.cpp.i"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\test && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\test\monomials.cpp > CMakeFiles\monomials.dir\monomials.cpp.i

test/CMakeFiles/monomials.dir/monomials.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/monomials.dir/monomials.cpp.s"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\test && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\test\monomials.cpp -o CMakeFiles\monomials.dir\monomials.cpp.s

# Object files for target monomials
monomials_OBJECTS = \
"CMakeFiles/monomials.dir/monomials.cpp.obj"

# External object files for target monomials
monomials_EXTERNAL_OBJECTS =

test/monomials.exe: test/CMakeFiles/monomials.dir/monomials.cpp.obj
test/monomials.exe: test/CMakeFiles/monomials.dir/build.make
test/monomials.exe: C:/Users/jonas/Downloads/casadi/build/casadi.lib
test/monomials.exe: lib/libgtest_main.a
test/monomials.exe: C:/msys64/mingw64/lib/libigraph.dll.a
test/monomials.exe: static/libFROLS_DataFrame.a
test/monomials.exe: static/libFROLS_Quantiles.a
test/monomials.exe: static/libFROLS_Eigen.a
test/monomials.exe: static/Features/libFROLS_Features.a
test/monomials.exe: static/libFROLS_Math.a
test/monomials.exe: static/Algorithm/libFROLS_Algorithm.a
test/monomials.exe: lib/libgtest.a
test/monomials.exe: C:/msys64/mingw64/lib/libsundials_cvode.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libsundials_nvecserial.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libarpack.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libopenblas.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libcxsparse.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libglpk.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libgmp.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libopenblas.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libcxsparse.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libglpk.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libgmp.dll.a
test/monomials.exe: C:/msys64/mingw64/lib/libxml2.dll.a
test/monomials.exe: static/Features/libFROLS_Features.a
test/monomials.exe: static/Algorithm/libFROLS_Algorithm.a
test/monomials.exe: static/libFROLS_Eigen.a
test/monomials.exe: static/libFROLS_DataFrame.a
test/monomials.exe: static/libFROLS_Math.a
test/monomials.exe: test/CMakeFiles/monomials.dir/linklibs.rsp
test/monomials.exe: test/CMakeFiles/monomials.dir/objects1.rsp
test/monomials.exe: test/CMakeFiles/monomials.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\jonas\Documents\Network_Robust_MPC\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable monomials.exe"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\monomials.dir\link.txt --verbose=$(VERBOSE)
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\test && C:\msys64\mingw64\bin\cmake.exe -D TEST_TARGET=monomials -D TEST_EXECUTABLE=C:/Users/jonas/Documents/Network_Robust_MPC/build/test/monomials.exe -D TEST_EXECUTOR= -D TEST_WORKING_DIR=C:/Users/jonas/Documents/Network_Robust_MPC/build/test -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D TEST_FILTER= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=monomials_TESTS -D CTEST_FILE=C:/Users/jonas/Documents/Network_Robust_MPC/build/test/monomials[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=5 -D TEST_XML_OUTPUT_DIR= -P C:/msys64/mingw64/share/cmake/Modules/GoogleTestAddTests.cmake

# Rule to build all files generated by this target.
test/CMakeFiles/monomials.dir/build: test/monomials.exe
.PHONY : test/CMakeFiles/monomials.dir/build

test/CMakeFiles/monomials.dir/clean:
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\test && $(CMAKE_COMMAND) -P CMakeFiles\monomials.dir\cmake_clean.cmake
.PHONY : test/CMakeFiles/monomials.dir/clean

test/CMakeFiles/monomials.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\jonas\Documents\Network_Robust_MPC\Cpp C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\test C:\Users\jonas\Documents\Network_Robust_MPC\build C:\Users\jonas\Documents\Network_Robust_MPC\build\test C:\Users\jonas\Documents\Network_Robust_MPC\build\test\CMakeFiles\monomials.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/monomials.dir/depend
