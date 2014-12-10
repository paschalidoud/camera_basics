# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vchoutas/Programming/camera_basics

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vchoutas/Programming/camera_basics

# Include any dependencies generated for this target.
include CMakeFiles/backprojection.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/backprojection.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/backprojection.out.dir/flags.make

CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o: CMakeFiles/backprojection.out.dir/flags.make
CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o: histograms/backprojection.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/vchoutas/Programming/camera_basics/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o -c /home/vchoutas/Programming/camera_basics/histograms/backprojection.cpp

CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/vchoutas/Programming/camera_basics/histograms/backprojection.cpp > CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.i

CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/vchoutas/Programming/camera_basics/histograms/backprojection.cpp -o CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.s

CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.requires:
.PHONY : CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.requires

CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.provides: CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.requires
	$(MAKE) -f CMakeFiles/backprojection.out.dir/build.make CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.provides.build
.PHONY : CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.provides

CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.provides.build: CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o

# Object files for target backprojection.out
backprojection_out_OBJECTS = \
"CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o"

# External object files for target backprojection.out
backprojection_out_EXTERNAL_OBJECTS =

backprojection.out: CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o
backprojection.out: /opt/ros/hydro/lib/libopencv_videostab.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_ts.a
backprojection.out: /opt/ros/hydro/lib/libopencv_superres.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_stitching.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_contrib.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
backprojection.out: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
backprojection.out: CMakeFiles/backprojection.out.dir/build.make
backprojection.out: CMakeFiles/backprojection.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable backprojection.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/backprojection.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/backprojection.out.dir/build: backprojection.out
.PHONY : CMakeFiles/backprojection.out.dir/build

CMakeFiles/backprojection.out.dir/requires: CMakeFiles/backprojection.out.dir/histograms/backprojection.cpp.o.requires
.PHONY : CMakeFiles/backprojection.out.dir/requires

CMakeFiles/backprojection.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/backprojection.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/backprojection.out.dir/clean

CMakeFiles/backprojection.out.dir/depend:
	cd /home/vchoutas/Programming/camera_basics && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics/CMakeFiles/backprojection.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/backprojection.out.dir/depend

