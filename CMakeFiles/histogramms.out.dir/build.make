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
include CMakeFiles/histogramms.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/histogramms.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/histogramms.out.dir/flags.make

CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o: CMakeFiles/histogramms.out.dir/flags.make
CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o: histograms/histogramms.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/vchoutas/Programming/camera_basics/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o -c /home/vchoutas/Programming/camera_basics/histograms/histogramms.cpp

CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/vchoutas/Programming/camera_basics/histograms/histogramms.cpp > CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.i

CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/vchoutas/Programming/camera_basics/histograms/histogramms.cpp -o CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.s

CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.requires:
.PHONY : CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.requires

CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.provides: CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.requires
	$(MAKE) -f CMakeFiles/histogramms.out.dir/build.make CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.provides.build
.PHONY : CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.provides

CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.provides.build: CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o

# Object files for target histogramms.out
histogramms_out_OBJECTS = \
"CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o"

# External object files for target histogramms.out
histogramms_out_EXTERNAL_OBJECTS =

histogramms.out: CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o
histogramms.out: /opt/ros/hydro/lib/libopencv_videostab.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_ts.a
histogramms.out: /opt/ros/hydro/lib/libopencv_superres.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_stitching.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_contrib.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
histogramms.out: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
histogramms.out: CMakeFiles/histogramms.out.dir/build.make
histogramms.out: CMakeFiles/histogramms.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable histogramms.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/histogramms.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/histogramms.out.dir/build: histogramms.out
.PHONY : CMakeFiles/histogramms.out.dir/build

CMakeFiles/histogramms.out.dir/requires: CMakeFiles/histogramms.out.dir/histograms/histogramms.cpp.o.requires
.PHONY : CMakeFiles/histogramms.out.dir/requires

CMakeFiles/histogramms.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/histogramms.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/histogramms.out.dir/clean

CMakeFiles/histogramms.out.dir/depend:
	cd /home/vchoutas/Programming/camera_basics && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics /home/vchoutas/Programming/camera_basics/CMakeFiles/histogramms.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/histogramms.out.dir/depend
