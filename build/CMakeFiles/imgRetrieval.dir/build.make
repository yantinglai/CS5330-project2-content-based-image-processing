# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sundri/Desktop/CS5330/Project2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sundri/Desktop/CS5330/Project2/build

# Include any dependencies generated for this target.
include CMakeFiles/imgRetrieval.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/imgRetrieval.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/imgRetrieval.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imgRetrieval.dir/flags.make

CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o: CMakeFiles/imgRetrieval.dir/flags.make
CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o: /Users/sundri/Desktop/CS5330/Project2/imgRetrieval.cpp
CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o: CMakeFiles/imgRetrieval.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sundri/Desktop/CS5330/Project2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o -MF CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o.d -o CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o -c /Users/sundri/Desktop/CS5330/Project2/imgRetrieval.cpp

CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sundri/Desktop/CS5330/Project2/imgRetrieval.cpp > CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.i

CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sundri/Desktop/CS5330/Project2/imgRetrieval.cpp -o CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.s

CMakeFiles/imgRetrieval.dir/readfiles.cpp.o: CMakeFiles/imgRetrieval.dir/flags.make
CMakeFiles/imgRetrieval.dir/readfiles.cpp.o: /Users/sundri/Desktop/CS5330/Project2/readfiles.cpp
CMakeFiles/imgRetrieval.dir/readfiles.cpp.o: CMakeFiles/imgRetrieval.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sundri/Desktop/CS5330/Project2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/imgRetrieval.dir/readfiles.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/imgRetrieval.dir/readfiles.cpp.o -MF CMakeFiles/imgRetrieval.dir/readfiles.cpp.o.d -o CMakeFiles/imgRetrieval.dir/readfiles.cpp.o -c /Users/sundri/Desktop/CS5330/Project2/readfiles.cpp

CMakeFiles/imgRetrieval.dir/readfiles.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgRetrieval.dir/readfiles.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sundri/Desktop/CS5330/Project2/readfiles.cpp > CMakeFiles/imgRetrieval.dir/readfiles.cpp.i

CMakeFiles/imgRetrieval.dir/readfiles.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgRetrieval.dir/readfiles.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sundri/Desktop/CS5330/Project2/readfiles.cpp -o CMakeFiles/imgRetrieval.dir/readfiles.cpp.s

CMakeFiles/imgRetrieval.dir/features.cpp.o: CMakeFiles/imgRetrieval.dir/flags.make
CMakeFiles/imgRetrieval.dir/features.cpp.o: /Users/sundri/Desktop/CS5330/Project2/features.cpp
CMakeFiles/imgRetrieval.dir/features.cpp.o: CMakeFiles/imgRetrieval.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sundri/Desktop/CS5330/Project2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/imgRetrieval.dir/features.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/imgRetrieval.dir/features.cpp.o -MF CMakeFiles/imgRetrieval.dir/features.cpp.o.d -o CMakeFiles/imgRetrieval.dir/features.cpp.o -c /Users/sundri/Desktop/CS5330/Project2/features.cpp

CMakeFiles/imgRetrieval.dir/features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgRetrieval.dir/features.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sundri/Desktop/CS5330/Project2/features.cpp > CMakeFiles/imgRetrieval.dir/features.cpp.i

CMakeFiles/imgRetrieval.dir/features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgRetrieval.dir/features.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sundri/Desktop/CS5330/Project2/features.cpp -o CMakeFiles/imgRetrieval.dir/features.cpp.s

CMakeFiles/imgRetrieval.dir/csv_util.cpp.o: CMakeFiles/imgRetrieval.dir/flags.make
CMakeFiles/imgRetrieval.dir/csv_util.cpp.o: /Users/sundri/Desktop/CS5330/Project2/csv_util.cpp
CMakeFiles/imgRetrieval.dir/csv_util.cpp.o: CMakeFiles/imgRetrieval.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sundri/Desktop/CS5330/Project2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/imgRetrieval.dir/csv_util.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/imgRetrieval.dir/csv_util.cpp.o -MF CMakeFiles/imgRetrieval.dir/csv_util.cpp.o.d -o CMakeFiles/imgRetrieval.dir/csv_util.cpp.o -c /Users/sundri/Desktop/CS5330/Project2/csv_util.cpp

CMakeFiles/imgRetrieval.dir/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/imgRetrieval.dir/csv_util.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sundri/Desktop/CS5330/Project2/csv_util.cpp > CMakeFiles/imgRetrieval.dir/csv_util.cpp.i

CMakeFiles/imgRetrieval.dir/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/imgRetrieval.dir/csv_util.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sundri/Desktop/CS5330/Project2/csv_util.cpp -o CMakeFiles/imgRetrieval.dir/csv_util.cpp.s

# Object files for target imgRetrieval
imgRetrieval_OBJECTS = \
"CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o" \
"CMakeFiles/imgRetrieval.dir/readfiles.cpp.o" \
"CMakeFiles/imgRetrieval.dir/features.cpp.o" \
"CMakeFiles/imgRetrieval.dir/csv_util.cpp.o"

# External object files for target imgRetrieval
imgRetrieval_EXTERNAL_OBJECTS =

imgRetrieval: CMakeFiles/imgRetrieval.dir/imgRetrieval.cpp.o
imgRetrieval: CMakeFiles/imgRetrieval.dir/readfiles.cpp.o
imgRetrieval: CMakeFiles/imgRetrieval.dir/features.cpp.o
imgRetrieval: CMakeFiles/imgRetrieval.dir/csv_util.cpp.o
imgRetrieval: CMakeFiles/imgRetrieval.dir/build.make
imgRetrieval: /opt/homebrew/lib/libopencv_gapi.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_stitching.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_alphamat.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_aruco.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_bgsegm.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_bioinspired.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_ccalib.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_dnn_objdetect.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_dnn_superres.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_dpm.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_face.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_freetype.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_fuzzy.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_hfs.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_img_hash.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_intensity_transform.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_line_descriptor.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_mcc.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_quality.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_rapid.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_reg.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_rgbd.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_saliency.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_sfm.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_signal.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_stereo.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_structured_light.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_superres.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_surface_matching.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_tracking.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_videostab.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_viz.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_wechat_qrcode.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_xfeatures2d.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_xobjdetect.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_xphoto.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_shape.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_highgui.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_datasets.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_plot.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_text.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_ml.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_phase_unwrapping.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_optflow.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_ximgproc.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_video.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_videoio.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_imgcodecs.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_objdetect.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_calib3d.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_dnn.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_features2d.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_flann.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_photo.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_imgproc.4.10.0.dylib
imgRetrieval: /opt/homebrew/lib/libopencv_core.4.10.0.dylib
imgRetrieval: CMakeFiles/imgRetrieval.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/sundri/Desktop/CS5330/Project2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable imgRetrieval"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imgRetrieval.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imgRetrieval.dir/build: imgRetrieval
.PHONY : CMakeFiles/imgRetrieval.dir/build

CMakeFiles/imgRetrieval.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imgRetrieval.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imgRetrieval.dir/clean

CMakeFiles/imgRetrieval.dir/depend:
	cd /Users/sundri/Desktop/CS5330/Project2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sundri/Desktop/CS5330/Project2 /Users/sundri/Desktop/CS5330/Project2 /Users/sundri/Desktop/CS5330/Project2/build /Users/sundri/Desktop/CS5330/Project2/build /Users/sundri/Desktop/CS5330/Project2/build/CMakeFiles/imgRetrieval.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/imgRetrieval.dir/depend

