cmake_minimum_required(VERSION 2.8)

set (EXEC yolov3)

project (${EXEC})

#option(WITH_OPENCV "this is a default for opencv" OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
endif()


find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(	
	${CMAKE_SOURCE_DIR} 
	/usr/include/x86_64-linux-gnu
	/usr/include
	/usr/local/include
)


set(CMAKE_CXX_FLAGS "-DOPENCV ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-DOPENCV ${CMAKE_C_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "-w -g -O0 -std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "-w -g -O0 ${CMAKE_C_FLAGS}")

set(CMAKE_CXX_FLAGS_RELEASE "-w -O3 -std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-w -O3 ${CMAKE_C_FLAGS}")


file(GLOB C_LIST "darknet/src/*.c")
list(REMOVE_ITEM C_LIST "${CMAKE_SOURCE_DIR}/darknet/src/gettimeofday.c")
list(REMOVE_ITEM C_LIST "${CMAKE_SOURCE_DIR}/darknet/src/darknet.c")

file(GLOB CPP_LIST "darknet/src/*.cpp")
list(REMOVE_ITEM CPP_LIST "${CMAKE_SOURCE_DIR}/darknet/src/yolo_console_dll.cpp")
list(REMOVE_ITEM CPP_LIST "${CMAKE_SOURCE_DIR}/darknet/src/yolo_v2_class.cpp")

#file(GLOB SRC_LIST "darknet/src/*.cu")

set(SRC_LIST 
   ${C_LIST} 
   ${CPP_LIST}
   wrapper/detector.cpp)

add_executable(${EXEC} ${SRC_LIST} ./yolo_console_dll.cpp)

target_link_libraries(${EXEC}  ${OpenCV_LIBS} X11 pthread dl)


