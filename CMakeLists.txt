cmake_minimum_required(VERSION 3.12)
project(ImageAnalysis)


add_definitions("-std=c++11")

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

################################################################################
#PACKAGES

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options regex)
find_package(GTest REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

################################################################################
#INCLUDES

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    /opt/include/opencv4/
    ${Qt5Widgets_INCLUDE_DIRS}
)

include(CTest)

################################################################################
#SOURCES

file(GLOB_RECURSE SRC_REGISTRATION "src/Shared/Filter/*.cpp" "src/ImageRegistration/*.cpp")
file(GLOB_RECURSE SRC_EDGE_DETECT "src/Shared/Filter/*.cpp" "src/EdgeDetector/*.cpp")
file(GLOB_RECURSE SRC_SHARED "src/Shared/*.cpp")

################################################################################
#LINK

LIST(APPEND LINK_LIBS
   ${OpenCV_LIBS}
   ${YAML_CPP_LIBRARIES}
   ${Boost_SYSTEM_LIBRARY}
   ${Boost_FILESYSTEM_LIBRARY}
   ${Boost_PROGRAM_OPTIONS_LIBRARY}
   ${Boost_REGEX_LIBRARY}
   ${Qt5Widgets_LIBRARIES}
)

LIST(APPEND LINK_TEST_LIBS
   ${CONAN_LIBS}
   ${OpenCV_LIBS}
   gtest_main
)

link_libraries(stdc++fs)

################################################################################
#EXECUTABLE
#
add_executable(Demo_Registration ${SRC_SHARED} ${SRC_REGISTRATION} ${SRC_EDGE_DETECT} src/Demo_Registration.cpp)
target_link_libraries(Demo_Registration ${LINK_LIBS})

add_executable(Demo_EdgeDetector ${SRC_EDGE_DETECT} src/Demo_EdgeDetector.cpp)
target_link_libraries(Demo_EdgeDetector ${LINK_LIBS})

add_executable(Demo_DFT ${SRC_SHARED} src/Demo_DFT.cpp)
target_link_libraries(Demo_DFT ${LINK_LIBS})
