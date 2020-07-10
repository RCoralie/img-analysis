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

################################################################################
#INCLUDES

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    /opt/include/opencv4/
)

include(CTest)

################################################################################
#SOURCES

file(GLOB_RECURSE SRC_REGISTRATION "src/Filter/*.cpp" "src/ImageRegistration/*.cpp")

################################################################################
#LINK

LIST(APPEND LINK_LIBS
   ${OpenCV_LIBS}
   ${YAML_CPP_LIBRARIES}
   ${Boost_SYSTEM_LIBRARY}
   ${Boost_FILESYSTEM_LIBRARY}
   ${Boost_PROGRAM_OPTIONS_LIBRARY}
   ${Boost_REGEX_LIBRARY}
)

LIST(APPEND LINK_TEST_LIBS
   ${CONAN_LIBS}
   ${OpenCV_LIBS}
   gtest_main
)

link_libraries(stdc++fs)

################################################################################
#EXECUTABLE

add_executable(Demo_RegistrationFB ${SRC_REGISTRATION} src/Demo_ImageRegistration_FeaturesBased.cpp)
target_link_libraries(Demo_RegistrationFB ${LINK_LIBS})

add_executable(Demo_RegistrationECC ${SRC_REGISTRATION} src/Demo_ImageRegistration_ECC.cpp)
target_link_libraries(Demo_RegistrationECC ${LINK_LIBS})

add_executable(Demo_RegistrationFMT ${SRC_REGISTRATION} src/Demo_ImageRegistration_FMT.cpp)
target_link_libraries(Demo_RegistrationFMT ${LINK_LIBS})

add_executable(Compare_ShiftRegistration ${SRC_REGISTRATION} src/Compare_ShiftRegistration.cpp)
target_link_libraries(Compare_ShiftRegistration ${LINK_LIBS})
