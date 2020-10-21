cmake_minimum_required(VERSION 3.17)

project(SoftbodySimulation 
	VERSION 0.0.1
	LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_PREFIX_PATH "C:\\Qt\\5.15.0\\msvc2019_64\\")
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

add_subdirectory(SoftbodyPhysics)
add_subdirectory(SoftbodySimulation)